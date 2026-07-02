from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.utils import div_ceil, div_even, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig

_FP8 = torch.float8_e4m3fn
_BLOCK = [128, 128]

# sglang's triton fused_moe resolves tuned tile configs from
# $SGLANG_MOE_CONFIG_DIR/configs/triton_<ver>/<shape>.json; point it at the
# configs shipped in minisgl/moe (setdefault keeps user overrides working).
os.environ.setdefault(
    "SGLANG_MOE_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "moe"),
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    # GPT-J / interleaved rotation: pairs are (x[0],x[1]),(x[2],x[3]),...
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# MTP draft hidden source A/B: "pre" (pre-final-norm residual, DeepSeek paper
# convention, default) vs "post" (post-norm output)
_MTP_HIDDEN_POST = os.environ.get("MINISGL_MTP_HIDDEN", "pre") == "post"

_fused_rope = None


def _get_fused_rope(rope_dim: int, dtype: torch.dtype):
    # sglang's CUDA jit fused_rope applies GPT-J interleaved RoPE to q/k in place,
    # consuming positions + an fp32 cos/sin cache: one kernel replaces the whole
    # eager rope chain (casts/mul/add/stack, ~10 tiny kernels per layer) plus the
    # per-forward cos/sin computation. Compile eagerly so a broken toolchain falls
    # back to the eager path instead of crashing mid-forward.
    global _fused_rope
    if _fused_rope is None:
        try:
            from sglang.jit_kernel.rope import _jit_fused_rope_module, apply_rope_inplace

            _jit_fused_rope_module(False, rope_dim, dtype)
            _fused_rope = apply_rope_inplace
        except Exception:
            _fused_rope = False
    return _fused_rope


def _rope_cos_sin(positions: torch.Tensor, dim: int, base: float):
    # Compute GPT-J interleaved RoPE cos/sin ONCE per forward. positions are identical
    # across all layers, so this is shared instead of recomputed 78x (a big kernel-count
    # cut at bs=1, where the model is dominated by many tiny sequential kernels).
    # Returns cos, sin as [T, 1, dim], ready to broadcast over heads.
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = freqs.cos().repeat_interleave(2, dim=-1)[:, None, :]
    sin = freqs.sin().repeat_interleave(2, dim=-1)[:, None, :]
    return cos, sin


# --------------------------------------------------------------------------- #
# Block-FP8 linears (reuse sglang's triton w8a8 block kernel for sglang parity)
# --------------------------------------------------------------------------- #
def _fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear

    return triton_w8a8_block_fp8_linear(x, weight, _BLOCK, scale)


class Fp8LinearReplicated(BaseOP):
    def __init__(self, in_f: int, out_f: int):
        self.weight = torch.empty(out_f, in_f, dtype=_FP8)
        self.weight_scale_inv = torch.empty(div_ceil(out_f, 128), div_ceil(in_f, 128), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fp8_linear(x, self.weight, self.weight_scale_inv)


class Fp8LinearCol(BaseOP):
    """Column parallel: output dim sharded across TP."""

    def __init__(self, in_f: int, out_f: int):
        tp = get_tp_info().size
        local_out = div_even(out_f, tp)
        self.weight = torch.empty(local_out, in_f, dtype=_FP8)
        self.weight_scale_inv = torch.empty(div_ceil(local_out, 128), div_ceil(in_f, 128), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fp8_linear(x, self.weight, self.weight_scale_inv)


class Fp8LinearRow(BaseOP):
    """Row parallel: input dim sharded across TP, output all-reduced."""

    def __init__(self, in_f: int, out_f: int):
        tp = get_tp_info().size
        local_in = div_even(in_f, tp)
        self.weight = torch.empty(out_f, local_in, dtype=_FP8)
        self.weight_scale_inv = torch.empty(div_ceil(out_f, 128), div_ceil(local_in, 128), dtype=torch.float32)
        self._comm = DistributedCommunicator()
        self._tp = tp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _fp8_linear(x, self.weight, self.weight_scale_inv)
        return self._comm.all_reduce(y) if self._tp > 1 else y


# Only the routed experts (the ~96% of params) stay FP8; the MLA / dense / shared-expert
# projections run in bf16. The naive (non-absorbed) MLA does extra large GEMMs (kv_b, o_proj)
# whose per-layer FP8 quant error compounds over 78 layers far more than sglang's absorbed MLA,
# degrading generation; keeping them bf16 costs only a few GB/GPU and restores accuracy.
def _col(config, in_f, out_f):
    return LinearColParallelMerged(in_f, [out_f], has_bias=False)


def _repl(config, in_f, out_f):
    return LinearReplicated(in_f, out_f, has_bias=False)


def _orow(config, in_f, out_f):
    return LinearOProj(in_f, out_f, has_bias=False)


class Fp8GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig, intermediate_size: int):
        self.gate_up_proj = Fp8LinearCol(config.hidden_size, 2 * intermediate_size)
        self.down_proj = Fp8LinearRow(intermediate_size, config.hidden_size)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.layers import silu_and_mul

        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


def _gated_mlp(config, intermediate_size):
    return GatedMLP(config, intermediate_size=intermediate_size)


# --------------------------------------------------------------------------- #
# Attention (MLA: absorbed when the backend supports it, else naive materialized K/V)
# --------------------------------------------------------------------------- #
class GlmMLAAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp = get_tp_info().size
        self.layer_id = layer_id
        num_heads = config.num_qo_heads
        self.local_heads = div_even(num_heads, tp)
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_base = float(config.rotary_config.base)
        self.scale = self.qk_head_dim**-0.5

        self.q_a_proj = _repl(config, config.hidden_size, self.q_lora_rank)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = _col(config, self.q_lora_rank, num_heads * self.qk_head_dim)

        self.kv_a_proj_with_mqa = _repl(config, config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = _col(config, self.kv_lora_rank, num_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.o_proj = _orow(config, num_heads * self.v_head_dim, config.hidden_size)
        self.w_kc = None  # absorbed W_UK [H,qk_nope,kv_lora], built lazily on first forward
        self.w_vc = None  # absorbed W_UV^T [H,kv_lora,v_head]
        self._qkv_a_weight = None  # fused [q_a; kv_a] weight, built lazily on first forward

    def _fuse_qkv_a(self) -> None:
        # q_a_proj and kv_a_proj_with_mqa are both replicated GEMMs over the same
        # input; fuse their weights so one GEMM (and one read of x) replaces two.
        # Built lazily (after weight load, before cuda-graph capture); the original
        # op weights become views into the fused buffer so nothing is duplicated.
        wq = self.q_a_proj.weight
        wkv = self.kv_a_proj_with_mqa.weight
        self._qkv_a_weight = torch.cat([wq, wkv], dim=0).contiguous()
        self.q_a_proj.weight = self._qkv_a_weight[: wq.shape[0]]
        self.kv_a_proj_with_mqa.weight = self._qkv_a_weight[wq.shape[0] :]

    def _apply_rope(self, q_pe, k_pe, cos, sin):
        # GLM-5.2 uses INTERLEAVED (GPT-J) RoPE (config rope_interleave=true ->
        # sglang is_neox_style=False). cos/sin are precomputed once per forward.
        qf = q_pe.to(torch.float32)
        kf = k_pe.to(torch.float32)
        q_out = qf * cos + _rotate_gptj(qf) * sin
        k_out = kf * cos + _rotate_gptj(kf) * sin
        return q_out.to(q_pe.dtype), k_out.to(k_pe.dtype)

    def _build_absorb_weights(self) -> None:
        # Split kv_b_proj per head into W_UK (absorbed into q) and W_UV (absorbed into o)
        # so attention runs directly in the kv_lora latent space (sglang-style absorbed MLA).
        # kv_b_proj is bf16 here (only the routed experts stay fp8), so no dequant is needed.
        W = self.kv_b_proj.weight.view(
            self.local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        self.w_kc = W[:, : self.qk_nope_head_dim, :].contiguous()                  # [H,nope,lora]
        self.w_vc = W[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()  # [H,lora,vhead]

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]
        backend = ctx.attn_backend

        if self._qkv_a_weight is None:
            self._fuse_qkv_a()
        qkv_a = F.linear(x, self._qkv_a_weight)
        q_a, k_compressed, k_pe = qkv_a.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        q = self.q_b_proj.forward(self.q_a_layernorm.forward(q_a))
        q = q.view(T, self.local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        k_compressed = self.kv_a_layernorm.forward(k_compressed)

        k_pe = k_pe.reshape(T, 1, self.qk_rope_head_dim)
        if sin is None:
            # Fused-rope mode: `cos` carries the fp32 [max_pos, rope_dim] cos/sin
            # cache (see GlmMoeDsaModel.forward). The kernel rotates the strided
            # q/k slices in place — no contiguous copies needed.
            _get_fused_rope(self.qk_rope_head_dim, q_pe.dtype)(
                q_pe, k_pe, cos, ctx.batch.positions, is_neox=False
            )
        else:
            q_pe, k_pe = self._apply_rope(q_pe, k_pe, cos, sin)  # [T,H,rope], [T,1,rope]

        if hasattr(backend, "forward_mla"):
            # ---- absorbed MLA: attention in the kv_lora latent space ----
            if self.w_kc is None:
                self._build_absorb_weights()
            # bmm on transposed VIEWS: cublas strided-batch takes the layouts
            # directly, killing the einsum's permute materializations; the
            # downstream concat/store kernels all accept strided inputs.
            q_nope_latent = torch.bmm(
                q_nope.to(self.w_kc.dtype).transpose(0, 1), self.w_kc
            ).transpose(0, 1)
            o_latent = backend.forward_mla(
                q_nope_latent,
                q_pe,
                k_compressed,
                k_pe.squeeze(1),
                self.layer_id,
                ctx.batch,
            )  # [T,H,kv_lora]
            o = torch.einsum("thl,hlv->thv", o_latent.to(self.w_vc.dtype), self.w_vc)  # [T,H,vhead]
            return self.o_proj.forward(o.reshape(T, self.local_heads * self.v_head_dim))

        # ---- naive MLA: materialize per-head K/V ----
        kv = self.kv_b_proj.forward(k_compressed)
        kv = kv.view(T, self.local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe_h = k_pe.expand(T, self.local_heads, self.qk_rope_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1).contiguous()
        k = torch.cat([k_nope, k_pe_h], dim=-1).reshape(T, self.local_heads * self.qk_head_dim).contiguous()
        v = v.reshape(T, self.local_heads * self.v_head_dim).contiguous()
        o = backend.forward(q, k, v, self.layer_id, ctx.batch)
        o = o.reshape(T, self.local_heads * self.v_head_dim)
        return self.o_proj.forward(o)


# --------------------------------------------------------------------------- #
# MoE (DeepSeek noaux_tc gate + routed experts + shared expert)
# --------------------------------------------------------------------------- #
class GlmMoeGate(BaseOP):
    def __init__(self, config: ModelConfig):
        self.weight = torch.empty(config.num_experts, config.hidden_size)
        # MUST stay fp32: values are ~34 with <0.6 inter-expert spread; bf16 ULP at that
        # magnitude is 0.25 > the spread, which collapses expert ranking and destroys routing.
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import router_gemv

        # One CUDA GEMV kernel (block per expert, bf16 in / fp32 accum) replaces
        # cast-to-fp32 + cublas simt sgemm (~13us/layer at bs=4). fp32 logits
        # are still required (see the bias note above).
        return router_gemv(x, self.weight)


class Fp8Experts(BaseOP):
    def __init__(self, num_experts: int, hidden: int, inter_tp: int):
        self.gate_up_proj = torch.empty(num_experts, 2 * inter_tp, hidden, dtype=_FP8)
        self.gate_up_proj_scale_inv = torch.empty(
            num_experts, div_ceil(2 * inter_tp, 128), div_ceil(hidden, 128), dtype=torch.float32
        )
        self.down_proj = torch.empty(num_experts, hidden, inter_tp, dtype=_FP8)
        self.down_proj_scale_inv = torch.empty(
            num_experts, div_ceil(hidden, 128), div_ceil(inter_tp, 128), dtype=torch.float32
        )


class Fp4Experts(BaseOP):
    """NVFP4 (W4A4, group-16) routed experts, as exported by compressed-tensors
    checkpoints like nvidia/GLM-5.2-NVFP4. Weights are packed 2-per-byte with
    fp8 block scales and per-tensor global scales; the trtllm-gen FP4 MoE kernel
    consumes shuffled copies built lazily on first forward."""

    def __init__(self, num_experts: int, hidden: int, inter_tp: int):
        FP8 = torch.float8_e4m3fn
        self.gate_up_proj = torch.empty(num_experts, 2 * inter_tp, hidden // 2, dtype=torch.uint8)
        self.gate_up_proj_scale = torch.empty(num_experts, 2 * inter_tp, hidden // 16, dtype=FP8)
        self.gate_up_proj_gscale = torch.empty(num_experts, 2, dtype=torch.float32)
        self.gate_up_proj_in_gscale = torch.empty(num_experts, 2, dtype=torch.float32)
        self.down_proj = torch.empty(num_experts, hidden, inter_tp // 2, dtype=torch.uint8)
        self.down_proj_scale = torch.empty(num_experts, hidden, inter_tp // 16, dtype=FP8)
        self.down_proj_gscale = torch.empty(num_experts, 1, dtype=torch.float32)
        self.down_proj_in_gscale = torch.empty(num_experts, 1, dtype=torch.float32)


# activation global scales of the last quantized main layer, borrowed by the
# MTP layer's on-the-fly expert quantization (no calibration data of its own)
_last_fp4_act_scales = {}

_trtllm_moe_ok = None


_IN_MTP = False


def _use_custom_moe() -> bool:
    mode = os.environ.get("MINISGL_CUSTOM_MOE", "mtp")
    if mode == "1":
        return True
    if mode == "verify":  # custom kernel only in the main model
        return not _IN_MTP
    if mode == "mtp":  # custom kernel only in the MTP draft layer
        return _IN_MTP
    return False


def _use_trtllm_moe() -> bool:
    # flashinfer's trtllm-gen fused MoE (opt-in via MINISGL_TRTLLM_MOE=1).
    # Measured on 8xB300 vs the tuned triton path: only ~38us vs ~47us per layer
    # at M=4, and its intermediate quantization is ~3x noisier (cos-to-true
    # 0.9971 vs 0.9990), which visibly degrades long generations. Kept for
    # future FP4 / retuned-kernel experiments.
    global _trtllm_moe_ok
    if _trtllm_moe_ok is None:
        if os.environ.get("MINISGL_TRTLLM_MOE", "0") != "1":
            _trtllm_moe_ok = False
        else:
            try:
                from flashinfer.fused_moe import trtllm_fp8_block_scale_moe  # noqa: F401

                _trtllm_moe_ok = True
            except Exception:
                _trtllm_moe_ok = False
    return _trtllm_moe_ok


_fused_gate = None


def _get_fused_gate():
    # Newer sglang exports the CUDA kernel as moe_fused_gate_jit, older as
    # moe_fused_gate (same signature); support both.
    global _fused_gate
    if _fused_gate is None:
        try:
            import sglang.jit_kernel.moe_fused_gate as _mfg

            fn = getattr(_mfg, "moe_fused_gate_jit", None) or getattr(_mfg, "moe_fused_gate", None)
            can_use = getattr(_mfg, "can_use_moe_fused_gate", None)
            if fn is None:
                _fused_gate = False
            elif can_use is not None:
                _fused_gate = fn if can_use() else False
            else:
                _mfg._jit_moe_fused_gate_module()  # force JIT compile; raises if broken
                _fused_gate = fn
        except Exception:
            _fused_gate = False
    return _fused_gate


def _moe_route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.n_group == 1 and router_logits.dtype == torch.float32:
        # sglang's CUDA jit_kernel replaces the whole sigmoid/+bias/top-k/gather/
        # renorm/scale chain (~7 kernels, 15.8us) with one kernel (4.3us/layer,
        # cuda-graph timed). Weights match the python path to 1 ULP (bit-exact at
        # bs=1); ids identical.
        fused_gate = _get_fused_gate()
        if fused_gate:
            topk_w, topk_ids = fused_gate(
                router_logits,
                self.gate.e_score_correction_bias,
                self.top_k,
                renormalize=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=True,
            )
            return topk_ids, topk_w
    scores = router_logits.sigmoid()
    scores_for_choice = scores + self.gate.e_score_correction_bias.to(torch.float32)
    E, g = self.num_experts, self.n_group
    if g == 1:
        # n_group==1 (GLM-5.2): grouped-topk degenerates to plain top-k over all experts.
        # Skip the no-op group machinery (view/topk2/sum/topk/scatter/expand/masked_fill,
        # ~8 tiny kernels/layer) — bit-identical result, big kernel-count cut at bs=1.
        topk_ids = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
    else:
        group_scores = scores_for_choice.view(-1, g, E // g).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, g, E // g).reshape(-1, E)
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_ids = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]
    topk_w = scores.gather(1, topk_ids)
    if self.norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * self.routed_scaling_factor
    return topk_ids.to(torch.int32), topk_w.to(torch.float32)


class GlmSparseMoE(BaseOP):
    def __init__(self, config: ModelConfig, quantized: bool = True):
        tp = get_tp_info().size
        self.gate = GlmMoeGate(config)
        # fp8 checkpoints quantize every layer's experts (incl. MTP); modelopt
        # NVFP4 leaves the MTP layer's experts bf16, so only fp4 keys off `quantized`
        self.is_fp8 = config.is_fp8
        self.is_fp4 = config.is_fp4 and quantized
        self._fp4 = None  # trtllm-shuffled weights + alphas, built on first forward
        self._mtp_fp4_pending = (
            config.is_fp4
            and not quantized
            and os.environ.get("MINISGL_MTP_FP4", "1") == "1"
        )
        if self.is_fp4:
            self.experts = Fp4Experts(
                config.num_experts, config.hidden_size, div_even(config.moe_intermediate_size, tp)
            )
        elif config.is_fp8:
            self.experts = Fp8Experts(
                config.num_experts, config.hidden_size, div_even(config.moe_intermediate_size, tp)
            )
        else:
            self.experts = MoELayer(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                renormalize=config.norm_topk_prob,
            )
        self.shared_experts = _gated_mlp(config, config.n_shared_experts * config.moe_intermediate_size)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self._comm = DistributedCommunicator()
        self._tp = tp

    _route = _moe_route

    def _quantize_bf16_experts_to_fp4(self) -> None:
        from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

        e = self.experts  # bf16 MoELayer-style weights [E, 2I, H] / [E, H, I]
        E = self.num_experts
        FP4_MAX, FP8_MAX = 6.0, 448.0

        def quant(w):
            gs = (FP4_MAX * FP8_MAX) / w.float().abs().amax(dim=(1, 2))
            packed, scales = [], []
            for i in range(E):
                p, sc = fp4_quantize(w[i].to(torch.bfloat16), gs[i : i + 1], 16, False, False)
                packed.append(p)
                scales.append(sc.reshape(w.shape[1], -1))
            return torch.stack(packed), torch.stack(scales).view(torch.float8_e4m3fn), (1.0 / gs)

        gu_p, gu_s, gu_s2 = quant(e.gate_up_proj)
        dn_p, dn_s, dn_s2 = quant(e.down_proj)
        fp4 = Fp4Experts.__new__(Fp4Experts)
        fp4.gate_up_proj = gu_p
        fp4.gate_up_proj_scale = gu_s
        fp4.gate_up_proj_gscale = torch.stack([gu_s2, gu_s2], dim=1)
        fp4.gate_up_proj_in_gscale = _last_fp4_act_scales["in1"].reshape(1, 1).expand(E, 2)
        fp4.down_proj = dn_p
        fp4.down_proj_scale = dn_s
        fp4.down_proj_gscale = dn_s2.reshape(E, 1)
        fp4.down_proj_in_gscale = _last_fp4_act_scales["in2"].reshape(1, 1).expand(E, 1)
        self.experts = fp4
        self.is_fp4 = True

    def _prep_fp4(self) -> None:
        # Mirror sglang's compressed-tensors NVFP4 flashinfer-trtllm prep:
        # [gate;up] -> [up;gate] row order, invert global scales, precompute the
        # per-expert alpha chain, and shuffle weights/scales for the kernel.
        from sglang.srt.layers.quantization.utils import (
            prepare_static_weights_for_trtllm_fp4_moe,
            reorder_w1w3_to_w3w1,
        )

        e = self.experts
        E = self.num_experts
        hidden = e.down_proj.shape[1]
        inter = e.down_proj.shape[2] * 2
        w13, w13_s = reorder_w1w3_to_w3w1(e.gate_up_proj, e.gate_up_proj_scale, dim=-2)
        g1w, g1s, g2w, g2s = prepare_static_weights_for_trtllm_fp4_moe(
            w13, e.down_proj, w13_s, e.down_proj_scale, hidden, inter, E
        )
        # modelopt semantics: weight_scale_2 / input_scale are DEQUANT multipliers
        # (no inversion); the kernel's activation-quant scale is their inverse.
        w13_scale_2 = e.gate_up_proj_gscale[:, 0].float()
        w2_scale_2 = e.down_proj_gscale[:, 0].float()
        in1 = e.gate_up_proj_in_gscale.max().float()  # scalar, shared across experts
        in2 = e.down_proj_in_gscale.max().float()
        _last_fp4_act_scales["in1"] = in1
        _last_fp4_act_scales["in2"] = in2
        g1_alphas = (in1 * w13_scale_2).expand(E).contiguous()
        self._fp4 = {
            "g1w": g1w,
            "g1s": g1s.view(torch.float8_e4m3fn),
            "g2w": g2w,
            "g2s": g2s.view(torch.float8_e4m3fn),
            "g1_alphas": g1_alphas,
            "g2_alphas": (in2 * w2_scale_2).expand(E).contiguous(),
            "g1_scale_c": ((1.0 / in2) * g1_alphas).contiguous(),
            "act_scale": (1.0 / in1).reshape(1).contiguous(),
            "inter": inter,
        }
        # free the unshuffled originals
        e.gate_up_proj = e.gate_up_proj_scale = None
        e.down_proj = e.down_proj_scale = None

    def _trtllm_fp4_moe(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        from flashinfer import trtllm_fp4_block_scale_moe
        from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

        if self._fp4 is None:
            self._prep_fp4()
        p = self._fp4
        M, H = x.shape
        hb, sb = fp4_quantize(x, p["act_scale"], 16, False, False)
        out = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=self.gate.e_score_correction_bias,
            hidden_states=hb.reshape(M, H // 2),
            hidden_states_scale=sb.view(torch.float8_e4m3fn).reshape(*sb.shape[:-1], -1),
            gemm1_weights=p["g1w"],
            gemm1_weights_scale=p["g1s"],
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=p["g2w"],
            gemm2_weights_scale=p["g2s"],
            gemm2_bias=None,
            output1_scale_scalar=p["g1_scale_c"],
            output1_scale_gate_scalar=p["g1_alphas"],
            output2_scale_scalar=p["g2_alphas"],
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            intermediate_size=p["inter"],
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=2,  # DeepSeekV3: sigmoid + bias top-k, fp32 logits
            do_finalize=True,
        )[0]
        return out

    def _trtllm_moe(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # One trtllm-gen kernel fuses routing (DeepSeekV3 sigmoid+bias top-k),
        # both fp8 block-scale GEMMs, SwiGLU and the weighted finalize —
        # replacing the whole moe_fused_gate/quant/align/sort/gemm/act/sum chain
        # (~66us -> ~38us per layer at M=4 on B300).
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
        from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8

        aq, asf = per_token_group_quant_fp8(x, _BLOCK[1], column_major_scales=True)
        out = trtllm_fp8_block_scale_moe(
            router_logits,
            self.gate.e_score_correction_bias,
            aq,
            asf.t(),
            self.experts.gate_up_proj,
            self.experts.gate_up_proj_scale_inv,
            self.experts.down_proj,
            self.experts.down_proj_scale_inv,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=self.n_group,
            topk_group=self.topk_group,
            intermediate_size=self.experts.down_proj.shape[2],
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=2,  # DeepSeekV3: sigmoid + bias grouped top-k
            norm_topk_prob=self.norm_topk_prob,
        )
        return out[0] if isinstance(out, (list, tuple)) else out

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        router_logits = self.gate.forward(x)
        # Shared-expert partial WITHOUT its row-parallel all-reduce: since
        # all-reduce is linear, summing routed+shared partials first and reducing
        # once is equivalent and saves one all-reduce per MoE layer.
        se = self.shared_experts
        shared = F.linear(se.act_fn(se.gate_up_proj.forward(x)), se.down_proj.weight)

        if (
            not self.is_fp4
            and self._mtp_fp4_pending
            and _last_fp4_act_scales
        ):
            self._mtp_fp4_pending = False
            self._quantize_bf16_experts_to_fp4()
        if self.is_fp4:
            routed = self._trtllm_fp4_moe(x.contiguous(), router_logits)
        elif self.is_fp8 and _use_trtllm_moe():
            routed = self._trtllm_moe(x.contiguous(), router_logits)
        elif self.is_fp8 and _use_custom_moe() and num_tokens <= 16:
            from minisgl.kernel.moe_decode import moe_decode

            topk_ids, topk_w = self._route(router_logits)
            routed = moe_decode(
                x.contiguous(),
                self.experts.gate_up_proj,
                self.experts.gate_up_proj_scale_inv,
                self.experts.down_proj,
                self.experts.down_proj_scale_inv,
                topk_ids,
                topk_w,
            )
            if os.environ.get("MINISGL_MOE_XCHECK", "0") == "1" and not torch.cuda.is_current_stream_capturing():
                from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                    fused_experts_impl,
                )

                ref = fused_experts_impl(
                    x.contiguous(),
                    self.experts.gate_up_proj,
                    self.experts.down_proj,
                    topk_w,
                    topk_ids,
                    inplace=False,
                    use_fp8_w8a8=True,
                    w1_scale=self.experts.gate_up_proj_scale_inv,
                    w2_scale=self.experts.down_proj_scale_inv,
                    block_shape=_BLOCK,
                )
                rel = (routed.float() - ref.float()).abs().max() / (
                    ref.float().abs().max() + 1e-9
                )
                from minisgl.utils import init_logger

                init_logger(__name__).info_rank0(
                    f"[xcheck] M={num_tokens} rel={rel.item():.3e} "
                    f"|mine|={routed.float().abs().max().item():.3e} |ref|={ref.float().abs().max().item():.3e}"
                )
        elif self.is_fp8:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

            topk_ids, topk_w = self._route(router_logits)
            routed = fused_experts_impl(
                x.contiguous(),
                self.experts.gate_up_proj,
                self.experts.down_proj,
                topk_w,
                topk_ids,
                inplace=False,
                use_fp8_w8a8=True,
                w1_scale=self.experts.gate_up_proj_scale_inv,
                w2_scale=self.experts.down_proj_scale_inv,
                block_shape=_BLOCK,
            )
        else:
            from minisgl.moe.fused import fused_experts_impl

            topk_ids, topk_w = self._route(router_logits)
            routed = fused_experts_impl(
                x.contiguous(),
                self.experts.gate_up_proj,
                self.experts.down_proj,
                topk_w,
                topk_ids,
                activation="silu",
                apply_router_weight_on_input=False,
            )
        out = routed + shared
        if self._tp > 1:
            out = self._comm.all_reduce(out)
        return out.view(num_tokens, hidden_dim)


class GlmMoeDsaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = GlmMLAAttention(config, layer_id)
        if layer_id < config.first_k_dense_replace:
            self.mlp = _gated_mlp(config, config.intermediate_size)
        else:
            # modelopt NVFP4 checkpoints leave the MTP/NextN layer's experts
            # unquantized (bf16), so quantization is per-layer
            self.mlp = GlmSparseMoE(config, quantized=layer_id < config.num_layers)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x, cos, sin, residual=None):
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x, cos, sin)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GlmMoeDsaMTP(BaseOP):
    """MTP / NextN draft layer (checkpoint layer index == num_layers): predicts
    token i+2 from token i+1's embedding and token i's pre-final-norm hidden.
    eh_proj(cat(enorm(emb), hnorm(prev_hidden))) -> decoder (MLA+MoE, own KV slot)
    -> shared_head norm; logits come from the shared main lm_head."""

    def __init__(self, config: ModelConfig):
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.decoder = GlmMoeDsaDecoderLayer(config, config.num_layers)
        self.shared_head_norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, emb: torch.Tensor, prev_hidden: torch.Tensor, cos, sin
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global _IN_MTP
        _IN_MTP = True
        try:
            h = self.eh_proj.forward(
                torch.cat([self.enorm.forward(emb), self.hnorm.forward(prev_hidden)], dim=-1)
            )
            x, residual = self.decoder.forward(h, cos, sin, None)
            normed, residual = self.shared_head_norm.forward(x, residual)
        finally:
            _IN_MTP = False
        return normed, (normed if _MTP_HIDDEN_POST else residual)


class GlmMoeDsaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [GlmMoeDsaDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.mtp = GlmMoeDsaMTP(config) if config.num_nextn > 0 else None
        self._last_hidden: torch.Tensor | None = None  # pre-norm residual, for MTP
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._rope_dim = config.qk_rope_head_dim
        self._rope_base = float(config.rotary_config.base)
        self._rope_max_position = config.rotary_config.max_position
        self._cos_sin_cache: torch.Tensor | None = None

    def _rope_cache(self, device: torch.device) -> torch.Tensor:
        # fp32 [P, rope_dim] cache for the fused rope kernel: first half cos, second
        # half sin, NON-interleaved frequencies (the kernel pairs (2i, 2i+1) itself).
        # Sized by the engine's real max seq len (page_table cols), not the model's
        # nominal max_position (1M for GLM-5.2, which would be a 256MB buffer).
        if self._cos_sin_cache is None:
            P = min(int(get_global_ctx().page_table.shape[1]) + 1, self._rope_max_position)
            inv_freq = 1.0 / (
                self._rope_base
                ** (
                    torch.arange(0, self._rope_dim, 2, dtype=torch.float32, device=device)
                    / self._rope_dim
                )
            )
            freqs = torch.outer(torch.arange(P, dtype=torch.float32, device=device), inv_freq)
            self._cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
        return self._cos_sin_cache

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        # Fused rope passes the fp32 cache via the `cos` slot with sin=None as the
        # sentinel; the python fallback gets per-forward cos/sin instead.
        cos, sin = self.rope_args(input_ids.device)
        residual = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, cos, sin, residual)
        out, self._last_hidden = self.norm.forward(x, residual)
        if _MTP_HIDDEN_POST:
            self._last_hidden = out
        return out

    def rope_args(self, device: torch.device):
        if _get_fused_rope(self._rope_dim, torch.bfloat16):
            return self._rope_cache(device), None
        positions = get_global_ctx().batch.positions
        return _rope_cos_sin(positions, self._rope_dim, self._rope_base)


class GlmMoeDsaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = GlmMoeDsaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def forward_mtp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run only the MTP/NextN draft layer. Reads input_ids (the NEXT-position
        tokens) and spec_prev_hidden from the active batch; returns (logits, the
        draft layer's pre-norm residual for chaining further draft steps)."""
        batch = get_global_ctx().batch
        model = self.model
        emb = model.embed_tokens.forward(batch.input_ids)
        cos, sin = model.rope_args(batch.input_ids.device)
        normed, hidden = model.mtp.forward(emb, batch.spec_prev_hidden, cos, sin)
        return self.lm_head.forward(normed), hidden


__all__ = ["GlmMoeDsaForCausalLM"]
