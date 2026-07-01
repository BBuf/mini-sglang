from __future__ import annotations

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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    # GPT-J / interleaved rotation: pairs are (x[0],x[1]),(x[2],x[3]),...
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


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

        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(T, self.local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        ckv_all = self.kv_a_proj_with_mqa.forward(x)
        k_compressed, k_pe = ckv_all.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm.forward(k_compressed.contiguous())

        k_pe = k_pe.reshape(T, 1, self.qk_rope_head_dim)
        q_pe, k_pe = self._apply_rope(q_pe, k_pe, cos, sin)  # q_pe [T,H,rope], k_pe [T,1,rope]

        if hasattr(backend, "forward_mla"):
            # ---- absorbed MLA: attention in the kv_lora latent space ----
            if self.w_kc is None:
                self._build_absorb_weights()
            q_nope_latent = torch.einsum("thn,hnl->thl", q_nope.to(self.w_kc.dtype), self.w_kc)
            o_latent = backend.forward_mla(
                q_nope_latent.contiguous(),
                q_pe.contiguous(),
                k_compressed,
                k_pe.squeeze(1).contiguous(),
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
        return F.linear(x.to(torch.float32), self.weight.to(torch.float32))


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


def _moe_route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = router_logits.sigmoid()
    scores_for_choice = scores + self.gate.e_score_correction_bias.to(torch.float32)
    E, g = self.num_experts, self.n_group
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
    def __init__(self, config: ModelConfig):
        tp = get_tp_info().size
        self.gate = GlmMoeGate(config)
        self.is_fp8 = config.is_fp8
        if config.is_fp8:
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

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        topk_ids, topk_w = self._route(self.gate.forward(x))
        shared = self.shared_experts.forward(x)

        if self.is_fp8:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

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

            routed = fused_experts_impl(
                x.contiguous(),
                self.experts.gate_up_proj,
                self.experts.down_proj,
                topk_w,
                topk_ids,
                activation="silu",
                apply_router_weight_on_input=False,
            )
        if self._tp > 1:
            routed = self._comm.all_reduce(routed)
        return (routed + shared).view(num_tokens, hidden_dim)


class GlmMoeDsaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = GlmMLAAttention(config, layer_id)
        if layer_id < config.first_k_dense_replace:
            self.mlp = _gated_mlp(config, config.intermediate_size)
        else:
            self.mlp = GlmSparseMoE(config)
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


class GlmMoeDsaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [GlmMoeDsaDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._rope_dim = config.qk_rope_head_dim
        self._rope_base = float(config.rotary_config.base)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        # RoPE cos/sin computed ONCE per forward, shared across all layers.
        cos, sin = _rope_cos_sin(get_global_ctx().batch.positions, self._rope_dim, self._rope_base)
        residual = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, cos, sin, residual)
        return self.norm.forward(x, residual)[0]


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


__all__ = ["GlmMoeDsaForCausalLM"]
