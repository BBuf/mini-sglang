from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _jit_moe_glue_module(threads: int = 256) -> Module:
    args = make_cpp_args(threads)
    return load_jit(
        "moe_glue",
        *args,
        cuda_files=["moe_glue.cu"],
        cuda_wrappers=[
            ("align", f"MoeGlueKernel<{args}>::align"),
            ("sum_add", f"MoeGlueKernel<{args}>::sum_add"),
        ],
    )


def tiny_align(
    topk_ids: torch.Tensor, block_m: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch moe_align_block_size for decode shapes (numel <= 256).

    Matches the sgl contract: sorted ids are indices into the flattened
    topk_ids, padded with numel; expert_ids has one entry per BLOCK_SIZE_M
    block; only [0, num_tokens_post_padded) of either is meaningful.
    """
    import torch

    numel = topk_ids.numel()
    sorted_ids = torch.empty(numel * block_m, dtype=torch.int32, device=topk_ids.device)
    expert_blocks = torch.empty(numel, dtype=torch.int32, device=topk_ids.device)
    total = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    _jit_moe_glue_module().align(topk_ids, sorted_ids, expert_blocks, total, num_experts, block_m)
    return sorted_ids, expert_blocks, total


def fused_experts_fp8_decode(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    """Triton block-fp8 MoE with a single-launch tiny_align replacing
    moe_align_block_size. Everything downstream (gemm mul_routed, combine)
    stays bitwise-identical to fused_experts_impl: the deep MTP chain's
    full-accept mode dies on ANY main-model numerics change, even the
    ~1e-3 bf16 rounding shift from moving the routed-weight multiply."""
    import torch
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _fused_moe_kernel_sequence,
        try_get_optimal_moe_config,
    )

    M, H = x.shape
    E = w13.shape[0]
    K = topk_ids.shape[1]
    config, (down_config, _) = try_get_optimal_moe_config(
        w13.shape,
        w2.shape,
        K,
        "fp8_w8a8",
        M,
        block_shape=[128, 128],
        return_down_config=True,
    )
    if down_config is not None:
        down_config = dict(down_config)
        assert not down_config.pop("USE_TMA", False), "TMA down path not supported"

    import os

    if os.environ.get("MINISGL_GLUE_ALIGN", "1") == "1":
        sorted_ids, expert_blocks, total = tiny_align(topk_ids, config["BLOCK_SIZE_M"], E)
    else:
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        sorted_ids, expert_blocks, total = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E
        )
    c3 = _fused_moe_kernel_sequence(
        x,
        w13,
        w2,
        topk_w,
        topk_ids,
        sorted_ids,
        expert_blocks,
        total,
        config,
        down_config,
        False,
        b1=None,
        b2=None,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=[128, 128],
        activation="silu",
        is_gated=True,
        no_combine=False,
        inplace=False,
        apply_router_weight_on_input=False,
        routed_scaling_factor=None,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=True,
    )
    return c3 + shared  # c3 is the combined routed output under no_combine=False
