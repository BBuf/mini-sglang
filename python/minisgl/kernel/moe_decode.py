from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _jit_moe_decode_module(
    threads1: int = 256,
    slice_i: int = 16,
    threads2: int = 128,
    slice_h: int = 64,
) -> Module:
    args = make_cpp_args(threads1, slice_i, threads2, slice_h)
    return load_jit(
        "moe_decode",
        *args,
        cuda_files=["moe_decode.cu"],
        cuda_wrappers=[("launch", f"MoeDecodeKernel<{args}>::run")],
    )


def moe_decode(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    gu_scale: torch.Tensor,
    down: torch.Tensor,
    dn_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused block-fp8 MoE for decode shapes (M <= ~16), bf16 activations.

    gate_up: [E, 2I, H] fp8e4m3 (w1w3 order), down: [E, H, I] fp8e4m3,
    scales fp32 with [128,128] blocks. Weights are passed as uint8 views to
    dodge dlpack fp8 dtype gaps.
    """
    import torch

    M, H = x.shape
    K = topk_ids.shape[1]
    I = down.shape[2]
    E = down.shape[0]
    assert gate_up.shape == (E, 2 * I, H) and gate_up.is_contiguous(), gate_up.shape
    assert down.is_contiguous() and x.is_contiguous()
    assert gu_scale.shape == (E, 2 * I // 128, H // 128) and gu_scale.is_contiguous(), (
        gu_scale.shape,
        gu_scale.stride(),
    )
    assert dn_scale.shape == (E, H // 128, I // 128) and dn_scale.is_contiguous(), (
        dn_scale.shape,
        dn_scale.stride(),
    )
    assert gu_scale.dtype == torch.float32 and dn_scale.dtype == torch.float32, gu_scale.dtype
    assert topk_ids.dtype == torch.int32 and topk_w.dtype == torch.float32
    inter = torch.empty(M, K, I, dtype=x.dtype, device=x.device)
    if out is None:
        out = torch.empty(M, H, dtype=x.dtype, device=x.device)
    module = _jit_moe_decode_module()
    module.launch(
        x,
        gate_up.view(torch.uint8),
        gu_scale,
        down.view(torch.uint8),
        dn_scale,
        topk_ids,
        topk_w,
        inter,
        out,
    )
    return out
