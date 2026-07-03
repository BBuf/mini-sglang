from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

_MAX_M = 8


@functools.cache
def _jit_router_gemv_module(
    max_m: int = _MAX_M,
    num_threads: int = 128,
    use_pdl: bool = False,
) -> Module:
    args = make_cpp_args(max_m, num_threads, use_pdl)
    return load_jit(
        "router_gemv",
        *args,
        cuda_files=["router_gemv.cu"],
        cuda_wrappers=[("launch", f"RouterGemvKernel<{args}>::run")],
    )


def router_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 router logits = x[M,K](bf16) @ w[E,K](bf16)^T with fp32 accumulation
    (one thread block per expert). Matches an fp32 GEMM over pre-cast operands
    up to accumulation order. Arbitrary M via <=_MAX_M row chunks (decode M is
    tiny; prefill just launches a few more blocks-worth)."""
    import torch

    M = x.shape[0]
    out = torch.empty(M, w.shape[0], dtype=torch.float32, device=x.device)
    module = _jit_router_gemv_module()
    for i in range(0, M, _MAX_M):
        j = min(i + _MAX_M, M)
        module.launch(x[i:j], w, out[i:j])
    return out
