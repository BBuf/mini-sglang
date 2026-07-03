from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

_MAX_M = 8

# (K, N) -> (rows_per_block, num_threads), from the shape sweep on B300; shapes
# not listed fall back to a block-count heuristic.
_TUNED: dict = {}


@functools.cache
def _jit_skinny_gemv_module(
    rows: int,
    threads: int = 128,
    max_m: int = _MAX_M,
    use_pdl: bool = False,
) -> Module:
    args = make_cpp_args(rows, threads, max_m, use_pdl)
    return load_jit(
        "skinny_gemv",
        *args,
        cuda_files=["skinny_gemv.cu"],
        cuda_wrappers=[("launch", f"SkinnyGemvKernel<{args}>::run")],
    )


def _pick_config(K: int, N: int) -> tuple:
    cfg = _TUNED.get((K, N))
    if cfg is not None:
        return cfg
    # heuristic: enough blocks to fill ~2x148 SMs, but keep >= 4KB of W per block
    rows = 1
    while N // (rows * 2) > 296 and rows < 16:
        rows *= 2
    return rows, 128


def skinny_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    rows: int | None = None,
    threads: int = 128,
) -> torch.Tensor:
    """out[M,N] = x[M,K](bf16) @ w[N,K](bf16)^T with fp32 accumulation, M <= 8."""
    import torch

    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if rows is None:
        rows, threads = _pick_config(K, N)
    module = _jit_skinny_gemv_module(rows, threads)
    module.launch(x, w, out)
    return out
