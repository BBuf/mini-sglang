from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)


@functools.cache
def _jit_store_mla_module(
    ckv_size: int,
    kpe_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(ckv_size, kpe_size, *config)
    return load_jit(
        "store_mla",
        *args,
        cuda_files=["store_mla.cu"],
        cuda_wrappers=[("launch", f"StoreMLAKernel<{args}>::run")],
    )


def store_mla_cache(
    ckv_cache: torch.Tensor,  # [num_tokens, kv_lora_rank]
    kpe_cache: torch.Tensor,  # [num_tokens, qk_rope_head_dim]
    indices: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
) -> None:
    ckv_size = ckv_cache.shape[1] * ckv_cache.element_size()
    kpe_size = kpe_cache.shape[1] * kpe_cache.element_size()
    module = _jit_store_mla_module(ckv_size, kpe_size)
    module.launch(ckv_cache, kpe_cache, indices, ckv, kpe)
