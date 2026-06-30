from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """Latent KV cache for (absorbed) MLA.

    Stores the compressed latent per token per layer instead of per-head K/V:
      - ckv  : [num_layers, num_pages, page_size, kv_lora_rank]   (the "k" buffer)
      - k_pe : [num_layers, num_pages, page_size, qk_rope_head_dim] (the "v" buffer)
    The latent is shared across all heads (MQA-style), so it is NOT sharded over TP
    ranks -- each rank holds a full replica. This is the ~110x KV-cache reduction vs
    the naive per-head MHA pool.

    To fit the existing BaseKVCachePool interface, ckv plays the role of "k" and k_pe
    the role of "v"; store_kv(k=ckv, v=k_pe, ...) and k_cache/v_cache return them.
    """

    def __init__(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_layers: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._ckv_buffer = torch.empty(
            (num_layers, num_pages, page_size, kv_lora_rank), device=device, dtype=dtype
        )
        self._kpe_buffer = torch.empty(
            (num_layers, num_pages, page_size, qk_rope_head_dim), device=device, dtype=dtype
        )
        self._num_layers = num_layers
        self._device = device
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._ckv_flat = (num_pages * page_size, kv_lora_rank)
        self._kpe_flat = (num_pages * page_size, qk_rope_head_dim)

    # ckv == "k", k_pe == "v"
    def k_cache(self, index: int) -> torch.Tensor:
        return self._ckv_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._kpe_buffer[index]

    # explicit MLA-named aliases
    def ckv_cache(self, index: int) -> torch.Tensor:
        return self._ckv_buffer[index]

    def kpe_cache(self, index: int) -> torch.Tensor:
        return self._kpe_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # k == ckv [T, kv_lora_rank], v == k_pe [T, qk_rope_head_dim]; page_size==1 scatter
        self._ckv_buffer[layer_id].view(self._ckv_flat)[out_loc] = k
        self._kpe_buffer[layer_id].view(self._kpe_flat)[out_loc] = v

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._ckv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
