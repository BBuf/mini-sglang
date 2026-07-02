from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import BatchMLAPagedAttentionWrapper
    from minisgl.models import ModelConfig

logger = init_logger(__name__)

# flashinfer's "auto" resolves to fa2 on sm_103 (B300) because the fa3 gate is
# sm90a-only; the cutlass backend is the Blackwell-native MLA kernel.
import os

_MLA_BACKEND = os.environ.get("MINISGL_MLA_BACKEND", "auto")


@dataclass
class MLACaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens


@dataclass
class MLAMetadata(BaseAttnMetadata):
    # fmt: off
    qo_indptr:   torch.Tensor   # gpu int32 [bs+1]
    kv_indptr:   torch.Tensor   # gpu int32 [bs+1]
    kv_indices:  torch.Tensor   # gpu int32 [sum_kv]
    kv_len_arr:  torch.Tensor   # gpu int32 [bs]
    num_heads:   int
    causal:      bool
    wrapper:     "BatchMLAPagedAttentionWrapper"
    initialized: bool = False
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_indptr[1 : 1 + bs] - 1


class MLABackend(BaseAttnBackend):
    """Absorbed-MLA attention backend using FlashInfer BatchMLAPagedAttentionWrapper.

    The model passes the latent query (q_nope already absorbed via W_UK -> [T,H,512])
    and q_pe [T,H,64]; we store the compressed (ckv, k_pe) into the MLA pool and run
    paged latent attention. Returns o_latent [T,H,512]; the model then absorbs W_UV.
    """

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import BatchMLAPagedAttentionWrapper

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.ckv_dim = config.kv_lora_rank
        self.kpe_dim = config.qk_rope_head_dim
        self.sm_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5

        tp_size = get_tp_info().size
        self.num_heads_local = div_even(config.num_qo_heads, tp_size)

        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(self.float_workspace_buffer, backend=_MLA_BACKEND)

        # cuda graph state
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, "BatchMLAPagedAttentionWrapper"] = {}
        self.capture: MLACaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    # The generic (q,k,v) path is unused for MLA; the model calls forward_mla.
    def forward(self, q, k, v, layer_id, batch):  # type: ignore[override]
        raise NotImplementedError("MLABackend uses forward_mla()")

    def _plan_once(self, metadata: MLAMetadata) -> None:
        if metadata.initialized:
            return
        metadata.initialized = True
        self.last_event.synchronize()
        metadata.wrapper.plan(
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
            metadata.num_heads,
            self.ckv_dim,
            self.kpe_dim,
            1,  # page_size
            metadata.causal,
            self.sm_scale,
            self.kvcache.dtype,
            self.kvcache.dtype,
        )
        self.last_event.record()

    def forward_mla(
        self,
        q_nope: torch.Tensor,   # [T, H_local, ckv_dim] (absorbed latent query)
        q_pe: torch.Tensor,     # [T, H_local, kpe_dim]
        ckv: torch.Tensor,      # [T, ckv_dim]
        k_pe: torch.Tensor,     # [T, kpe_dim]
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        self._plan_once(metadata)
        self.kvcache.store_kv(ckv, k_pe, batch.out_loc, layer_id)
        ckv_cache = self.kvcache.ckv_cache(layer_id)  # [num_pages, page_size=1, ckv_dim]
        kpe_cache = self.kvcache.kpe_cache(layer_id)  # [num_pages, page_size=1, kpe_dim]
        return metadata.wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        qo_indptr = torch.tensor([0] + seqlens_q, **CPU).cumsum_(0).to(torch.int32)
        kv_indptr = torch.tensor([0] + seqlens_k, **CPU).cumsum_(0).to(torch.int32)
        kv_len_arr = torch.tensor(seqlens_k, **CPU)
        page_table = get_global_ctx().page_table
        kv_indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        dev = self.device
        batch.attn_metadata = MLAMetadata(
            qo_indptr=qo_indptr.to(dev, non_blocking=True),
            kv_indptr=kv_indptr.to(dev, non_blocking=True),
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr.to(dev, non_blocking=True),
            num_heads=self.num_heads_local,
            causal=True,
            wrapper=self.wrapper,
        )

    # ----- cuda graph -----
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None
        max_bs = max(bs_list)
        capture = MLACaptureData.create(max_bs, max_seq_len, self.device)
        capture.page_table = capture.page_table.view(-1)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def _make_graph_wrapper(self, bs: int) -> "BatchMLAPagedAttentionWrapper":
        from flashinfer import BatchMLAPagedAttentionWrapper

        cap = self.capture
        assert cap is not None
        return BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer,
            use_cuda_graph=True,
            qo_indptr=cap.cu_seqlens_q[: bs + 1],
            kv_indptr=cap.cu_seqlens_k[: bs + 1],
            kv_indices=cap.page_table,
            kv_len_arr=cap.seq_lens[:bs],
            backend=_MLA_BACKEND,
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers
        self.graph_wrappers[bs] = self._make_graph_wrapper(bs)
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._plan_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, MLAMetadata) and not metadata.initialized
        assert bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._plan_once(metadata)
