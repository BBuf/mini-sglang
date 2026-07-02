from __future__ import annotations

import os
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
_MLA_BACKEND = os.environ.get("MINISGL_MLA_BACKEND", "auto")


@dataclass
class MLACaptureData(BaseCaptureData):
    block_tables: torch.Tensor | None = None

    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens


@dataclass
class MLAMetadata(BaseAttnMetadata):
    # fmt: off
    qo_indptr:    torch.Tensor   # gpu int32 [bs+1]
    kv_indptr:    torch.Tensor   # gpu int32 [bs+1] (block counts)
    kv_indices:   torch.Tensor   # gpu int32 [sum_blocks]
    kv_len_arr:   torch.Tensor   # gpu int32 [bs] (token lengths)
    num_heads:    int
    causal:       bool
    wrapper:      "BatchMLAPagedAttentionWrapper"
    use_trtllm:   bool = False
    block_tables: torch.Tensor | None = None  # gpu int32 [bs, W] (trtllm decode)
    initialized:  bool = False
    # fmt: on

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_indptr[1 : 1 + bs] - 1


class MLABackend(BaseAttnBackend):
    """Absorbed-MLA attention backend.

    Decode batches (extend_len == 1 per row) run the stateless trtllm-gen MLA
    decode kernel: its sm_100f cubins run on B300 where flashinfer's "auto"
    falls back to fa2 (~2x slower), and replays need no plan call — just two
    buffer copies. Prefill/extend keeps the FlashInfer wrapper. Both read the
    SAME combined (ckv | k_pe) paged pool via views; trtllm needs page_size
    32/64 (launch with --page-size 64).
    """

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import BatchMLAPagedAttentionWrapper

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.page_size = get_global_ctx().page_size
        self.device = self.kvcache.device
        self.ckv_dim = config.kv_lora_rank
        self.kpe_dim = config.qk_rope_head_dim
        self.nope_dim = config.qk_nope_head_dim
        self.sm_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5

        tp_size = get_tp_info().size
        self.num_heads_local = div_even(config.num_qo_heads, tp_size)

        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer, backend=_MLA_BACKEND
        )

        self.use_trtllm_decode = (
            os.environ.get("MINISGL_TRTLLM_MLA", "1") == "1"
            and self.page_size in (32, 64)
            and hasattr(self.kvcache, "combined_cache")
        )
        if self.use_trtllm_decode:
            try:
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

                self._trtllm_decode = trtllm_batch_decode_with_kv_cache_mla
                # trtllm-gen sizes its split-k partials by bs x max_seq_len; the
                # shared 128MB flashinfer workspace is too small at engine limits
                self.trtllm_workspace = torch.empty(
                    512 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
            except ImportError:
                self.use_trtllm_decode = False
        logger.info_rank0(
            f"MLA decode path: {'trtllm-gen' if self.use_trtllm_decode else 'flashinfer'}"
        )

        # cuda graph state
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, "BatchMLAPagedAttentionWrapper"] = {}
        self.capture: MLACaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()
        self._max_seq_len = 0

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
            self.page_size,
            metadata.causal,
            self.sm_scale,
            self.kvcache.dtype,
            self.kvcache.dtype,
        )
        self.last_event.record()

    def _max_seq_len_hint(self) -> int:
        if self._max_seq_len == 0:
            self._max_seq_len = int(get_global_ctx().page_table.shape[1])
        return self._max_seq_len

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
        self.kvcache.store_kv(ckv, k_pe, batch.out_loc, layer_id)
        if metadata.use_trtllm:
            try:
                from sgl_kernel import concat_mla_absorb_q

                q = concat_mla_absorb_q(q_nope, q_pe)
            except ImportError:
                q = torch.cat([q_nope, q_pe], dim=-1)
            out = self._trtllm_decode(
                query=q.unsqueeze(1),
                kv_cache=self.kvcache.combined_cache(layer_id).unsqueeze(1),
                workspace_buffer=self.trtllm_workspace,
                qk_nope_head_dim=self.nope_dim,
                kv_lora_rank=self.ckv_dim,
                qk_rope_head_dim=self.kpe_dim,
                block_tables=metadata.block_tables,
                seq_lens=metadata.kv_len_arr,
                max_seq_len=self._max_seq_len_hint(),
                bmm1_scale=self.sm_scale,
            )
            return out.squeeze(1)
        self._plan_once(metadata)
        ckv_cache = self.kvcache.ckv_cache(layer_id)  # [num_pages, page_size, ckv_dim]
        kpe_cache = self.kvcache.kpe_cache(layer_id)  # [num_pages, page_size, kpe_dim]
        return metadata.wrapper.run(
            q_nope.contiguous(), q_pe.contiguous(), ckv_cache, kpe_cache
        )

    def _block_tables_for(self, batch: Batch) -> torch.Tensor:
        # page-aligned allocation guarantees page_table[t, k*ps] is a page start
        reqs = batch.padded_reqs
        ps = self.page_size
        max_blocks = max(-(-req.device_len // ps) for req in reqs)
        tables = torch.tensor(
            [r.table_idx for r in reqs], dtype=torch.int64, device=self.device
        )
        pt = get_global_ctx().page_table[tables, : max_blocks * ps : ps]
        return torch.div(pt, ps, rounding_mode="floor").to(torch.int32)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        ps = self.page_size
        seqlens_k = [req.device_len for req in reqs]
        CPU = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        kv_len_arr = torch.tensor(seqlens_k, **CPU)
        dev = self.device
        use_trtllm = self.use_trtllm_decode and batch.is_decode
        page_table = get_global_ctx().page_table
        if use_trtllm:
            # the trtllm decode path reads only kv_len_arr + block_tables; skip
            # the indptr construction (2 pinned allocs + copies per batch).
            kv_len_gpu = kv_len_arr.to(dev, non_blocking=True)
            batch.attn_metadata = MLAMetadata(
                qo_indptr=kv_len_gpu,  # placeholder, unused on this path
                kv_indptr=kv_len_gpu,
                kv_indices=kv_len_gpu,
                kv_len_arr=kv_len_gpu,
                num_heads=self.num_heads_local,
                causal=True,
                wrapper=self.wrapper,
                use_trtllm=True,
                block_tables=(
                    None
                    if getattr(batch, "spec_reuse_bt", False)
                    else self._block_tables_for(batch)
                ),
            )
            return
        seqlens_q = [req.extend_len for req in reqs]
        num_blocks = [-(-l // ps) for l in seqlens_k]
        qo_indptr = torch.tensor([0] + seqlens_q, **CPU).cumsum_(0).to(torch.int32)
        kv_indptr = torch.tensor([0] + num_blocks, **CPU).cumsum_(0).to(torch.int32)
        kv_indices = torch.cat(
            [
                torch.div(
                    page_table[req.table_idx, : req.device_len : ps],
                    ps,
                    rounding_mode="floor",
                )
                for req in reqs
            ]
        )
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
        self._max_seq_len = max_seq_len
        max_bs = max(bs_list)
        capture = MLACaptureData.create(max_bs, max_seq_len, self.device)
        capture.page_table = capture.page_table.view(-1)
        capture.block_tables = torch.zeros(
            (max_bs, -(-max_seq_len // self.page_size)),
            dtype=torch.int32,
            device=self.device,
        )
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

    def _bind_capture_buffers(self, metadata: MLAMetadata, bs: int) -> None:
        """Copy the freshly computed values into the static capture buffers and
        point the metadata at them, so graph replays see the updates. When
        block_tables is None (speculative chained drafts on the same request),
        the rows already sitting in the capture buffer are reused as-is."""
        cap = self.capture
        assert cap is not None
        cap.seq_lens[:bs].copy_(metadata.kv_len_arr[:bs])
        if metadata.block_tables is not None:
            w = metadata.block_tables.shape[1]
            cap.block_tables[:bs, :w].copy_(metadata.block_tables)
        metadata.kv_len_arr = cap.seq_lens[:bs]
        metadata.block_tables = cap.block_tables[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, MLAMetadata)
        if metadata.use_trtllm:
            self._bind_capture_buffers(metadata, bs)
            return
        assert bs not in self.graph_wrappers
        self.graph_wrappers[bs] = self._make_graph_wrapper(bs)
        metadata.wrapper = self.graph_wrappers[bs]
        self._plan_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, MLAMetadata) and not metadata.initialized
        assert bs in self.capture_bs
        if metadata.use_trtllm:
            self._bind_capture_buffers(metadata, bs)
            metadata.initialized = True
            return
        metadata.wrapper = self.graph_wrappers[bs]
        self._plan_once(metadata)
