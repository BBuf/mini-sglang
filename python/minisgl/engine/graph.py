from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    import os

    env = os.environ.get("MINISGL_GRAPH_BS")
    if env:  # e.g. "1,2,4,5,8" — lets speculative k+1 hit an exact graph size
        return sorted(int(x) for x in env.split(","))
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        import contextlib

        from minisgl.distributed import get_custom_ar

        ca = get_custom_ar()
        # the custom one-shot allreduce records graph buffer addresses during
        # capture and registers them (IPC exchange) on context exit
        with ca.capture() if ca is not None else contextlib.nullcontext():
            self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # bs -> the model's pre-final-norm hidden buffer inside that graph (for MTP)
        self.hidden_map: Dict[int, torch.Tensor] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
            hidden = getattr(getattr(model, "model", None), "_last_hidden", None)
            if hidden is not None:
                self.hidden_map[bs] = hidden

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

        self._capture_mtp_graphs(model, pool)

    def _capture_mtp_graphs(self, model: BaseLLMModel, pool) -> None:
        # Speculative-decoding draft graphs: the MTP layer runs as tiny decode-shaped
        # batches (extend over accepted rows / chained q=1 steps). Capture them so
        # draft refill is not eager-launch bound.
        import os

        self.mtp_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.mtp_chain_graph = None
        spec_steps = int(os.environ.get("MINISGL_SPEC_STEPS", "0"))
        if spec_steps <= 0 or not hasattr(model, "forward_mtp"):
            return
        mtp = getattr(getattr(model, "model", None), "mtp", None)
        if mtp is None:
            return
        import os as _os
        _skip = int(_os.environ.get("MINISGL_MTP_SKIP_BS", "0"))
        bs_list = [
            bs for bs in self.graph_bs_list if bs <= spec_steps + 2 and bs != _skip
        ]
        if not bs_list:
            return
        max_bs = max(bs_list)
        dev = self.device
        # probe hidden size/dtype from the captured main hidden buffer
        any_hidden = next(iter(self.hidden_map.values()))
        H, hdtype = any_hidden.shape[-1], any_hidden.dtype
        vocab = self.buffer.logits.shape[1]
        self.mtp_buffer = GraphCaptureBuffer(
            input_ids=torch.zeros(max_bs, dtype=torch.int32, device=dev),
            out_loc=torch.zeros(max_bs, dtype=torch.int32, device=dev),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=dev),
            logits=torch.empty(max_bs, vocab, dtype=torch.float32, device=dev),
        )
        self.mtp_hidden_in = torch.zeros(max_bs, H, dtype=hdtype, device=dev)
        self.mtp_hidden_out = torch.empty(max_bs, H, dtype=hdtype, device=dev)
        logger.info_rank0(f"Capturing MTP draft graphs with sizes: {bs_list}")
        for bs in sorted(bs_list, reverse=True):
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_metadata(batch)
            self.attn_backend.prepare_for_replay(batch)
            self.mtp_buffer.set_batch(batch)
            batch.spec_prev_hidden = self.mtp_hidden_in[:bs]
            graph = torch.cuda.CUDAGraph()
            with get_global_ctx().forward_batch(batch):
                logits, hidden = model.forward_mtp()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    logits, hidden = model.forward_mtp()
                    self.mtp_buffer.logits[:bs] = logits
                    self.mtp_hidden_out[:bs] = hidden
            self.mtp_graph_map[bs] = graph

        self._chain_args = (model, pool, spec_steps)

    def capture_mtp_chain(self, token_pool: torch.Tensor) -> None:
        """Deferred (token_pool is created by the scheduler after engine init)."""
        if self.mtp_chain_graph is not None or not hasattr(self, "_chain_args"):
            return
        model, pool, k = self._chain_args
        self.chain_token_pool = token_pool
        try:
            self._capture_mtp_chain(model, pool, k)
        except Exception as e:
            logger.warning(f"MTP chain graph unavailable, per-step fallback: {e}")
            self.mtp_chain_graph = None

    def _capture_mtp_chain(self, model: BaseLLMModel, pool, k: int) -> None:
        """Speculative draft chain (steps 2..k) as ONE graph: each step feeds the
        previous step's in-graph argmax into the embedding, runs the MTP layer +
        shared head, and the final scatter writes all k drafts into token_pool.
        Removes the per-step python/replay orchestration (~0.13ms x k-1)."""
        self.mtp_chain_graph = None
        C = k - 1
        if C < 1:
            return
        dev = self.device
        H = self.mtp_hidden_in.shape[1]
        # round inputs, filled by the scheduler before replay
        self.chain_pos = torch.zeros(C, dtype=torch.int32, device=dev)      # attn positions
        self.chain_seq = torch.zeros(C, dtype=torch.int32, device=dev)      # kv lens
        self.chain_out_loc = torch.zeros(C, dtype=torch.int32, device=dev)  # MTP-KV slots
        self.chain_row = torch.zeros(1, dtype=torch.int64, device=dev)      # extend's last real row
        self.chain_tok_idx = torch.zeros(k, dtype=torch.int64, device=dev)  # token_pool flat slots
        self.chain_drafts = torch.zeros(k, dtype=torch.int32, device=dev)   # d_1..d_k

        from minisgl.attention.mla import MLAMetadata

        cap = self.attn_backend.capture
        buf_logits, buf_hidden = self.mtp_buffer.logits, self.mtp_hidden_out
        batches = []
        for i in range(C):
            b = Batch(reqs=[self.dummy_req], phase="decode")
            b.padded_reqs = b.reqs
            b.positions = self.chain_pos[i : i + 1]
            b.out_loc = self.chain_out_loc[i : i + 1]
            b.attn_metadata = MLAMetadata(
                qo_indptr=self.chain_seq[i : i + 1],
                kv_indptr=self.chain_seq[i : i + 1],
                kv_indices=self.chain_seq[i : i + 1],
                kv_len_arr=self.chain_seq[i : i + 1],
                num_heads=self.attn_backend.num_heads_local,
                causal=True,
                wrapper=self.attn_backend.wrapper,
                use_trtllm=True,
                block_tables=cap.block_tables[:1],
                initialized=True,
            )
            batches.append(b)

        def run_chain():
            d = torch.argmax(
                buf_logits.index_select(0, self.chain_row), dim=-1
            ).to(torch.int32)
            h = buf_hidden.index_select(0, self.chain_row)
            self.chain_drafts[0:1] = d
            for i in range(C):
                b = batches[i]
                b.spec_prev_hidden = h
                b.input_ids = d
                with get_global_ctx().forward_batch(b):
                    logits, h = model.forward_mtp()
                d = torch.argmax(logits[:1], dim=-1).to(torch.int32)
                self.chain_drafts[i + 1 : i + 2] = d
            self.chain_token_pool.view(-1)[self.chain_tok_idx] = self.chain_drafts

        graph = torch.cuda.CUDAGraph()
        run_chain()  # warmup (also lockstep across TP ranks)
        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
            run_chain()
        self.mtp_chain_graph = graph
        logger.info_rank0(f"Captured fused MTP chain graph ({C} steps)")

    def replay_mtp(self, batch: Batch, input_ids: torch.Tensor):
        bs = batch.padded_size
        buf = self.mtp_buffer
        buf.input_ids[:bs] = input_ids
        buf.positions[:bs] = batch.positions
        buf.out_loc[:bs] = batch.out_loc
        self.mtp_hidden_in[:bs] = batch.spec_prev_hidden[:bs]
        self.attn_backend.prepare_for_replay(batch)
        self.mtp_graph_map[bs].replay()
        return buf.logits[: batch.size], self.mtp_hidden_out[: batch.size]

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
