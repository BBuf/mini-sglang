from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import os

import torch
from minisgl.core import Batch, Req
from minisgl.message import DetokenizeMsg
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .scheduler import Scheduler

logger = init_logger(__name__)


class _VirtualReq(Req):
    """A view of a real request at a shifted (cached_len, device_len), sharing its
    page-table row. Lets speculative verify / draft steps masquerade as ordinary
    decode batches, so they reuse the captured decode cuda graphs and the normal
    attention-metadata path (each row attends causally to slots [0, device_len))."""

    def __init__(self, base: Req, cached_len: int, device_len: int):
        self.input_ids = base.input_ids  # unused by the forward path
        self.table_idx = base.table_idx
        self.cached_len = cached_len
        self.output_len = 1
        self.uid = base.uid
        self.sampling_params = base.sampling_params
        self.cache_handle = base.cache_handle
        self.device_len = device_len
        self.max_device_len = device_len + 1


@dataclass
class _SpecState:
    alloc_len: int  # page-table high-water mark: positions < alloc_len have pages
    drafts_ready: bool = False


class SpecManager:
    """MTP / NextN speculative decoding (bs=1, greedy).

    Per round: verify the k pending drafts with one graphed decode batch of k+1
    staggered virtual requests, accept the matching prefix (+1 bonus token), then
    run the 1-layer MTP draft model to refill k drafts. Draft token values only
    ever live in token_pool on GPU; the CPU sees just the accepted tokens.
    """

    def __init__(self, sched: Scheduler, steps: int):
        import os

        self.s = sched
        self.k = steps
        self.engine = sched.engine
        self.device = sched.device
        self.states: Dict[int, _SpecState] = {}
        self._log_stats = os.environ.get("MINISGL_SPEC_LOG", "0") == "1"
        self._rounds = 0
        self._accepted = 0
        self._t_verify = 0.0
        self._t_draft = 0.0
        # drafts are copied to pinned host memory as they are produced, so the
        # verify round reads them without a blocking .cpu()
        self._drafts_gpu = torch.empty(steps, dtype=torch.int32, device=self.device)
        self._chain_pos_buf = torch.empty(1, dtype=torch.int32, device=self.device)
        CPU = {"dtype": torch.int32, "pin_memory": True}
        self._pin_pos = torch.empty(max(steps - 1, 1), **CPU)
        self._pin_seq = torch.empty(max(steps - 1, 1), **CPU)
        self._pin_tok = torch.empty(steps, dtype=torch.int64, pin_memory=True)
        self._drafts_pin = torch.empty(steps, dtype=torch.int32, pin_memory=True)
        self._drafts_event = torch.cuda.Event()

    # ------------------------------------------------------------- helpers ----
    @property
    def _token_pool(self) -> torch.Tensor:
        return self.s.token_pool

    @property
    def _page_table(self) -> torch.Tensor:
        return self.engine.page_table

    def _ensure_pages(self, req: Req, need_len: int) -> None:
        st = self.states[req.uid]
        if need_len > st.alloc_len:
            v = _VirtualReq(req, st.alloc_len, need_len)
            self.s.cache_manager.allocate_paged([v])
            # allocate_paged rounds up to whole pages; track the aligned mark so
            # the tail frees below release exactly what was allocated
            ps = self.s.cache_manager.page_size
            st.alloc_len = -(-need_len // ps) * ps

    def _make_batch(
        self,
        req: Req,
        rows: List[Tuple[int, int]],  # (cached_len, device_len) per row
        positions: torch.Tensor,  # int32 gpu, attention position per row
        phase: str,
        use_graph: bool,
        reuse_bt: bool = False,
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
        batch = Batch(reqs=[_VirtualReq(req, c, d) for c, d in rows], phase=phase)
        if reuse_bt:
            # chained MTP drafts hit the same page-table row the extend pass just
            # bound; skip recomputing/copying the block table
            batch.spec_reuse_bt = True
        if use_graph:
            self.engine.graph_runner.pad_batch(batch)
        else:
            batch.padded_reqs = batch.reqs
        pad = batch.padded_size - batch.size
        if pad > 0:
            dummy_idx = self.engine.dummy_req.table_idx
            positions = torch.cat(
                [positions, torch.zeros(pad, dtype=torch.int32, device=self.device)]
            )
            table = torch.tensor(
                [r.table_idx for r in batch.reqs] + [dummy_idx] * pad,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            table = torch.full(
                (len(positions),), req.table_idx, dtype=torch.int64, device=self.device
            )
        out_pos = positions.to(torch.int64)
        batch.positions = positions
        batch.out_loc = self._page_table[(table, out_pos)]
        self.engine.attn_backend.prepare_metadata(batch)
        return batch, table, out_pos

    # ------------------------------------------------------------ bootstrap ----
    def try_bootstrap(self, batch: Batch) -> None:
        """After an unchunked, prefix-cache-free, single-request prefill: run the
        MTP layer over the whole prompt to build its KV and produce k drafts."""
        if len(batch.reqs) != 1:
            return
        req = batch.reqs[0]
        if not req.sampling_params.is_greedy or not req.can_decode:
            return
        L = req.device_len - 1  # prompt length (device_len advanced by complete_one)
        if len(batch.positions) != L:
            return  # chunked or prefix-cached prefill: no hidden for early rows
        if req.remain_len <= self.k + 1:
            return
        hidden = self.engine.get_last_hidden(batch)[:L]
        self.states[req.uid] = _SpecState(alloc_len=L)
        table = req.table_idx
        self._ensure_pages(req, L + self.k + 1)
        # rows 0..L-1: input tokens are positions 1..L (prompt shifted + sampled t_L)
        positions = torch.arange(0, L, dtype=torch.int32, device=self.device)
        mtp_batch, tbl, out_pos = self._make_batch(req, [(0, L)], positions, "prefill", False)
        mtp_batch.spec_prev_hidden = hidden
        input_ids = self._token_pool[(tbl, out_pos + 1)]
        logits, mtp_hidden = self.engine.forward_mtp_batch(mtp_batch, input_ids)
        d = torch.argmax(logits[-1:], dim=-1).to(torch.int32)
        self._token_pool[table, L + 1] = d[0]
        self._chain_drafts(req, next_pos=L, prev_draft=d, prev_hidden=mtp_hidden[-1:])
        self.states[req.uid].drafts_ready = True

    def _chain_drafts(
        self, req: Req, next_pos: int, prev_draft: torch.Tensor, prev_hidden: torch.Tensor
    ) -> None:
        """Autoregressive draft steps 2..k through the MTP layer (q=1 each).
        `next_pos` is the first attention position after the rows already run
        through the MTP layer. Chained row i (i=1..k-1) sits at position
        next_pos+i-1, consumes emb(d_i) (== the token at its position+1) plus the
        previous MTP hidden, and its argmax d_{i+1} lands at token position+2."""
        table = req.table_idx
        self._drafts_gpu[0] = prev_draft[0]
        for i in range(1, self.k):
            pos = next_pos + i - 1
            positions = self._chain_pos_buf
            positions.fill_(pos)
            b, _, _ = self._make_batch(
                req, [(pos, pos + 1)], positions, "decode", True, reuse_bt=True
            )
            b.spec_prev_hidden = prev_hidden
            logits, mtp_hidden = self.engine.forward_mtp_batch(b, prev_draft)
            prev_draft = torch.argmax(logits[:1], dim=-1).to(torch.int32)
            prev_hidden = mtp_hidden[:1]
            self._token_pool[table, pos + 2] = prev_draft[0]
            self._drafts_gpu[i] = prev_draft[0]
        self._drafts_pin.copy_(self._drafts_gpu, non_blocking=True)
        self._drafts_event.record()

    # ---------------------------------------------------------------- round ----
    def pick_req(self) -> Optional[Req]:
        running = self.s.decode_manager.running_reqs
        if len(running) != 1:
            return None
        req = next(iter(running))
        st = self.states.get(req.uid)
        if st is None or not st.drafts_ready:
            return None
        if not req.sampling_params.is_greedy or req.remain_len <= self.k + 1:
            return None
        return req

    def run_round(self, req: Req) -> None:
        k = self.k
        D = req.device_len
        table = req.table_idx
        self._ensure_pages(req, D + 2 * k + 1)

        # ---- verify: k+1 staggered rows through the normal (graphed) decode path
        drafts_cpu = self._drafts_pin
        rows = [(D - 1 + i, D + i) for i in range(k + 1)]
        positions = torch.arange(D - 1, D + k, dtype=torch.int32, device=self.device)
        batch, tbl, out_pos = self._make_batch(req, rows, positions, "decode", True)
        batch.input_ids = self._token_pool[(tbl, out_pos)]
        sample_args = self.engine.sampler.prepare(batch)
        out = self.engine.forward_batch(batch, sample_args)
        hidden = self.engine.get_last_hidden(batch)

        # ---- accept the longest matching prefix (+ the bonus prediction)
        out.copy_done_event.synchronize()
        self._drafts_event.synchronize()  # ordered earlier on the stream: ~free
        preds_cpu = out.next_tokens_cpu[: k + 1]
        if self._log_stats and self._rounds < 30:
            logger.info_rank0(
                f"[specdbg] r={self._rounds} D={D} drafts={drafts_cpu[:k].tolist()} "
                f"preds={preds_cpu.tolist()}"
            )
        n = 0
        while n < k and int(preds_cpu[n]) == int(drafts_cpu[n]):
            n += 1
        n_new = n + 1  # accepted drafts + bonus

        finished = False
        accepted = preds_cpu[:n_new]
        if not req.sampling_params.ignore_eos:
            eos = (accepted == self.s.eos_token_id).nonzero()
            if eos.numel() > 0:
                n_new = int(eos[0, 0]) + 1
                accepted = accepted[:n_new]
                finished = True
        remain = req.max_device_len - D
        if n_new >= remain:
            n_new = remain
            accepted = accepted[:n_new]
            finished = True

        # publish accepted tokens (idempotent for the matched prefix); the host
        # bookkeeping (append/detok/send) happens AFTER the draft phase is
        # launched so it overlaps the draft GPU work
        self._token_pool[table, D : D + n_new] = out.next_tokens_gpu[:n_new]
        req.cached_len = D + n_new - 1
        req.device_len = D + n_new

        if not finished:
            self._refill_drafts(req, hidden, D, n_new, table)

        req.append_host(accepted.to(torch.int32))
        self.s.send_result(
            [
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=int(t),
                    finished=finished and (i == n_new - 1),
                )
                for i, t in enumerate(accepted)
            ]
        )
        if finished:
            self.finish_req(req)
        return

    def _refill_drafts(self, req: Req, hidden, D: int, n_new: int, table: int) -> None:
        # ---- refill drafts: MTP extend over the accepted rows, then chain
        k = self.k
        m = n_new  # rows D-1 .. D-1+m-1 consume tokens D..D+m-1 and hidden rows 0..m-1
        positions = torch.arange(D - 1, D - 1 + m, dtype=torch.int32, device=self.device)
        rows = [(D - 1 + i, D + i) for i in range(m)]
        mtp_batch, tbl, out_pos = self._make_batch(req, rows, positions, "decode", True)
        mtp_batch.spec_prev_hidden = hidden[: mtp_batch.padded_size]
        input_ids = self._token_pool[(tbl, out_pos + 1)]
        J = D + n_new - 2  # last accepted main position
        gr = self.engine.graph_runner
        if (
            getattr(gr, "mtp_chain_graph", None) is not None
            and mtp_batch.padded_size in gr.mtp_graph_map
            and os.environ.get("MINISGL_CHAIN_GRAPH", "1") == "1"
        ):
            # fused chain: the extend replay leaves logits/hidden in the MTP
            # buffers; one more replay runs argmax->emb->layer x (k-1) plus the
            # token_pool scatter of all k drafts entirely in-graph.
            self.engine.forward_mtp_batch(mtp_batch, input_ids)
            k1 = self.k - 1
            base = J + 1
            gr.chain_row.fill_(m - 1)
            torch.arange(base, base + k1, dtype=torch.int32, out=self._pin_pos[:k1])
            torch.arange(base + 1, base + k1 + 1, dtype=torch.int32, out=self._pin_seq[:k1])
            W = self._token_pool.shape[1]
            torch.arange(
                table * W + J + 2, table * W + J + 2 + self.k, dtype=torch.int64,
                out=self._pin_tok[: self.k],
            )
            gr.chain_pos.copy_(self._pin_pos[:k1], non_blocking=True)
            gr.chain_seq.copy_(self._pin_seq[:k1], non_blocking=True)
            gr.chain_tok_idx.copy_(self._pin_tok[: self.k], non_blocking=True)
            gr.chain_out_loc.copy_(self._page_table[table, base : base + k1])
            gr.mtp_chain_graph.replay()
            self._drafts_pin.copy_(gr.chain_drafts[: self.k], non_blocking=True)
            self._drafts_event.record()
        else:
            logits, mtp_hidden = self.engine.forward_mtp_batch(mtp_batch, input_ids)
            d = torch.argmax(logits[m - 1 : m], dim=-1).to(torch.int32)
            self._token_pool[table, J + 2] = d[0]
            self._chain_drafts(
                req, next_pos=J + 1, prev_draft=d, prev_hidden=mtp_hidden[m - 1 : m]
            )

        self._rounds += 1
        self._accepted += n_new
        if self._log_stats:
            if not hasattr(self, "_n_hist"):
                self._n_hist = [0] * (self.k + 2)
            self._n_hist[n_new] += 1
            if self._rounds % 200 == 0:
                logger.info_rank0(
                    f"[spec] rounds={self._rounds} "
                    f"mean_accept={self._accepted / self._rounds:.2f} "
                    f"hist={self._n_hist}"
                )

    # ------------------------------------------------------------- cleanup ----
    def _free_tail(self, req: Req, alloc_len: int) -> None:
        # free the whole pages past the request's live region; the partial page
        # containing cached_len (if any) is owned/freed by cache_req, and _free
        # picks page starts by striding, so the slice must begin page-aligned
        ps = self.s.cache_manager.page_size
        start = -(-req.cached_len // ps) * ps
        if alloc_len > start:
            self.s.cache_manager._free(self._page_table[req.table_idx, start:alloc_len])

    def finish_req(self, req: Req) -> None:
        st = self.states.pop(req.uid, None)
        self.s.decode_manager.remove_req(req)
        self.s._free_req_resources(req)
        if st is not None:
            self._free_tail(req, st.alloc_len)

    def drop_req(self, req: Req) -> None:
        """Called when a spec'd request degrades to the normal decode path (its
        drafts / hidden chain are gone) or gets aborted. The normal path's
        allocate_paged re-allocates from cached_len itself, so everything from
        the next page boundary up is released here."""
        st = self.states.pop(req.uid, None)
        if st is not None:
            self._free_tail(req, st.alloc_len)
