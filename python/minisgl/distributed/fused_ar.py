from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo

# FlashInfer trtllm allreduce fusion: one kernel does AR + residual-add +
# RMSNorm (the pattern sglang auto-enables on SM10x). At decode shapes this
# replaces [NCCL NVLS AR (~12us) + fused_add_rmsnorm] twice per layer.
_FUSED: Optional["FusedARNorm"] = None
_PASS_ACTIVE: bool = False


class FusedARNorm:
    def __init__(self, tp_rank: int, tp_size: int, group, hidden: int, max_tokens: int):
        from flashinfer.comm import (
            AllReduceFusionPattern,
            trtllm_allreduce_fusion,
            trtllm_create_ipc_workspace_for_all_reduce_fusion,
        )

        self.rank = tp_rank
        self.tp = tp_size
        self.max_tokens = max_tokens
        self.hidden = hidden
        self._fn = trtllm_allreduce_fusion
        self._pattern = AllReduceFusionPattern.kARResidualRMSNorm
        self._pattern_ar = AllReduceFusionPattern.kAllReduce
        self.ipc_handles, self.workspace = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank, tp_size, max_tokens, hidden, use_fp32_lamport=False, group=group
        )

    def ar_add_rmsnorm(
        self, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        self._fn(
            allreduce_in=x,
            world_size=self.tp,
            world_rank=self.rank,
            token_num=x.shape[0],
            hidden_dim=x.shape[1],
            workspace_ptrs=self.workspace,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=os.environ.get("MINISGL_FUSED_AR_FP32", "1") == "1",
            pattern_code=self._pattern,
            use_oneshot=True,
            allreduce_out=None,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=gamma,
            rms_eps=eps,
            scale_factor=None,
            layout_code=None,
        )
        return norm_out, residual_out


    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Plain one-shot AR (no fused norm): isolates the reduction-algorithm
        swap from the norm-kernel swap; NCCL on this fleet is ~58us/call
        in-graph while the one-shot kernel is ~15us at decode sizes."""
        out = torch.empty_like(x)
        self._fn(
            allreduce_in=x,
            world_size=self.tp,
            world_rank=self.rank,
            token_num=x.shape[0],
            hidden_dim=x.shape[1],
            workspace_ptrs=self.workspace,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=os.environ.get("MINISGL_FUSED_AR_FP32", "1") == "1",
            pattern_code=self._pattern_ar,
            use_oneshot=True,
            allreduce_out=out,
            residual_in=None,
            residual_out=None,
            norm_out=None,
            quant_out=None,
            scale_out=None,
            rms_gamma=None,
            rms_eps=None,
            scale_factor=None,
            layout_code=None,
        )
        return out


def init_fused_ar(tp_info: "DistributedInfo", group, hidden: int) -> None:
    global _FUSED
    # default OFF: measured on B200/fp4/fi-MLA it cuts the round 17.2 -> 15.5ms
    # but the AR numerics-style change collapses deep-chain accept 4.37 -> 3.40
    # (fp32_acc; bf16 acc 3.10) - net negative. Retest on B300 worlds.
    # default ON in full-fused mode: large-sample GSM wall-clock (60 ex,
    # ~60K tok/side) is ~6% faster (216-220s vs 233s NCCL) at neutral accept
    # (4.51 vs 4.55 over 13k+ rounds). The morning's "AR collapses accept"
    # was a 200-round small-sample artifact. Note the win is modest because
    # NCCL AR mostly overlaps compute (41% of summed GPU time but not the
    # serial critical path). Off with MINISGL_FUSED_AR=0.
    mode = os.environ.get("MINISGL_FUSED_AR", "1")
    if tp_info.size <= 1 or mode not in ("1", "ar"):
        return
    try:
        max_tokens = int(os.environ.get("MINISGL_FUSED_AR_MAX_TOKENS", "64"))
        _FUSED = FusedARNorm(tp_info.rank, tp_info.size, group, hidden, max_tokens)
    except Exception as e:  # missing flashinfer API / IPC failure: silently off
        from minisgl.utils import init_logger

        init_logger(__name__).warning(f"fused AR unavailable, NCCL path kept: {e}")
        _FUSED = None


def get_fused_ar() -> Optional[FusedARNorm]:
    return _FUSED


def set_fused_pass(num_tokens: int) -> bool:
    """Called at the top of a model forward; returns whether this pass runs
    with row-parallel ARs deferred into the fused AR+norm calls. In plain-AR
    mode (MINISGL_FUSED_AR=ar) the pass flag stays off: the norm sites keep
    their kernels and DistributedCommunicator swaps just the reduction."""
    global _PASS_ACTIVE
    _PASS_ACTIVE = (
        _FUSED is not None
        and os.environ.get("MINISGL_FUSED_AR", "1") == "1"
        and num_tokens <= _FUSED.max_tokens
    )
    return _PASS_ACTIVE


def plain_ar_or_none(x: torch.Tensor) -> Optional[torch.Tensor]:
    if (
        _FUSED is not None
        and os.environ.get("MINISGL_FUSED_AR", "0") == "ar"
        and x.dim() == 2
        and x.shape[0] <= _FUSED.max_tokens
        and x.shape[1] == _FUSED.hidden
    ):
        return _FUSED.all_reduce(x)
    return None


def fused_pass_active() -> bool:
    return _PASS_ACTIVE
