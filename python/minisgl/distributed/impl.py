from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


@dataclass
class CustomARDistributedImpl(DistributedImpl):
    """One-shot NVLink allreduce (sgl_kernel custom_all_reduce, vLLM-style) for
    small tensors; anything it declines falls back to the previous plugin.
    At bs<=8 decode the per-call latency is ~3-5us vs ~20us for 8-rank NCCL."""

    ca: object
    fallback: DistributedImpl

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca.custom_all_reduce(x)
        return out if out is not None else self.fallback.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback.all_gather(x)


_custom_ar = None


def get_custom_ar():
    return _custom_ar


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))

    import os

    # opt-in: measured slower than NCCL NVLS on B300 (32us vs 20.4us per call),
    # and newer sglang trees pull heavyweight deps on this import path
    if os.environ.get("MINISGL_CUSTOM_AR", "0") != "1":
        return
    try:
        from sglang.srt.distributed.device_communicators import (
            custom_all_reduce as _car_mod,
        )

        # The stock NVLink probe walks sglang's own parallel_state (never
        # initialized under minisgl). We only enable this path on single-node
        # NVSwitch boxes, where the answer is a constant True.
        _orig_probe = _car_mod.can_use_custom_all_reduce_with_nvlink
        _car_mod.can_use_custom_all_reduce_with_nvlink = lambda **kw: True
        try:
            ca = _car_mod.CustomAllreduce(
                group=tp_cpu_group, device=torch.device(f"cuda:{tp_info.rank}")
            )
        finally:
            _car_mod.can_use_custom_all_reduce_with_nvlink = _orig_probe
        if getattr(ca, "disabled", True):
            return
        global _custom_ar
        _custom_ar = ca
        DistributedCommunicator.plugins.append(
            CustomARDistributedImpl(ca=ca, fallback=DistributedCommunicator.plugins[-1])
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"custom allreduce unavailable: {e}")


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
