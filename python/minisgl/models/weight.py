from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj", ".q_b_proj", ".kv_b_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_LAYER_IDX_PATTERN = re.compile(r"model\.layers\.(\d+)\.")
_SCALE_SUFFIX = ".weight_scale_inv"


def _remap_mtp_key(name: str, num_layers: int) -> str:
    """The checkpoint stores the MTP / NextN draft layer as model.layers.<num_layers>;
    remap the pieces mini models onto model.mtp.*. Whatever stays under
    layers.<num_layers> afterwards (indexer, the tied shared_head.head, embeds) is
    dropped by _should_skip_key's layer-index threshold."""
    prefix = f"model.layers.{num_layers}."
    if not name.startswith(prefix):
        return name
    rest = name[len(prefix) :]
    if ".indexer" in rest:
        return name
    if rest.startswith(("enorm", "hnorm", "eh_proj")):
        return f"model.mtp.{rest}"
    if rest.startswith("shared_head.norm"):
        return "model.mtp.shared_head_norm" + rest[len("shared_head.norm") :]
    if rest.startswith(("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")):
        return f"model.mtp.decoder.{rest}"
    return name


def _should_skip_key(name: str, num_layers: int) -> bool:
    """Drop weights not modeled in mini-sglang: the DSA lightning indexer (for
    seq_len <= index_topk attention is dense, so the indexer is unused) and the
    MTP / NextN speculative layers appended after the main decoder stack."""
    if ".indexer" in name or "indexers_proj" in name:
        return True
    m = _LAYER_IDX_PATTERN.match(name)
    if m is not None and int(m.group(1)) >= num_layers:
        return True
    return False


def _dequant_block_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantize a block-wise FP8 (e4m3) weight to bf16.

    w_bf16[i, j] = w_fp8[i, j] * scale_inv[i // block, j // block]
    """
    out_f, in_f = weight.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0)[:out_f, :]
    s = s.repeat_interleave(block, dim=1)[:, :in_f]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key.

    weight:  ...experts.{i}.gate_up_proj.weight            -> ...experts.gate_up_proj
    fp8 scale: ...experts.{i}.gate_up_proj.weight_scale_inv -> ...experts.gate_up_proj_scale_inv
    """
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(_SCALE_SUFFIX):
        packed_name = packed_name[: -len(_SCALE_SUFFIX)] + "_scale_inv"
    elif packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # In FP8 mode the model keeps weights in fp8 and consumes the block scales as
    # separate (sharded/merged/stacked) tensors; otherwise scales are folded in via dequant.
    is_fp8 = config.is_fp8

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                stripped = name.removeprefix("language_model.")
                if config.num_nextn > 0:
                    stripped = _remap_mtp_key(stripped, config.num_layers)
                # Only the routed experts stay FP8; all other fp8 weights are dequantized to bf16.
                fp8_keep = is_fp8 and _EXPERT_PATTERN.match(stripped) is not None
                if name.endswith(_SCALE_SUFFIX):
                    if not fp8_keep:  # scale consumed by dequant (or bf16 mode) -> drop
                        continue
                if _should_skip_key(stripped, config.num_layers):
                    continue
                raw = f.get_tensor(name)
                if raw.dtype == torch.float8_e4m3fn and not fp8_keep:
                    scale_key = name[: -len(".weight")] + _SCALE_SUFFIX
                    raw = _dequant_block_fp8(raw, f.get_tensor(scale_key))
                name = stripped
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
