from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # ---- MLA / DeepSeek-style MoE extras (0 / defaults for non-MLA models) ----
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    n_shared_experts: int = 0
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    is_fp8: bool = False
    is_fp4: bool = False
    num_nextn: int = 0

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # MLA / DeepSeek-MoE extras
        q_lora_rank = getattr(config, "q_lora_rank", 0) or 0
        kv_lora_rank = getattr(config, "kv_lora_rank", 0) or 0
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0) or 0
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0) or 0
        v_head_dim = getattr(config, "v_head_dim", 0) or 0
        n_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0) or 0
        routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0) or 1.0
        n_group = getattr(config, "n_group", 1) or 1
        topk_group = getattr(config, "topk_group", 1) or 1

        qc = getattr(config, "quantization_config", None)
        if isinstance(qc, dict):
            is_fp8 = qc.get("quant_method") == "fp8"
        else:
            is_fp8 = getattr(qc, "quant_method", None) == "fp8"
        # compressed-tensors NVFP4 (e.g. nvidia/GLM-5.2-NVFP4): experts are
        # W4A4 group-16, everything else stays high precision
        def _is_nvfp4(q):
            groups = (q.get("config_groups", {}) if isinstance(q, dict) else getattr(q, "config_groups", {}) or {})
            for g in groups.values():
                w = g.get("weights", {}) if isinstance(g, dict) else {}
                if w.get("num_bits") == 4 and w.get("type") == "float":
                    return True
            return False

        is_fp4 = qc is not None and _is_nvfp4(qc)

        if kv_lora_rank > 0:  # MLA (e.g. glm_moe_dsa / deepseek): use materialized head dim
            num_experts = getattr(config, "n_routed_experts", num_experts)
            # NOTE: in GlmMoeDsaConfig `head_dim` is an alias for `qk_rope_head_dim`, so the
            # raw qk_rope attribute is unreliable. Derive it from qk_head_dim - qk_nope.
            qk_head_dim = getattr(config, "qk_head_dim", 0) or (qk_nope_head_dim + qk_rope_head_dim)
            qk_rope_head_dim = qk_head_dim - qk_nope_head_dim
            head_dim = qk_head_dim
            num_kv_heads = config.num_attention_heads
            rope_params = getattr(config, "rope_parameters", None) or {}
            rope_theta = rope_params.get("rope_theta", None) or getattr(config, "rope_theta", 10000.0)
            rope_scaling = None
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
            rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_nextn=getattr(config, "num_nextn_predict_layers", 0) or 0,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            n_shared_experts=n_shared_experts,
            first_k_dense_replace=first_k_dense_replace,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            is_fp8=is_fp8,
            is_fp4=is_fp4,
        )
