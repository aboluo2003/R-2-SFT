"""Distributed training utilities for FSDP model sharding and checkpoint loading."""

import re
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from onerec_llm.utils.ds_utils import format_dict_or_list
from onerec_llm.utils.distributed import get_world_size_and_rank

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import DeviceMesh


def _parse_pattern_list(patterns_str: Optional[str]) -> List[str]:
    """Parse comma-separated pattern string into a list of non-empty patterns.

    Args:
        patterns_str: Comma-separated patterns (e.g. 'mtp.*,visual.*'), or None.

    Returns:
        List of pattern strings (empty list if None or empty string).
    """
    if patterns_str is None or patterns_str.strip() == '':
        return []
    return [p.strip() for p in patterns_str.split(',') if p.strip()]


def _match_any_pattern(key: str, patterns: List[str]) -> bool:
    """Check if a key matches any of the regex patterns.

    Args:
        key: Parameter name to check.
        patterns: List of regex pattern strings.

    Returns:
        True if key matches at least one pattern.
    """
    for pat in patterns:
        if re.fullmatch(pat, key):
            return True
    return False


def shard_model(
    model: nn.Module,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: Optional[DeviceMesh] = None,
    fp32_weight: bool = True,
    model_class: str = 'Qwen3ForCausalLM',
    fp32_reduce: bool = True
) -> None:
    """Shard a model with FSDP using the PyTorch Distributed fully_shard API.
    
    Args:
        model: Model to shard with FSDP.
        cpu_offload: If True, FSDP will offload parameters to CPU.
        reshard_after_forward: Whether to reshard after forward pass.
        dp_mesh: Device mesh for FSDP sharding under multiple parallelism.
        fp32_weight: If True, use fp32 for weights with bfloat16 params.
        model_class: Model class name. Supports 'Qwen3ForCausalLM' and
            'Qwen3_5ForConditionalGeneration'.
        fp32_reduce: If True, use fp32 for gradient reduction.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    
    if fp32_weight:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32 if fp32_reduce else torch.bfloat16
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if model_class == 'Qwen3ForCausalLM':
        layers = list(model.model.layers)
    elif model_class in ('Qwen3_5ForConditionalGeneration', 'Qwen3_5ForConditionalGeneration_sp', 'Qwen3_5ForConditionalGeneration_sp2'):
        # Structure: ConditionalGeneration → .model (Qwen3_5Model) → .language_model (TextModel) → .layers
        layers = list(model.model.language_model.layers)
    else:
        raise ValueError(f"Unsupported model_class: {model_class}")
    
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    
    fully_shard(model, **fsdp_kwargs)

    # Set up forward prefetch for layers
    prev = None
    for layer in reversed(layers):
        if prev is not None:
            layer.set_modules_to_forward_prefetch([prev])
        prev = layer
    model.set_modules_to_forward_prefetch([prev])


def load_from_full_model_state_dict(
    model: "FSDPModule",
    full_sd: Dict[str, Any],
    allow_random_init_params: Optional[str] = None,
    ignore_param_patterns: Optional[str] = None,
    use_tie_weights: bool = False,
    tie_weights_key_map: Optional[Dict[str, str]] = None,
) -> None:
    """Load full state dict into an FSDP-sharded model.

    Args:
        model: FSDP-sharded model to load into.
        full_sd: Full (unsharded) state dictionary.
        allow_random_init_params: Comma-separated regex patterns for parameter names
            that can be randomly initialized if not found in full_sd.
            Supports regex patterns (e.g. 'mtp.*'). Default: None.
        ignore_param_patterns: Comma-separated regex patterns for parameter names
            that should be completely ignored (removed from full_sd).
            Supports regex patterns (e.g. 'mtp.*,visual.*'). Default: None.
        use_tie_weights: If True, tie lm_head.weight to model.embed_tokens.weight.
        tie_weights_key_map: Optional dict mapping tied weight key names.
            e.g. {"lm_head.weight": "model.language_model.embed_tokens.weight"}
            If None and use_tie_weights=True, defaults to tying
            lm_head.weight to model.embed_tokens.weight.
    """
    allow_random_patterns = _parse_pattern_list(allow_random_init_params)
    ignore_patterns = _parse_pattern_list(ignore_param_patterns)

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    if dist.get_rank() == 0:
        # ── Step 1: Handle tie weights ─────────────────────────────────────────
        if use_tie_weights:
            if tie_weights_key_map is not None:
                # Use provided key mapping (e.g. for Qwen3_5ForConditionalGeneration)
                for dst_key, src_key in tie_weights_key_map.items():
                    if src_key in full_sd and dst_key not in full_sd:
                        full_sd[dst_key] = full_sd[src_key]
            else:
                # Default: tie lm_head.weight to model.embed_tokens.weight (Qwen3)
                embed_key = 'model.embed_tokens.weight'
                if embed_key in full_sd:
                    full_sd['lm_head.weight'] = full_sd[embed_key]

        # ── Step 2: Remove ignored params from full_sd ──────────────────────────
        # These params exist in the checkpoint but we want to completely skip them.
        # They won't be loaded into the model.
        ignored_keys = [k for k in full_sd if _match_any_pattern(k, ignore_patterns)]
        for k in ignored_keys:
            del full_sd[k]

        # ── Step 3: Compute key differences ────────────────────────────────────
        extra_meta_keys = set(meta_sharded_sd.keys()) - set(full_sd.keys())
        extra_full_keys = set(full_sd.keys()) - set(meta_sharded_sd.keys())

        extra_meta_info = {
            k: (v.shape, v.device, v.dtype)
            for k, v in meta_sharded_sd.items()
            if k in extra_meta_keys
        }
        extra_full_info = {
            k: (v.shape, v.device, v.dtype)
            for k, v in full_sd.items()
            if k in extra_full_keys
        }

        # ── Step 4: Random init for missing params that match allow patterns ───
        # Params in model but not in checkpoint: if they match allow_random_init
        # patterns, we initialize them randomly instead of raising an error.
        device0 = full_sd[list(full_sd)[0]]
        for k in list(extra_meta_keys):
            if _match_any_pattern(k, allow_random_patterns):
                shape, _, dtype = extra_meta_info[k]
                full_sd[k] = torch.rand(shape) * 0.1
                if full_sd[k].ndim >= 2:
                    nn.init.kaiming_normal_(full_sd[k], a=0, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.zeros_(full_sd[k])
                full_sd[k] = full_sd[k].to(device=device0, dtype=dtype)
                # Remove from extra since we've added it to full_sd
                extra_meta_keys.discard(k)
                extra_meta_info.pop(k, None)

        # ── Step 5: Validate key sets match ────────────────────────────────────
        # After all adjustments, the two dicts should have the same keys.
        assert len(meta_sharded_sd) == len(full_sd), (
            f"Sharded State Dict doesn't equal to Full State Dict, "
            f"{len(meta_sharded_sd)} vs {len(full_sd)}\n"
            f"extra_in_model={format_dict_or_list(extra_meta_info)}, "
            f"extra_in_checkpoint={format_dict_or_list(extra_full_info)}"
        )
        assert sorted(list(meta_sharded_sd.keys())) == sorted(list(full_sd.keys())), \
            "Keys of Sharded State Dict doesn't equal to Full State Dict"

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if dist.get_rank() == 0:
            full_tensor = full_sd[param_name].detach().cuda().type(sharded_meta_param.dtype)
        else:
            full_tensor = torch.empty(
                sharded_meta_param.size(),
                device="cuda",
                dtype=sharded_meta_param.dtype,
            )
        
        mesh = sharded_meta_param.device_mesh
        dist.broadcast(full_tensor, src=0, group=mesh.get_group(0))
        dist.barrier()
        
        sharded_tensor = distribute_tensor(
            full_tensor, mesh, sharded_meta_param.placements
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)

    model.load_state_dict(sharded_sd, assign=True)