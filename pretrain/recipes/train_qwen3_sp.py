"""Qwen3 Training Script

Multi-node, multi-GPU training script for Qwen3 models using FSDP (Fully Sharded Data Parallel).
Supports distributed training, checkpointing, and comprehensive monitoring.
"""

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from onerec_llm.models.qwen3_5.modeling_qwen3_5_sp import Qwen3_5ForConditionalGeneration
import datetime
import argparse
import contextlib
import gc
import itertools
import json
import logging
import time
from functools import partial
from typing import Dict, Optional, Tuple
import torch

import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer

from onerec_llm.data.dataloaders import get_dataloader
from onerec_llm.losses import CrossEntropyLoss, ChunkedLossComputer
from onerec_llm.training.activations import set_activation_checkpointing
from onerec_llm.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    load_hf_checkpoint,
    save_model_checkpoint,
    load_checkpoint,
    init_profiler,
    initialize_distributed,
)
from onerec_llm.training.common import set_default_dtype
from onerec_llm.training.gradients import (
    EmbeddingGradientMasker,
    clip_grad_norm,
    compute_fsdp_zero2_grad_norm,
)
from onerec_llm.training.distributed import (
    load_from_full_model_state_dict,
    shard_model,
)
from onerec_llm.training.lr_schedulers import get_scheduler
from onerec_llm.training.logging import TrainingMetrics, TensorBoardLogger
from onerec_llm.training.sp_utils import SPBatchIterator
from onerec_llm.utils.common import (
    Timer,
    dist_reduce_dict,
    get_optimizer_grouped_parameters,
    print_rank_0,
    set_random_seed,
    to_cuda,
)
from onerec_llm.utils.ds_utils import format_dict_or_list, print_input_info
from onerec_llm.utils.mfu_stats import MFUStats
from onerec_llm.utils.time_tracker import TimeTracker

# Disable garbage collection for performance
gc.disable()

# Set CUDA memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Qwen3 Training Script")
    
    # Checkpoint arguments
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Directory of the pretrained model")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Checkpoint directory to resume from")
    parser.add_argument("--resume_from_tag", type=str, default=None,
                       help="Checkpoint tag to resume from")
    parser.add_argument("--resume_training_state", action="store_true",
                       help="Whether to resume training state including optimizer, scheduler, and dataloader")
    parser.add_argument("--use_fp32_weight", action="store_true",
                       help="Use fp32 for model weight updating")
    parser.add_argument("--use_fp32_reduce", action="store_true",
                       help="Use fp32 for gradient reduction")
    parser.add_argument("--reshard_after_forward", action="store_true",
                       help="Enable reshard_after_forward to enable Zero3")
    parser.add_argument("--save_checkpoint_per_step", type=int, default=1000,
                       help="Number of steps to save a checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to write the trained model")
    parser.add_argument("--model_class", type=str, default="Qwen3ForCausalLM",
                       help="Model class name")
    
    # Dataset arguments
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Path to dataset configuration JSON file")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Max tokens per sentence")
    parser.add_argument("--minibatch_size", type=int, default=4096,
                       help="Minibatch size")
    parser.add_argument("--start_optimize_embedding_index", type=int, default=0,
                       help="Start optimize embedding index for finetuning")
    
    # Learning rate arguments
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr",
                       help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                       help="Number of warmup steps")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum learning rate after cosine schedule")
    
    # Optimizer arguments
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay for AdamW")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help="Beta2 for AdamW")
    
    # Training arguments
    parser.add_argument("--use_tie_weights", action="store_true",
                       help="Tie embedding and lm_head weights")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm (global L2); set to 0 to disable")
    parser.add_argument("--freeze_llm", action="store_true",
                       help="Freeze all LLM parameters")
    parser.add_argument("--freeze_sid_embed_tokens", action="store_true",
                       help="Freeze sid_embed_tokens")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--allow_random_init_params", type=str, default='',
                       help="Comma-separated regex patterns for params that can be randomly initialized "
                            "if not in checkpoint (e.g. 'mtp.*,visual.*')")
    parser.add_argument("--ignore_param_patterns", type=str, default='',
                       help="Comma-separated regex patterns for params to completely ignore from checkpoint "
                            "(e.g. 'mtp.*'). Ignored params are removed before loading.")
    parser.add_argument("--logging_per_step", type=int, default=100,
                       help="Number of steps to log training info")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed")
    parser.add_argument("--monitor_datasource_loss", action="store_true",
                       help="Monitor loss of each datasource")
    parser.add_argument("--monitor_datasource_cnt", action="store_true",
                       help="Monitor count of each datasource")
    parser.add_argument("--use_chunked_loss_computer", action="store_true",
                       help="Use chunked loss computer")
    parser.add_argument("--text_token_weight", type=float, default=1.0,
                       help="Loss weight multiplier for text (non-itemic) token positions. "
                            "Itemic token positions use itemic_token_weight. "
                            "Set < 1.0 to down-weight text tokens relative to itemic tokens. "
                            "Default: 1.0 (equal weighting).")
    def _parse_itemic_token_weight(s):
        """Parse itemic_token_weight from string. Supports comma or space separated values."""
        if s is None or s == '':
            return None
        # Handle comma-separated or space-separated values
        values = s.replace(',', ' ').split()
        return [float(v) for v in values]
    
    parser.add_argument("--itemic_token_weight", type=_parse_itemic_token_weight, default=None,
                       help="Loss weight multipliers for itemic token segments. "
                            "Can be comma-separated (e.g., '4,1,4,1') or space-separated (e.g., '4 1 4 1'). "
                            "If itemic_id_mask has values 0,1,2,3... then: "
                            "- 0 (text) -> text_token_weight, "
                            "- 1 (segment a) -> itemic_token_weight[0], "
                            "- 2 (segment b) -> itemic_token_weight[1], etc. "
                            "If not set, all itemic segments use weight 1.0.")
    
    # Profiling arguments
    parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable PyTorch profiler for performance analysis")
    
    # Sequence Parallel arguments
    parser.add_argument(
        "--sequence_parallel_group_size",
        type=int,
        default=1,
        help=(
            "Number of ranks per Sequence Parallel (Ulysses) group. "
            "Default 1 = disabled (standard data-parallel only). "
            "world_size must be divisible by this value. "
            "Example: --sequence_parallel_group_size 4  means every 4 consecutive "
            "ranks form one SP group."
        ),
    )
    
    return parser


class StateDictConverter:
    """Converter for state dict transformations (identity by default)."""
    
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert state dict (e.g., for loading)."""
        return state_dict
    
    def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Revert state dict (e.g., for saving)."""
        return state_dict


class FirstForwardLogger:
    def __init__(self):
        self.t = 0

    def __call__(self, *args, **kwargs):
        if self.t == 0:
            print("First forward:", args, kwargs)
        self.t += 1


def initialize_model(
    args,
    device_mesh: DeviceMesh,
    state_dict: Optional[Dict[str, torch.Tensor]],
    converter: StateDictConverter,
) -> torch.nn.Module:
    """Initialize and shard model.
    
    Args:
        args: Training arguments
        device_mesh: Device mesh for distributed training
        state_dict: Optional pretrained state dict
        converter: State dict converter
    
    Returns:
        Initialized and sharded model
    """
    # Create model on meta device
    with set_default_dtype(torch.bfloat16), torch.device("meta"), init_empty_weights():
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        print(f"config: {config}")
        config._attn_implementation = "flash_attention_2"
        config.use_cache = False
        if hasattr(config, "text_config"):
            config.text_config.use_cache = False
        config.chunked_loss_computer = args.use_chunked_loss_computer
        model = eval(args.model_class)(config)
    
    # Verify all parameters are on meta device
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        assert tensor.device == torch.device("meta"), "All tensors must be on meta device"
    
    # Enable gradient checkpointing if requested
    if args.enable_gradient_checkpointing:
        print_rank_0("Enable gradient checkpointing")
        set_activation_checkpointing(
            model, auto_wrap_policy=eval(args.model_class).wrap_modules
        )
    
    # Convert to fp32 if needed
    if args.use_fp32_weight:
        model = model.float()
    
    # Shard model with FSDP
    shard_model(
        model=model,
        cpu_offload=False,
        reshard_after_forward=args.reshard_after_forward,
        dp_mesh=device_mesh,
        fp32_weight=args.use_fp32_weight,
        model_class=args.model_class,
        fp32_reduce=args.use_fp32_reduce
    )
    dist.barrier()
    
    # Load state dict
    with Timer("Load state dict"):
        # Build tie_weights_key_map based on model class
        tie_key_map = None
        if args.use_tie_weights and hasattr(model.model, 'language_model'):
            # Qwen3_5ForConditionalGeneration: lm_head tied to language_model.embed_tokens
            tie_key_map = {
                "lm_head.weight": "model.language_model.embed_tokens.weight"
            }
        load_from_full_model_state_dict(
            model=model,
            full_sd=state_dict,
            allow_random_init_params=args.allow_random_init_params,
            ignore_param_patterns=args.ignore_param_patterns,
            use_tie_weights=args.use_tie_weights,
            tie_weights_key_map=tie_key_map,
        )
    
    # Tie weights if requested
    # Sharing weights between embedding and output projection can reduce parameters
    # and improve training stability for some models
    if args.use_tie_weights:
        # Different model classes have different embed_tokens paths:
        #   Qwen3ForCausalLM               → model.model.embed_tokens
        
        #   Qwen3_5ForConditionalGeneration → model.model.language_model.embed_tokens
        if hasattr(model.model, 'language_model'):
            embed_tokens = model.model.language_model.embed_tokens
        else:
            embed_tokens = model.model.embed_tokens
        model.lm_head.weight = embed_tokens.weight
        # Verify weight tying: check if there are any differences (should be ~0)
        diff_weight = model.lm_head.weight - embed_tokens.weight
        diff_weight_cnt = (diff_weight.full_tensor().abs() > 1e-6).float().sum()
        print_rank_0(
            f"diff_weight_cnt: {diff_weight_cnt.item()}, "
            f"diff_weight_ratio: {diff_weight_cnt.item() / model.lm_head.weight.numel():.4f}"
        )
    
    # Initialize RoPE
    # If the model has a dedicated rope_init_for_training method (e.g. Qwen3_5ForConditionalGeneration),
    # use it — it correctly handles rope_type='default' and vision/text RoPE differences.
    # Otherwise, fall back to the generic per-module initialization logic.
    if hasattr(model, "rope_init_for_training"):
        print_rank_0("Initialize RoPE via model.rope_init_for_training()")
        with torch.device(torch.cuda.current_device()):
            model.rope_init_for_training()
    else:
        with torch.device(torch.cuda.current_device()):
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    print_rank_0("Initialize RoPE")
                    m.rope_init()
                elif hasattr(m, "inv_freq"):
                    print_rank_0(f"Initialize RoPE inv_freq for {m.__class__.__name__}")
                    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                    rope_type = getattr(m, "rope_type", "default")
                    rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                    inv_freq, attention_scaling = rope_init_fn(
                        m.config, device=torch.cuda.current_device()
                    )
                    m.register_buffer("inv_freq", inv_freq, persistent=False)
                    m.attention_scaling = attention_scaling
    
    # Freeze parameters if requested
    # When freeze_llm is enabled, only embedding and output head are trainable
    # This is useful for embedding-only fine-tuning or when using start_optimize_embedding_index
    if args.freeze_llm:
        assert args.start_optimize_embedding_index > 0
        for name, param in model.named_parameters():
            # "embed_tokens" matches both "model.embed_tokens" (Qwen3) and
            # "language_model.embed_tokens" (Qwen3_5 conditional generation)
            if "embed_tokens" in name or "lm_head" in name or "sid_embed_linear" in name:
                param.requires_grad = True  # Only embeddings and output head are trainable
            elif "sid_embed_tokens" in name and not args.freeze_sid_embed_tokens:
                param.requires_grad = True
            else:
                param.requires_grad = False  # Freeze all transformer layers
    
    # Print trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"Trainable parameter: {name}")
    print_rank_0("=" * 50)
    
    return model


def compute_forward_backward(
    model: torch.nn.Module,
    batch: Dict,
    compute_loss_fn,
    loss_fn: CrossEntropyLoss,
    args,
    embedding_masker: Optional[EmbeddingGradientMasker],
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute forward and backward pass.
    
    Args:
        model: Model instance
        batch: Input batch
        compute_loss_fn: Loss computation function
        loss_fn: Loss function instance
        args: Training arguments
        embedding_masker: Optional embedding gradient masker
        optimizer: Optimizer instance
    
    Returns:
        Tuple of (loss, per_token_loss, grad_norm) where grad_norm is the
        pre-clip global L2 gradient norm (0.0 if clipping is disabled).
    """
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch.get("attention_mask", None)
    cu_seqlens = batch.get("cu_seqlens", None)
    position_ids = batch.get("position_ids", None)
    itemic_id_mask = batch.get("itemic_id_mask", None)
    max_length = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    # Prepare labels
    # Zero out padding tokens (input_ids <= 0) to avoid computing loss on them
    input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)
    
    # Forward pass
    with Timer(desc="Fwd"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            max_length_k=max_length,
            max_length_q=max_length
        )
        
        logits = output.logits
        
        # Shift labels for next token prediction
        # For causal LM, we predict token[i] given tokens[0:i], so labels need to be shifted
        # by one position: label[i] should correspond to input[i+1]
        pad = torch.full(
            (input_ids.shape[0], 1),
            loss_fn.ignore_index,
            dtype=input_ids.dtype
        ).to(device=input_ids.device, non_blocking=True)
        labels = torch.cat([input_ids[:, 1:], pad], dim=-1)
        # Update labels: use input_ids where loss_mask==1, ignore_index where loss_mask==0
        # This allows selective loss computation on specific tokens (e.g., excluding special tokens)
        labels = labels * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

        # Build per-token loss weight for text_token_weight and itemic_token_weight scaling.
        # itemic_id_mask values: 0=text, 1=segment_a, 2=segment_b, etc.
        # text tokens (0) → weight text_token_weight
        # itemic tokens (1,2,3...) → weight from itemic_token_weight list or default 1.0
        tokenwise_loss_weight = None
        text_token_weight = getattr(args, "text_token_weight", 1.0)
        itemic_token_weight = getattr(args, "itemic_token_weight", None)
        
        if itemic_id_mask is not None:
            # itemic_id_mask can be 0, 1, 2, 3... (multi-segment)
            # shape matches input_ids: [batch, seq_len]
            if itemic_token_weight is not None:
                # Use per-segment weights: itemic_token_weight[0] for mask=1, [1] for mask=2, etc.
                tokenwise_loss_weight = torch.zeros_like(itemic_id_mask, dtype=torch.float32)
                # Text tokens (mask=0) get text_token_weight
                tokenwise_loss_weight = torch.where(
                    itemic_id_mask == 0,
                    text_token_weight,
                    tokenwise_loss_weight
                )
                # Itemic tokens (mask>=1) get their respective weights
                for i, weight in enumerate(itemic_token_weight, start=1):
                    tokenwise_loss_weight = torch.where(
                        itemic_id_mask == i,
                        weight,
                        tokenwise_loss_weight
                    )
            elif text_token_weight != 1.0:
                # Legacy mode: binary mask (0=text, >=1=itemic)
                itemic_mask_f = (itemic_id_mask > 0).float()
                tokenwise_loss_weight = (
                    itemic_mask_f * 1.0
                    + (1.0 - itemic_mask_f) * text_token_weight
                )
            
            if tokenwise_loss_weight is not None:
                # Apply loss_mask: zero out tokens that should not contribute to loss
                tokenwise_loss_weight = tokenwise_loss_weight * loss_mask.float()
        
        loss, per_token_loss = compute_loss_fn(logits, labels=labels, tokenwise_loss_weight=tokenwise_loss_weight)
        if per_token_loss is not None:
            per_token_loss = per_token_loss.to(loss.device)
    
    # Backward pass
    with Timer("bwd"):
        # When using ChunkedLossComputer, forward_and_backward already performs
        # the full backward (chunked lm_head + transformer), and returns a
        # detached loss. Calling loss.backward() on a detached leaf is a no-op
        # but unnecessary. Skip it and sync CUDA to ensure all async FSDP ops
        # complete before the all_reduce in compute_metrics.
        if args.use_chunked_loss_computer:
            torch.cuda.synchronize()
        else:
            loss.backward()
        
        # Apply gradient mask for embedding layers if needed
        # When start_optimize_embedding_index > 0, only embeddings with index >= threshold are trainable
        # This allows progressive unfreezing of embeddings during training
        if args.start_optimize_embedding_index > 0 and embedding_masker is not None:
            embedding_masker.apply_gradient_mask(optimizer)
        
        # Compute grad norm BEFORE clipping so the logged value reflects the
        # true gradient magnitude, not the post-clip capped value.
        grad_norm = compute_fsdp_zero2_grad_norm(model)

        if args.max_grad_norm and args.max_grad_norm > 0:
            clip_grad_norm(model, args.max_grad_norm)
    
    return loss, per_token_loss, grad_norm


def compute_metrics(
    batch: Dict,
    loss: torch.Tensor,
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_fn: CrossEntropyLoss,
    args,
    metrics: TrainingMetrics,
) -> Tuple[float, float, float, int, int, int]:
    """Compute and accumulate training metrics.
    
    Args:
        batch: Input batch
        loss: Loss tensor
        per_token_loss: Per-token loss tensor
        loss_mask: Loss mask tensor
        loss_fn: Loss function instance
        args: Training arguments
        metrics: Training metrics tracker
    
    Returns:
        Tuple of (avg_loss, avg_itemic_token_loss, avg_text_token_loss,
                 num_tokens, num_samples, num_valid_tokens)
    """
    input_ids = batch["input_ids"]
    cu_seqlens = batch.get("cu_seqlens", None)
    itemic_id_mask = batch.get("itemic_id_mask", None)
    data_source = batch.get("data_source", None)
    sample_idx = batch["sample_idx"]
    
    # Compute token metrics
    token_count = input_ids.numel()
    num_samples = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
    
    # Calculate number of valid tokens (tokens with loss_mask == 1)
    # Works for both 1D (flattened) and 2D (batch, seq_len) loss_mask
    num_valid_tokens = (loss_mask == 1).sum().item()
    
    # Aggregate metrics across all ranks
    token_metrics = torch.tensor(
        [token_count, num_samples, num_valid_tokens]
    ).cuda(non_blocking=True)
    dist.all_reduce(
        token_metrics, op=dist.ReduceOp.SUM, group=None
    )
    num_tokens, num_samples, num_valid_tokens = (
        token_metrics.detach().cpu().numpy()
    )
    
    # Update metrics
    metrics.update(num_tokens, num_samples, num_valid_tokens)
    metrics.period_num_steps += 1
    
    # Compute average loss for this step
    avg_loss = loss.detach()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    metrics.period_sum_loss += avg_loss
    
    # Compute itemic and text token losses
    if itemic_id_mask is not None and per_token_loss is not None and per_token_loss.numel() > 0:
        itemic_id_mask = itemic_id_mask.view(per_token_loss.shape)
        # Binary mask: 0=text, >0=itemic (any segment)
        itemic_binary_mask = (itemic_id_mask > 0).float()
        avg_itemic_token_loss = (itemic_binary_mask * per_token_loss).sum() / (itemic_binary_mask.sum() + 1e-6)
        avg_text_token_loss = ((1 - itemic_binary_mask) * per_token_loss).sum() / ((1 - itemic_binary_mask).sum() + 1e-6)
        dist.all_reduce(avg_itemic_token_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_text_token_loss, op=dist.ReduceOp.SUM)
        avg_itemic_token_loss = avg_itemic_token_loss.item() / dist.get_world_size()
        avg_text_token_loss = avg_text_token_loss.item() / dist.get_world_size()
        # Count itemic/text tokens locally (no cross-GPU communication)
        metrics.period_num_itemic_tokens += int(itemic_binary_mask.sum().item())
        metrics.period_num_text_tokens   += int((1 - itemic_binary_mask).sum().item())
    else:
        avg_itemic_token_loss = 0.0
        avg_text_token_loss = avg_loss
    
    metrics.period_sum_itemic_token_loss += avg_itemic_token_loss
    metrics.period_sum_text_token_loss += avg_text_token_loss
    
    # Monitor data source metrics
    if args.monitor_datasource_loss and data_source is not None and per_token_loss is not None and per_token_loss.numel() > 0:
        local_sample_idx = sample_idx.squeeze()
        unique_sample_idx = local_sample_idx.unique()
        # Get local loss mask for valid token counting
        local_loss_mask = loss_mask.squeeze()
        # Get itemic_id_mask if available for separating itemic/text loss
        local_itemic_mask = None
        if itemic_id_mask is not None:
            local_itemic_mask = itemic_id_mask.view(per_token_loss.shape).squeeze()
        
        for s_idx in unique_sample_idx:
            if s_idx < 0:
                continue
            mask = local_sample_idx == s_idx
            sum_loss = per_token_loss[mask].sum()
            key = data_source[int(s_idx.item())]
            metrics.period_data_source_loss[key] += sum_loss.item()
            metrics.period_data_source_tokens[key] += mask.sum().item()
            # Count valid tokens using loss mask
            metrics.period_valid_data_source_tokens[key] += (
                mask[local_loss_mask != 0].sum().item()
            )
            
            # Separate itemic and text loss for this data source
            if local_itemic_mask is not None:
                itemic_mask_for_sample = mask & (local_itemic_mask > 0)
                text_mask_for_sample = mask & (local_itemic_mask == 0)
                
                sum_itemic_loss = per_token_loss[itemic_mask_for_sample].sum()
                sum_text_loss = per_token_loss[text_mask_for_sample].sum()
                
                metrics.period_data_source_itemic_loss[key] += sum_itemic_loss.item()
                metrics.period_data_source_text_loss[key] += sum_text_loss.item()
                metrics.period_data_source_itemic_tokens[key] += itemic_mask_for_sample.sum().item()
                metrics.period_data_source_text_tokens[key] += text_mask_for_sample.sum().item()
    
    if args.monitor_datasource_cnt and data_source is not None:
        for data_source_name in data_source:
            metrics.local_period_data_source_samples[data_source_name] += 1
    
    return avg_loss, avg_itemic_token_loss, avg_text_token_loss, int(num_tokens), int(num_samples), int(num_valid_tokens)


def log_training_step(
    global_step: int,
    metrics: TrainingMetrics,
    args,
    lr_scheduler,
    grad_norm: float,
    period_start_time: float,
    training_start_time: float,
    mfu_stats: MFUStats,
    step_time_tracker: TimeTracker,
    iteration_time_tracker: TimeTracker,
    epoch_idx: int,
    tb_logger: TensorBoardLogger,
    chunked_loss_computer: Optional[ChunkedLossComputer],
) -> float:
    """Log training step metrics.
    
    Args:
        global_step: Global training step
        metrics: Training metrics tracker
        args: Training arguments
        lr_scheduler: Learning rate scheduler
        grad_norm: Gradient norm
        period_start_time: Start time of the current logging period
        training_start_time: Start time of the entire training run
        mfu_stats: MFU statistics tracker
        step_time_tracker: Time tracker for training steps (tracks forward/backward/optimizer)
        iteration_time_tracker: Time tracker for data iteration (tracks data loading)
        epoch_idx: Current epoch index
        tb_logger: TensorBoard logger
        chunked_loss_computer: Optional chunked loss computer
    
    Returns:
        Updated period_start_time for next logging period
    """
    end_time = time.time()
    model_lrs = lr_scheduler.get_last_lr()
    learning_rate = model_lrs[0]
    
    # Compute performance metrics for the logging period
    period_duration = end_time - period_start_time
    period_num_steps = max(metrics.period_num_steps, 1)  # Avoid division by zero
    
    # Current period metrics (_current): reflect performance in the current logging period
    # These metrics show recent performance and can fluctuate with short-term variations
    sec_per_step = period_duration / period_num_steps
    tokens_per_sec_per_gpu_current = (
        metrics.period_num_tokens / period_duration / dist.get_world_size()
    )
    samples_per_sec_per_gpu_current = (
        metrics.period_num_samples / period_duration / dist.get_world_size()
    )
    samples_per_step_per_gpu_current = (
        metrics.period_num_samples / period_num_steps / dist.get_world_size()
    )
    valid_tokens_per_sec_per_gpu_current = (
        metrics.period_num_valid_tokens / period_duration / dist.get_world_size()
    )
    
    # Average metrics (_avg): reflect average performance over entire training
    # These metrics smooth out short-term fluctuations and include all overhead
    # (checkpoint saving, logging, etc.), providing a more stable view of overall performance
    samples_per_step_per_gpu_avg = (
        metrics.total_num_samples / dist.get_world_size() / max(global_step, 1)
    )
    samples_per_sec_per_gpu_avg = (
        metrics.total_num_samples / dist.get_world_size() / (end_time - training_start_time)
    )
    tokens_per_step_per_gpu_avg = (
        metrics.total_num_tokens / dist.get_world_size() / max(global_step, 1)
    )
    tokens_per_sec_per_gpu_avg = (
        metrics.total_num_tokens / dist.get_world_size() / (end_time - training_start_time)
    )
    
    # Compute average losses over the logging period
    avg_loss = metrics.period_sum_loss / period_num_steps
    avg_itemic_token_loss = metrics.period_sum_itemic_token_loss / period_num_steps
    avg_text_token_loss = metrics.period_sum_text_token_loss / period_num_steps
    
    # Reduce data source metrics across all ranks
    period_data_source_loss = dist_reduce_dict(metrics.period_data_source_loss)
    period_data_source_itemic_loss = dist_reduce_dict(metrics.period_data_source_itemic_loss)
    period_data_source_text_loss = dist_reduce_dict(metrics.period_data_source_text_loss)
    period_data_source_tokens = dist_reduce_dict(metrics.period_data_source_tokens)
    period_data_source_itemic_tokens = dist_reduce_dict(metrics.period_data_source_itemic_tokens)
    period_data_source_text_tokens = dist_reduce_dict(metrics.period_data_source_text_tokens)
    period_valid_data_source_tokens = dist_reduce_dict(metrics.period_valid_data_source_tokens)
    total_data_source_samples = dist_reduce_dict(
        metrics.local_period_data_source_samples, group=None
    )
    # Update total data source tokens
    for ds_key, ds_num_tokens in period_data_source_tokens.items():
        metrics.total_data_source_tokens[ds_key] += ds_num_tokens
    
    # Build log dictionary
    log_dict = {
        "training/loss": avg_loss,
        "training/itemic_token_loss": avg_itemic_token_loss,
        "training/text_token_loss": avg_text_token_loss,
        "training/itemic_token_ratio": (
            metrics.period_num_itemic_tokens /
            max(metrics.period_num_itemic_tokens + metrics.period_num_text_tokens, 1)
        ),
        "training/grad_norm": grad_norm,
        "training/learning_rate": learning_rate,
        "perf/sec_per_step": sec_per_step,
        "perf/tokens_per_sec_per_gpu_current": tokens_per_sec_per_gpu_current,
        "perf/samples_per_sec_per_gpu_current": samples_per_sec_per_gpu_current,
        "perf/total_num_tokens": metrics.total_num_tokens,
        "perf/total_num_samples": metrics.total_num_samples,
        "perf/num_sample_per_gpu": metrics.total_num_samples / dist.get_world_size(),
        "perf/samples_per_step_per_gpu_current": samples_per_step_per_gpu_current,
        # Note: num_sample_per_sec_per_gpu is the same as samples_per_sec_per_gpu_current
        # Keeping for backward compatibility, but samples_per_sec_per_gpu_current should be used
        "perf/num_sample_per_sec_per_gpu": samples_per_sec_per_gpu_current,
        "perf/valid_total_num_tokens": metrics.total_num_valid_tokens,
        "perf/valid_tokens_per_sec_per_gpu_current": valid_tokens_per_sec_per_gpu_current,
        "perf/valid_token_ratio": metrics.total_num_valid_tokens / metrics.total_num_tokens,
        **mfu_stats.mfu(period_duration, global_step),
        "perf/samples_per_step_per_gpu_avg": samples_per_step_per_gpu_avg,
        "perf/samples_per_sec_per_gpu_avg": samples_per_sec_per_gpu_avg,
        "perf/tokens_per_step_per_gpu_avg": tokens_per_step_per_gpu_avg,
        "perf/tokens_per_sec_per_gpu_avg": tokens_per_sec_per_gpu_avg,
        "perf/epoch_idx": epoch_idx,
    }
    
    # Get ticker statistics
    ticker_stats = {}
    for t in [step_time_tracker, iteration_time_tracker]:
        ticker_stats.update(t.stat())
    
    # Log to TensorBoard
    tb_logger.log(
        global_step,
        log_dict,
        ticker_stats,
        period_data_source_loss,
        period_data_source_itemic_loss,
        period_data_source_text_loss,
        period_data_source_itemic_tokens,
        period_data_source_text_tokens,
        period_data_source_tokens,
        total_data_source_samples,
        monitor_loss=args.monitor_datasource_loss,
        monitor_cnt=args.monitor_datasource_cnt,
    )
    
    # Print to console
    print_rank_0(
        f"Step: {global_step}, Loss: {avg_loss:.4f}, "
        f"Learning Rate: {learning_rate:.2e}, "
        f"Grad Norm: {grad_norm:.4f}, "
        f"Sec per Step: {sec_per_step:.4f}",
        format_dict_or_list(log_dict),
        "\n",
        format_dict_or_list({
            "mfu_stats": mfu_stats.mfu_per_step_per_gpu,
            "step_time_tracker": step_time_tracker.stat()
        }),
        "\n",
        chunked_loss_computer.ticker.stat() if chunked_loss_computer else "",
    )
    
    return end_time


def train():
    """Main training function."""
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    assert args.learning_rate > 0.0, "Learning rate must be positive"
    assert args.save_checkpoint_per_step > 0, "save_checkpoint_per_step must be positive"
    
    # Initialize distributed training
    rank, world_size, local_rank = initialize_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
    
    set_random_seed(args.seed)
    
    # ── Sequence Parallel 初始化（sequence_parallel_group_size > 1 时启用）──────
    sp_size = getattr(args, "sequence_parallel_group_size", 1)
    if sp_size > 1:
        assert world_size % sp_size == 0, (
            f"world_size ({world_size}) 必须能被 sequence_parallel_group_size ({sp_size}) 整除"
        )
        # 动态导入 SP 模型模块（仅在 SP 模式下才需要）
        from onerec_llm.models.qwen3_5.modeling_qwen3_5_sp import (
            initialize_sequence_parallel,
        )
        initialize_sequence_parallel(sp_size)
        if rank == 0:
            logger.info(
                f"[SP] Sequence Parallel enabled: sp_size={sp_size}, "
                f"num_sp_groups={world_size // sp_size}"
            )
    else:
        if rank == 0:
            logger.info("[SP] Sequence Parallel disabled (sequence_parallel_group_size=1)")
    
    # Load dataset configuration
    logger.info(f"Loading dataset config from: {args.dataset_config}")
    with open(args.dataset_config, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset = dataset_config.pop("name")
    dataset_config["model_class"] = args.model_class
    if args.max_length:
        dataset_config["max_length"] = args.max_length
    
    # Load pretrained checkpoint
    converter = StateDictConverter()
    state_dict = None
    if dist.get_rank() == 0:
        with set_default_dtype(torch.bfloat16):
            state_dict = load_hf_checkpoint(args.model_dir)
            state_dict = converter.convert(state_dict)
    dist.barrier()
    
    # Save training arguments
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if dist.get_rank() == 0:
        args_dict = vars(args)
        args_str = json.dumps(args_dict, indent=4, ensure_ascii=False)
        print_rank_0(f"Training Arguments:\n{args_str}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            os.path.join(args.output_dir, f"args-{timestamp}.json"),
            'w', encoding="utf-8"
        ) as f:
            f.write(args_str + "\n")
    
    # Initialize TensorBoard
    tb_writer = None
    if dist.get_rank() == 0:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
    
    # Initialize model
    model = initialize_model(args, device_mesh, state_dict, converter)
    if state_dict is not None:
        del state_dict
    
    # Initialize optimizer
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=1.0e-8
    )
    
    # Initialize embedding gradient masker
    # This allows selective training of embeddings based on token index
    # Useful for fine-tuning where only certain token embeddings should be updated
    embedding_masker = EmbeddingGradientMasker(
        model, model.config, args.start_optimize_embedding_index
    )
    if args.start_optimize_embedding_index > 0:
        # Save frozen embedding parameters to restore after optimizer step
        # This prevents optimizer from updating frozen embeddings
        embedding_masker.save_frozen_params()
    
    # Initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        min_lr=args.min_lr
    )
    
    # Initialize checkpointing
    app_state = AppState(model=model)
    dist_checkpointer = DistributedCheckpointer()
    
    # Load checkpoint if resuming
    dataloader_state_dict, global_step = load_checkpoint(
        args, app_state, dist_checkpointer, converter, optimizer, lr_scheduler
    )
    dist.barrier()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # Save dataset configuration
    if dist.get_rank() == 0:
        with open(
            os.path.join(args.output_dir, f"dataset-{timestamp}.json"),
            'w', encoding="utf-8"
        ) as f:
            f.write(json.dumps(dataset_config, ensure_ascii=False, indent=2) + "\n")
    
    # Build dataloader
    with Timer("Build dataloader"):
        try:
            dataloader = get_dataloader(name=dataset, **dataset_config)
        except Exception as e:
            logger.error(f"Failed to build dataloader: {e}", exc_info=True)
            raise
        if args.resume_training_state and dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)
    
    # Initialize profiler
    torch_profiler = init_profiler(
        output_dir=os.path.join(args.output_dir, "torch_profile"),
        enable=args.enable_profiler
    )
    
    # Initialize loss function
    loss_fn = CrossEntropyLoss(
        ignore_index=-100, return_token_loss=True, shift_labels=False
    )
    compute_loss_fn = loss_fn
    chunked_loss_computer = None
    if args.use_chunked_loss_computer:
        chunked_loss_computer = ChunkedLossComputer(
            lm_head=model.lm_head,
            loss_fn=loss_fn,
            minibatch_size=args.minibatch_size,
            shift_labels=False
        )
        compute_loss_fn = chunked_loss_computer.forward_and_backward
        # Tell the model to return hidden_states instead of logits,
        # so ChunkedLossComputer can apply lm_head internally.
        # Qwen3_5ForConditionalGeneration checks this attribute in its forward:
        #   if getattr(self, 'chunked_loss_computer', False):
        #       logits = hidden_states  # skip lm_head
        model.chunked_loss_computer = True
    
    # Initialize training state
    training_start_time = time.time()
    period_start_time = training_start_time
    remaining_debug_samples = 1  # Number of sample batches to print for debugging
    # Only reset global_step if not resuming from checkpoint
    # If resume_from_tag exists, global_step is already set in load_checkpoint
    if args.resume_from_tag is None:
        global_step = 0
    
    metrics = TrainingMetrics()
    mfu_stats = MFUStats(args)
    # step_time_tracker: tracks time for training steps (forward/backward/optimizer)
    step_time_tracker = TimeTracker(n=args.logging_per_step)
    # iteration_time_tracker: tracks time for data iteration (data loading)
    iteration_time_tracker = TimeTracker(n=args.logging_per_step)
    tb_logger = TensorBoardLogger(tb_writer)
    
    # Create data iterator
    data_iter = iter(dataloader)
    # ── SP batch 处理 ──────────────────────────────────────────────────────────
    # SPBatchIterator 封装了 all-gather + seq 切分逻辑：
    #   sp_size > 1: SP 组内各 rank 的 batch 先 all-gather 成列表，
    #                然后取本 sp_rank 对应的 batch，沿 seq 维度切分
    #   sp_size = 1: 退化为直接取 next(data_iter)，与原逻辑完全一致
    sp_batch_iter = SPBatchIterator(data_iter, debug=False)
    first_forward_logger = FirstForwardLogger()

    # Training loop
    while True:
        with contextlib.ExitStack() as ctx:
            if torch_profiler:
                ctx.enter_context(torch_profiler)
            
            step_time_tracker.tick("enter_context(torch_profiler)")
            _t0_data = time.perf_counter()
            try:
                batch = sp_batch_iter.next()
                cu_seqlens_cpu = batch.get("cu_seqlens", None)
            except StopIteration:
                break
            _data_load_ms = (time.perf_counter() - _t0_data) * 1000
            step_time_tracker.tick("next_batch")
            if dist.get_rank() == 0:
                _ids_shape = tuple(batch["input_ids"].shape)
                logger.warning(
                    f"[data_load] {_data_load_ms:.1f} ms  "
                    f"input_ids={_ids_shape}"
                )
            # Show sample data for debugging
            # Only print from first 8 ranks to avoid log spam (rank 0-7)
            # Sleep based on rank to stagger output and make logs easier to read
            if remaining_debug_samples > 0 and dist.get_rank() <= 8:
                with Timer("Show data"):
                    input_text = tokenizer.decode(batch['input_ids'][0])
                    # Stagger output by rank to avoid interleaved prints (0.3s per rank)
                    time.sleep(float(dist.get_rank()) * 0.3)
                    print(f"Input Text:\n\n{input_text}\n" + "=" * 100 + "\n\n")
                    print_input_info(batch, f"rank{dist.get_rank()}")
                    remaining_debug_samples -= 1
            
            # Move batch to CUDA
            to_cuda(batch)
            batch["cu_seqlens"] = cu_seqlens_cpu
            step_time_tracker.tick("to_cuda(batch)")
            
            # Update MFU stats
            token_count = batch["input_ids"].numel()
            num_samples = len(batch.get("cu_seqlens", [0, 1])) - 1
            mfu_stats.set(num_tokens=token_count, num_samples=num_samples)
            
            first_forward_logger(f"rank{dist.get_rank()} ready to forward")
            # Forward and backward pass
            loss, per_token_loss, grad_norm = compute_forward_backward(
                model, batch, compute_loss_fn, loss_fn, args,
                embedding_masker, optimizer
            )

            # Compute metrics
            epoch_idx = batch.get("epoch_idx", torch.tensor([0])).cpu().item()
            avg_loss, avg_itemic_token_loss, avg_text_token_loss, num_tokens, num_samples, num_valid_tokens = compute_metrics(
                batch, loss, per_token_loss, batch["loss_mask"], loss_fn, args, metrics
            )
            
            step_time_tracker.tick("compute_metrics")
            
            # Optimizer step
            # grad_norm is already computed (pre-clip) inside compute_forward_backward.
            optimizer.step()
            
            # Restore frozen parameters after optimizer step
            # This ensures frozen embeddings are not modified by the optimizer
            # even if they were included in the gradient computation
            if args.start_optimize_embedding_index > 0:
                embedding_masker.restore_frozen_params()
            
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            step_time_tracker.tick("optimizer.step")
            
            # Logging
            if global_step % args.logging_per_step == 0:
                period_start_time = log_training_step(
                    global_step, metrics, args, lr_scheduler, grad_norm,
                    period_start_time, training_start_time, mfu_stats, 
                    step_time_tracker, iteration_time_tracker,
                    epoch_idx, tb_logger, chunked_loss_computer
                )
                metrics.reset_period_accumulators()
            
            # Save checkpoint
            # Save at regular intervals (save_checkpoint_per_step) and at early steps (20, 200)
            # Early checkpoints help verify training setup and catch issues early
            should_save = (
                (global_step % args.save_checkpoint_per_step == 0 and global_step > 0) or
                global_step == 20 or  # Early checkpoint for initial verification
                global_step == 200    # Early checkpoint for training stability check
            )
            
            if should_save:
                torch.cuda.empty_cache()
                gc.collect()
                
                with Timer("save checkpoint"):
                    save_model_checkpoint(
                        save_dir=args.output_dir,
                        tag=f"step{global_step}",
                        global_step=global_step,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        dataloader=dataloader,
                        app_state=app_state.set_call_back(converter.revert),
                        dist_checkpointer=dist_checkpointer
                    )
                step_time_tracker.tick(f"save_ckpt*{args.save_checkpoint_per_step}")
            
            iteration_time_tracker.tick("iteration_time_tracker")
            if torch_profiler:
                torch_profiler.step()
    
    # Save final checkpoint
    save_model_checkpoint(
        save_dir=args.output_dir,
        tag=f"step{global_step}",
        global_step=global_step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
        app_state=app_state.set_call_back(converter.revert),
        dist_checkpointer=dist_checkpointer
    )


if __name__ == "__main__":
    train()
