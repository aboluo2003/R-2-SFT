"""Qwen3 Training Script

Multi-node, multi-GPU training script for Qwen3 models using FSDP (Fully Sharded Data Parallel).
Supports distributed training, checkpointing, and comprehensive monitoring.
"""

from __future__ import annotations

import os
import sys

sys.path.append("./onerec_llm/models")

import argparse
import collections
import contextlib
import csv
import datetime
import gc
import itertools
import json
import logging
import queue
import re
import threading
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer

from onerec_llm.data.dataloaders import get_dataloader
from onerec_llm.losses import get_loss_components
from onerec_llm.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from onerec_llm.training.activations import set_activation_checkpointing
from onerec_llm.training.checkpoint import (
    AppState,
    DistributedCheckpointer,
    load_hf_checkpoint,
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

# Process group timeout (24 hours)
PROCESS_GROUP_TIMEOUT = datetime.timedelta(minutes=60 * 24)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Manages training metrics accumulation and statistics.
    
    This class tracks metrics in two ways:
    - Period metrics (period_*): Accumulated over a logging period (logging_per_step steps)
    - Total metrics (total_*): Accumulated over the entire training run
    """
    
    def __init__(self):
        self.reset_period_accumulators()
        # Total metrics accumulated over entire training
        self.total_num_tokens = 0
        self.total_num_samples = 0
        self.total_num_valid_tokens = 0
        self.total_data_source_tokens = collections.defaultdict(int)
        self.local_period_data_source_samples = collections.defaultdict(int)
    
    def reset_period_accumulators(self):
        """Reset accumulated metrics for the current logging period."""
        # Period metrics: accumulated over logging_per_step steps
        self.period_sum_loss = 0.0
        self.period_sum_think_aux_loss = 0.0
        self.period_sum_target_aux_loss = 0.0
        self.period_sum_think_aux_jobs = 0.0
        self.period_sum_target_aux_jobs = 0.0
        self.period_sum_itemic_token_loss = 0.0
        self.period_sum_text_token_loss = 0.0
        self.period_num_tokens = 0
        self.period_num_samples = 0
        self.period_num_valid_tokens = 0
        self.period_data_source_loss = collections.defaultdict(float)
        self.period_data_source_tokens = collections.defaultdict(int)
        self.period_valid_data_source_tokens = collections.defaultdict(int)
        # Track number of steps in current period for averaging
        self.period_num_steps = 0
    
    def update(self, num_tokens, num_samples, num_valid_tokens):
        """Update both period and total metrics."""
        # Update period metrics (for current logging period)
        self.period_num_tokens += num_tokens
        self.period_num_samples += num_samples
        self.period_num_valid_tokens += num_valid_tokens
        
        # Update total metrics (for entire training)
        self.total_num_tokens += num_tokens
        self.total_num_samples += num_samples
        self.total_num_valid_tokens += num_valid_tokens


class TensorBoardLogger:
    """Manages TensorBoard logging in a separate thread."""
    
    def __init__(self, tb_writer: Optional[SummaryWriter]):
        self.tb_writer = tb_writer
        self.metrics_queue = queue.Queue(maxsize=8)
        self.thread = None
        
        if tb_writer is not None and dist.get_rank() == 0:
            self.thread = threading.Thread(
                target=self._write_async,
                args=(tb_writer, self.metrics_queue),
                daemon=True
            )
            self.thread.start()
    
    def _write_async(self, tb_writer, metrics_queue):
        """Async TensorBoard writer thread."""
        while True:
            global_step, log_dict, ticker_stats, ds_loss, ds_tokens, ds_samples = metrics_queue.get()
            total_num_samples = log_dict["perf/total_num_samples"]
            total_num_valid_tokens = log_dict["perf/valid_total_num_tokens"]
            
            # Log main metrics
            for name, data in log_dict.items():
                if data is not None and tb_writer:
                    tb_writer.add_scalar(
                        name, data, global_step=global_step, new_style=True
                    )
                    
                    # Log training metrics by valid tokens
                    if name.startswith("training/"):
                        tb_writer.add_scalar(
                            f"x_token_{name}",
                            data,
                            global_step=total_num_valid_tokens,
                            new_style=True
                        )
            
            # Log ticker stats
            for name, data in ticker_stats.items():
                tb_writer.add_scalar(
                    f"ticker/{name}", data, global_step=global_step, new_style=True
                )
            
            # Log data source metrics
            if ds_loss and tb_writer:
                for key, loss_sum in ds_loss.items():
                    tb_writer.add_scalar(
                        f"data_source_loss/{key}",
                        loss_sum / (ds_tokens.get(key, 0) + 1e-6),
                        global_step=global_step,
                        new_style=True
                    )
            
            if ds_samples and tb_writer:
                for key, samples in ds_samples.items():
                    tb_writer.add_scalar(
                        f"data_source_sample_ratio/{key}",
                        1.0 * samples / total_num_samples,
                        global_step=global_step,
                        new_style=True
                    )
                
                total_tokens = sum(ds_tokens.values())
                if total_tokens > 0:
                    for key, num_tokens in ds_tokens.items():
                        tb_writer.add_scalar(
                            f"data_source_token_ratio/{key}",
                            1.0 * num_tokens / total_tokens,
                            global_step=global_step,
                            new_style=True
                        )
    
    def log(self, global_step, log_dict, ticker_stats, ds_loss, ds_tokens, ds_samples):
        """Queue metrics for async logging."""
        if self.tb_writer is not None:
            self.metrics_queue.put((
                global_step, log_dict, ticker_stats, ds_loss, ds_tokens, ds_samples
            ))


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
    parser.add_argument("--enable_gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--allow_random_init_params", type=str, default='',
                       help="Allow random initialization for specified parameters")
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
    parser.add_argument("--loss_impl", type=str, default="ce", choices=["ce", "ce_v2"],
                       help="Loss implementation to use: ce keeps the original behavior; ce_v2 enables onerec_llm.losses.ce_v2")
    parser.add_argument("--main_loss_beta", type=float, default=1.0,
                       help="Beta for the main language-modeling CE loss; defaults to 1.0 for backward compatibility")
    parser.add_argument("--enable_think_loss", type=int, default=0,
                       help="Enable think-vs-no-think margin auxiliary loss")
    parser.add_argument("--think_loss_beta", type=float, default=1.0,
                       help="Beta for the think-vs-no-think margin auxiliary loss")
    parser.add_argument("--think_loss_sources_name", type=str, default="",
                       help="Comma-separated data sources that enable think auxiliary loss")
    parser.add_argument("--enable_target_loss", type=int, default=0,
                       help="Enable target-token auxiliary loss")
    parser.add_argument("--target_loss_beta", type=float, default=1.0,
                       help="Beta for the target-token auxiliary loss")
    parser.add_argument("--target_loss_sources_name", type=str, default="",
                       help="Comma-separated data sources that enable target auxiliary loss")
    parser.add_argument("--debug_loss_mask_step", type=int, default=0,
                       help="Export rank-local loss mask CSV at the given step; 0 disables it")
    
    # Profiling arguments
    parser.add_argument("--enable_profiler", action="store_true",
                       help="Enable PyTorch profiler for performance analysis")
    
    return parser


class StateDictConverter:
    """Converter for state dict transformations (identity by default)."""
    
    def convert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert state dict (e.g., for loading)."""
        return state_dict
    
    def revert(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Revert state dict (e.g., for saving)."""
        return state_dict


def _init_profiler(output_dir: str, enable: bool = False) -> Optional[torch.profiler.profile]:
    """Initialize PyTorch profiler.
    
    Args:
        output_dir: Directory to save profiler traces
        enable: Whether to enable the profiler. If False, returns None.
    
    Returns:
        PyTorch profiler instance if enabled, None otherwise.
    """
    if not enable:
        return None
    
    if not os.path.exists(output_dir):
        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
    
    def trace_handler(prof):
        prof.export_chrome_trace(
            os.path.join(output_dir, f"{prof.step_num}_w{dist.get_rank()}.json")
        )
    
    # Profiler schedule: wait 50 steps, warmup 1 step, profile 10 steps, repeat once
    # This avoids profiling initialization overhead and captures representative performance
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=50, warmup=1, active=10, repeat=1),
        on_trace_ready=trace_handler,
    )


def _sanitize_debug_name(text: str, max_length: int = 80) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._")
    if not sanitized:
        sanitized = "unknown"
    return sanitized[:max_length]


def _format_debug_token_text(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def export_loss_mask_debug_csv(
    batch: Dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    output_dir: str,
    step: int,
) -> List[str]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    debug_dir = os.path.join(output_dir, "debug_samples")
    os.makedirs(debug_dir, exist_ok=True)

    input_ids = batch["input_ids"][0].cpu().tolist()
    loss_mask = batch["loss_mask"][0].cpu().tolist()
    sample_idx_tensor = batch.get("sample_idx")
    sample_idx_list = None
    if isinstance(sample_idx_tensor, torch.Tensor):
        sample_idx_list = sample_idx_tensor[0].cpu().tolist()

    data_sources = batch.get("data_source", [])
    selected_samples = []
    seen_sources = set()

    if sample_idx_list is not None:
        for sample_id in sample_idx_list:
            sample_id = int(sample_id)
            if sample_id < 0:
                continue
            source_name = data_sources[sample_id] if sample_id < len(data_sources) else f"sample{sample_id}"
            if source_name in seen_sources:
                continue
            seen_sources.add(source_name)
            selected_samples.append((sample_id, str(source_name)))
    else:
        source_name = data_sources[0] if data_sources else "sample0"
        selected_samples.append((0, str(source_name)))

    exported_paths = []
    for sample_id, source_name in selected_samples:
        output_path = os.path.join(
            debug_dir,
            f"rank{rank:04d}_step{step}_sample{sample_id:03d}_{_sanitize_debug_name(source_name)}.csv",
        )
        with open(output_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["input_id", "token_text", "loss_mask"])
            if sample_idx_list is None:
                row_indices = range(len(input_ids))
            else:
                row_indices = (
                    idx for idx, current_sample_id in enumerate(sample_idx_list)
                    if int(current_sample_id) == sample_id
                )

            for idx in row_indices:
                token_id = int(input_ids[idx])
                writer.writerow([
                    token_id,
                    _format_debug_token_text(
                        tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    ),
                    int(loss_mask[idx]),
                ])

        exported_paths.append(output_path)

    logger.info(
        "Exported loss mask debug csvs: step=%s rank=%s files=%s",
        step,
        rank,
        len(exported_paths),
    )
    return exported_paths


def save_model_checkpoint(
    save_dir: str,
    tag: str,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    dataloader: Optional[object],
    app_state: AppState,
    dist_checkpointer: DistributedCheckpointer,
) -> None:
    """Save FSDP+TP model checkpoint.
    
    Args:
        save_dir: Save directory
        tag: Checkpoint tag
        global_step: Global training step
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
        dataloader: Optional dataloader for state saving
        app_state: Application state
        dist_checkpointer: Distributed checkpointer
    """
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    ckpt_path = os.path.join(save_dir, tag)
    if dist.get_rank() == 0:
        os.makedirs(ckpt_path, exist_ok=True)
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(tag)
    
    try:
        # Save model checkpoint
        dist_checkpointer.save_checkpoint(
            state_dict={"app": app_state},
            output_dir=ckpt_path,
            tag=str(global_step)
        )
        
        # Save dataloader state
        if dataloader is not None:
            try:
                dataloader_state = {"dataloader_state_dict": dataloader.state_dict()}
                dataloader_path = os.path.join(ckpt_path, "dataloader_ckpt")
                if dist.get_rank() == 0:
                    os.makedirs(dataloader_path, exist_ok=True)
                dist.barrier()
                
                torch.save(
                    dataloader_state,
                    os.path.join(dataloader_path, f"rank{dist.get_rank()}.pt")
                )
                print_rank_0(f"Saved dataloader state to {dataloader_path}")
            except Exception as e:
                logger.error(f"Failed to save dataloader state: {e}", exc_info=True)
        
        # Save optimizer and scheduler state
        optimizer_path = os.path.join(ckpt_path, "optimizer_ckpt")
        optimizer_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
        }
        if dist.get_rank() == 0:
            os.makedirs(optimizer_path, exist_ok=True)
        dist.barrier()
        torch.save(
            optimizer_state,
            os.path.join(optimizer_path, f"rank{dist.get_rank()}.pt")
        )
        print_rank_0(f"Saved optimizer state to {optimizer_path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
        raise
    finally:
        dist.barrier()


def initialize_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training environment.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        timeout=PROCESS_GROUP_TIMEOUT
    )
    
    return rank, world_size, local_rank


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
        config._attn_implementation = "flash_attention_2"
        config.use_cache = False
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
        load_from_full_model_state_dict(
            model=model,
            full_sd=state_dict,
            allow_random_init_params=args.allow_random_init_params,
            use_tie_weights=args.use_tie_weights
        )
    
    # Tie weights if requested
    # Sharing weights between embedding and output projection can reduce parameters
    # and improve training stability for some models
    if args.use_tie_weights:
        model.lm_head.weight = model.model.embed_tokens.weight
        # Verify weight tying: check if there are any differences (should be ~0)
        diff_weight = model.lm_head.weight - model.model.embed_tokens.weight
        diff_weight_cnt = (diff_weight.full_tensor().abs() > 1e-6).float().sum()
        print_rank_0(
            f"diff_weight_cnt: {diff_weight_cnt.item()}, "
            f"diff_weight_ratio: {diff_weight_cnt.item() / model.lm_head.weight.numel():.4f}"
        )
    
    # Initialize RoPE
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
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True  # Only embeddings and output head are trainable
            else:
                param.requires_grad = False  # Freeze all transformer layers
    
    # Print trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"Trainable parameter: {name}")
    print_rank_0("=" * 50)
    
    return model


def load_model_checkpoint(
    args,
    app_state: AppState,
    dist_checkpointer: DistributedCheckpointer,
    converter: StateDictConverter,
) -> None:
    """Load model checkpoint from distributed checkpoint.
    
    Args:
        args: Training arguments
        app_state: Application state
        dist_checkpointer: Distributed checkpointer
        converter: State dict converter
    """
    ckpt_path = os.path.join(args.resume_from, args.resume_from_tag)
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")
    
    state_dict = {"app": app_state.set_call_back(converter.convert)}
    dist_checkpointer.load_checkpoint(
        state_dict=state_dict,
        checkpoint_dir=args.resume_from,
        tag=args.resume_from_tag
    )
    print_rank_0("Successfully loaded model using distributed checkpoint")


def load_optimizer_checkpoint(
    args,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
) -> None:
    """Load optimizer and scheduler state from checkpoint.
    
    Args:
        args: Training arguments
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
    """
    optimizer_state_dict_path = os.path.join(
        args.resume_from, "optimizer_ckpt", f"rank{dist.get_rank()}.pt"
    )
    if os.path.exists(optimizer_state_dict_path):
        optimizer_state_dict = torch.load(optimizer_state_dict_path)
        lr_scheduler.load_state_dict(optimizer_state_dict["scheduler_state_dict"])
        optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
        print_rank_0(f"Successfully loaded optimizer and scheduler state from {optimizer_state_dict_path}")
    else:
        print_rank_0(f"Warning: Optimizer checkpoint {optimizer_state_dict_path} does not exist")


def load_dataloader_checkpoint(args) -> Optional[Dict]:
    """Load dataloader state from checkpoint.
    
    Args:
        args: Training arguments
    
    Returns:
        Dataloader state dict if found, None otherwise
    """
    dataloader_resume_path = os.path.join(
        args.resume_from, "dataloader_ckpt", f"rank{dist.get_rank()}.pt"
    )
    if os.path.exists(dataloader_resume_path):
        try:
            dataloader_state_dict = torch.load(dataloader_resume_path)["dataloader_state_dict"]
            print_rank_0(f"Successfully loaded dataloader state from {dataloader_resume_path}")
            return dataloader_state_dict
        except Exception as e:
            print_rank_0(f"Error loading dataloader checkpoint: {e}")
            return None
    else:
        print_rank_0(f"Warning: Dataloader checkpoint {dataloader_resume_path} does not exist")
        print_rank_0("Will start training without resuming dataloader state")
        return None


def load_checkpoint(
    args,
    app_state: AppState,
    dist_checkpointer: DistributedCheckpointer,
    converter: StateDictConverter,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
) -> Tuple[Optional[Dict], int]:
    """Load checkpoint if resuming training.
    
    This function orchestrates loading of model, optimizer, scheduler, and dataloader
    checkpoints. It delegates to specialized functions for each component.
    
    Args:
        args: Training arguments
        app_state: Application state
        dist_checkpointer: Distributed checkpointer
        converter: State dict converter
        optimizer: Optimizer instance
        lr_scheduler: Learning rate scheduler
    
    Returns:
        Tuple of (dataloader_state_dict, global_step)
    """
    dataloader_state_dict = None
    global_step = 0
    
    if args.resume_from_tag:
        ckpt_path = os.path.join(args.resume_from, args.resume_from_tag)
        if args.resume_training_state:
            global_step = int(args.resume_from_tag.split("step")[-1])
            print_rank_0(
                f"Resume from checkpoint: {ckpt_path}, global_step={global_step}"
            )
        else:
            print_rank_0(
                f"Resume model weights only from checkpoint: {ckpt_path}, "
                "global_step stays at 0"
            )
        
        # Load model checkpoint
        load_model_checkpoint(args, app_state, dist_checkpointer, converter)
        
        # Load optimizer, scheduler, and dataloader state if requested
        # Note: resume_training_state controls whether to restore the full training state
        # including optimizer momentum, scheduler step, and dataloader position.
        # This allows seamless continuation of training from a checkpoint.
        if args.resume_training_state:
            load_optimizer_checkpoint(args, optimizer, lr_scheduler)
            dataloader_state_dict = load_dataloader_checkpoint(args)
    
    return dataloader_state_dict, global_step


def compute_forward_backward(
    model: torch.nn.Module,
    batch: Dict,
    compute_loss_fn,
    loss_fn: CrossEntropyLoss,
    args,
    embedding_masker: Optional[EmbeddingGradientMasker],
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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
        Tuple of (loss, per_token_loss, aux_metrics)
    """
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch.get("attention_mask", None)
    cu_seqlens = batch.get("cu_seqlens", None)
    position_ids = batch.get("position_ids", None)
    
    # Prepare labels
    # Zero out padding tokens (input_ids <= 0) to avoid computing loss on them
    input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)

    def _get_selected_positions_and_targets(
        local_input_ids: torch.Tensor,
        local_target_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if local_target_mask is None:
            return None, None, 0

        shifted_mask = local_target_mask[:, :-1].bool()
        selected_count = int(shifted_mask.sum().item())
        if selected_count == 0:
            return None, None, 0

        if shifted_mask.shape[0] != 1:
            raise ValueError(
                f"Auxiliary loss expects batch_size=1, got {shifted_mask.shape[0]}"
            )

        selected_positions = shifted_mask[0].nonzero(as_tuple=False).flatten()
        selected_targets = local_input_ids[:, 1:][shifted_mask]
        return selected_positions, selected_targets, selected_count

    def _compute_selected_token_log_probs_from_logits(
        selected_logits: torch.Tensor,
        selected_targets: torch.Tensor,
    ) -> torch.Tensor:
        if selected_logits.ndim == 3:
            selected_logits = selected_logits.squeeze(0)
        selected_log_probs = F.log_softmax(selected_logits.float(), dim=-1)
        return selected_log_probs.gather(
            dim=-1,
            index=selected_targets.unsqueeze(-1),
        ).squeeze(-1)

    def _compute_selected_token_nll_mean_from_logits(
        selected_logits: torch.Tensor,
        selected_targets: torch.Tensor,
    ) -> torch.Tensor:
        return -_compute_selected_token_log_probs_from_logits(
            selected_logits,
            selected_targets,
        ).mean()

    def _zero_metric() -> torch.Tensor:
        return torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)

    def _forward_aux_logits(
        local_input_ids: torch.Tensor,
        local_position_ids: Optional[torch.Tensor],
        local_attention_mask: Optional[torch.Tensor],
        local_cu_seqlens: Optional[torch.Tensor],
        logits_to_keep,
        requires_grad: bool,
    ) -> torch.Tensor:
        prev_chunked = getattr(model, "chunked_loss_computer", False)
        if prev_chunked:
            model.chunked_loss_computer = False

        try:
            context = contextlib.nullcontext() if requires_grad else torch.no_grad()
            with context:
                aux_output = model(
                    input_ids=local_input_ids,
                    attention_mask=local_attention_mask,
                    labels=None,
                    cu_seqlens=local_cu_seqlens,
                    position_ids=local_position_ids,
                    logits_to_keep=logits_to_keep,
                )
            return aux_output.logits
        finally:
            if prev_chunked:
                model.chunked_loss_computer = prev_chunked

    def _build_padded_batch(
        sample_list: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(int(sample["input_ids"].shape[1]) for sample in sample_list)
        batch_size = len(sample_list)
        padded_input_ids = input_ids.new_zeros((batch_size, max_len))
        padded_attention_mask = input_ids.new_zeros((batch_size, max_len))
        padded_position_ids = input_ids.new_zeros((batch_size, max_len))

        for idx, sample in enumerate(sample_list):
            seq_len = int(sample["input_ids"].shape[1])
            padded_input_ids[idx, :seq_len] = sample["input_ids"][0]
            padded_attention_mask[idx, :seq_len] = 1
            padded_position_ids[idx, :seq_len] = sample["position_ids"][0]

        return padded_input_ids, padded_attention_mask, padded_position_ids

    def _gather_selected_logits_from_union(
        logits_union: torch.Tensor,
        union_positions: List[int],
        target_positions: List[int],
        batch_index: int = 0,
    ) -> torch.Tensor:
        position_to_union_index = {
            position: idx for idx, position in enumerate(union_positions)
        }
        gather_indices = [position_to_union_index[position] for position in target_positions]
        return logits_union[batch_index, gather_indices, :]

    def _precompute_think_alt_log_probs() -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, bool]:
        """Precompute alt path log_probs for think (gap) aux loss.

        Does a single no-grad forward through the alt path (empty-think responses)
        and returns alt_log_probs indexed in the same order as think_aux_main_mask's
        nonzero positions. This allows the chunked loss computer to compute the
        gap margin inline without a separate main-path aux forward.

        Returns:
            alt_log_probs: [num_main_think_positions] detached, or None if no jobs
            main_targets: [num_main_think_positions] target token ids, or None
            local_job_count: number of aux jobs on this rank
            has_global_jobs: whether any rank has jobs (all-reduced)
        """
        main_mask = batch.get("think_aux_main_mask", None)
        aux_samples = batch.get("think_aux_samples", None) or []
        if not args.enable_think_loss or args.think_loss_beta == 0 or main_mask is None:
            return None, None, 0, False

        sample_idx_tensor = batch["sample_idx"]
        alt_jobs = []

        for aux_sample in aux_samples:
            sample_idx = int(aux_sample["sample_idx"])
            sample_main_mask = (sample_idx_tensor == sample_idx) & (main_mask > 0)
            main_positions, main_targets, main_count = _get_selected_positions_and_targets(
                input_ids,
                sample_main_mask,
            )
            if main_positions is None or main_targets is None or main_count == 0:
                continue

            aux_input_ids = aux_sample["input_ids"].to(
                device=input_ids.device, non_blocking=True,
            )
            aux_position_ids = aux_sample["position_ids"].to(
                device=input_ids.device, non_blocking=True,
            )
            aux_target_mask = aux_sample["target_mask"].to(
                device=input_ids.device, non_blocking=True,
            )

            aux_positions, aux_targets, aux_count = _get_selected_positions_and_targets(
                aux_input_ids,
                aux_target_mask,
            )
            if aux_positions is None or aux_targets is None or aux_count != main_count:
                continue
            alt_jobs.append({
                "main_positions": main_positions.tolist(),
                "main_targets": main_targets,
                "alt_input_ids": aux_input_ids,
                "alt_position_ids": aux_position_ids,
                "alt_positions": aux_positions.tolist(),
                "alt_targets": aux_targets,
            })

        local_aux_job_count = len(alt_jobs)
        global_aux_job_flag = torch.tensor(
            1 if local_aux_job_count > 0 else 0,
            device=input_ids.device,
            dtype=torch.int32,
        )
        dist.all_reduce(global_aux_job_flag, op=dist.ReduceOp.MAX)
        if global_aux_job_flag.item() == 0:
            return None, None, 0, False

        if local_aux_job_count > 0:
            alt_batch_samples = [
                {
                    "input_ids": job["alt_input_ids"],
                    "position_ids": job["alt_position_ids"],
                }
                for job in alt_jobs
            ]
            alt_union_positions = sorted({
                position
                for job in alt_jobs
                for position in job["alt_positions"]
            })
            alt_batch_input_ids, alt_batch_attention_mask, alt_batch_position_ids = _build_padded_batch(
                alt_batch_samples
            )
        else:
            alt_union_positions = [0]
            alt_batch_input_ids = input_ids[:, :1]
            alt_batch_attention_mask = torch.ones_like(alt_batch_input_ids)
            alt_batch_position_ids = (
                position_ids[:, :1]
                if position_ids is not None else
                torch.zeros_like(alt_batch_input_ids)
            )

        # Single no-grad forward through the alt path
        alt_logits_union = _forward_aux_logits(
            local_input_ids=alt_batch_input_ids,
            local_position_ids=alt_batch_position_ids,
            local_attention_mask=alt_batch_attention_mask,
            local_cu_seqlens=None,
            logits_to_keep=torch.tensor(
                alt_union_positions,
                device=input_ids.device,
                dtype=torch.long,
            ),
            requires_grad=False,
        )

        if local_aux_job_count == 0:
            return None, None, 0, True

        # Gather alt_log_probs for each job, in main_positions order
        all_alt_log_probs = []
        all_main_targets = []
        for batch_index, aux_job in enumerate(alt_jobs):
            alt_selected_logits = _gather_selected_logits_from_union(
                logits_union=alt_logits_union,
                union_positions=alt_union_positions,
                target_positions=aux_job["alt_positions"],
                batch_index=batch_index,
            )
            alt_log_probs = _compute_selected_token_log_probs_from_logits(
                alt_selected_logits,
                aux_job["alt_targets"],
            ).detach()
            all_alt_log_probs.append(alt_log_probs)
            all_main_targets.append(aux_job["main_targets"])

        alt_log_probs_flat = torch.cat(all_alt_log_probs)  # [num_main_think_positions]
        main_targets_flat = torch.cat(all_main_targets)    # [num_main_think_positions]

        return alt_log_probs_flat, main_targets_flat, local_aux_job_count, True

    def _build_think_aux_loss() -> Tuple[torch.Tensor, int, bool]:
        main_mask = batch.get("think_aux_main_mask", None)
        aux_samples = batch.get("think_aux_samples", None) or []
        if not args.enable_think_loss or args.think_loss_beta == 0 or main_mask is None:
            return _zero_metric(), 0, False

        sample_idx_tensor = batch["sample_idx"]
        aux_jobs = []

        for aux_sample in aux_samples:
            sample_idx = int(aux_sample["sample_idx"])
            sample_main_mask = (sample_idx_tensor == sample_idx) & (main_mask > 0)
            main_positions, main_targets, main_count = _get_selected_positions_and_targets(
                input_ids,
                sample_main_mask,
            )
            if main_positions is None or main_targets is None or main_count == 0:
                continue

            aux_input_ids = aux_sample["input_ids"].to(
                device=input_ids.device,
                non_blocking=True,
            )
            aux_position_ids = aux_sample["position_ids"].to(
                device=input_ids.device,
                non_blocking=True,
            )
            aux_target_mask = aux_sample["target_mask"].to(
                device=input_ids.device,
                non_blocking=True,
            )

            aux_positions, aux_targets, aux_count = _get_selected_positions_and_targets(
                aux_input_ids,
                aux_target_mask,
            )
            if aux_positions is None or aux_targets is None or aux_count != main_count:
                continue
            aux_jobs.append({
                "main_positions": main_positions.tolist(),
                "main_targets": main_targets,
                "alt_input_ids": aux_input_ids,
                "alt_position_ids": aux_position_ids,
                "alt_positions": aux_positions.tolist(),
                "alt_targets": aux_targets,
            })

        local_aux_job_count = len(aux_jobs)
        global_aux_job_flag = torch.tensor(
            1 if local_aux_job_count > 0 else 0,
            device=input_ids.device,
            dtype=torch.int32,
        )
        dist.all_reduce(global_aux_job_flag, op=dist.ReduceOp.MAX)
        if global_aux_job_flag.item() == 0:
            return _zero_metric(), 0, False

        if local_aux_job_count > 0:
            alt_batch_samples = [
                {
                    "input_ids": job["alt_input_ids"],
                    "position_ids": job["alt_position_ids"],
                }
                for job in aux_jobs
            ]
            alt_union_positions = sorted({
                position
                for job in aux_jobs
                for position in job["alt_positions"]
            })
            main_union_positions = sorted({
                position
                for job in aux_jobs
                for position in job["main_positions"]
            })
            alt_batch_input_ids, alt_batch_attention_mask, alt_batch_position_ids = _build_padded_batch(
                alt_batch_samples
            )
        else:
            alt_union_positions = [0]
            main_union_positions = [0]
            alt_batch_input_ids = input_ids[:, :1]
            alt_batch_attention_mask = torch.ones_like(alt_batch_input_ids)
            alt_batch_position_ids = (
                position_ids[:, :1]
                if position_ids is not None else
                torch.zeros_like(alt_batch_input_ids)
            )

        alt_logits_union = _forward_aux_logits(
            local_input_ids=alt_batch_input_ids,
            local_position_ids=alt_batch_position_ids,
            local_attention_mask=alt_batch_attention_mask,
            local_cu_seqlens=None,
            logits_to_keep=torch.tensor(
                alt_union_positions,
                device=input_ids.device,
                dtype=torch.long,
            ),
            requires_grad=False,
        )
        main_logits_union = _forward_aux_logits(
            local_input_ids=input_ids,
            local_position_ids=position_ids,
            local_attention_mask=attention_mask,
            local_cu_seqlens=cu_seqlens,
            logits_to_keep=torch.tensor(
                main_union_positions,
                device=input_ids.device,
                dtype=torch.long,
            ),
            requires_grad=True,
        )

        if local_aux_job_count == 0:
            return main_logits_union.sum() * 0.0, 0, True

        aux_loss_terms = []
        for batch_index, aux_job in enumerate(aux_jobs):
            alt_selected_logits = _gather_selected_logits_from_union(
                logits_union=alt_logits_union,
                union_positions=alt_union_positions,
                target_positions=aux_job["alt_positions"],
                batch_index=batch_index,
            )
            main_selected_logits = _gather_selected_logits_from_union(
                logits_union=main_logits_union,
                union_positions=main_union_positions,
                target_positions=aux_job["main_positions"],
                batch_index=0,
            )
            alt_log_probs = _compute_selected_token_log_probs_from_logits(
                alt_selected_logits,
                aux_job["alt_targets"],
            ).detach()
            main_log_probs = _compute_selected_token_log_probs_from_logits(
                main_selected_logits,
                aux_job["main_targets"],
            )
            per_level_aux = torch.clamp(
                alt_log_probs - main_log_probs,
                min=0.0,
            )
            aux_loss_terms.append(args.think_loss_beta * per_level_aux.mean())

        return torch.stack(aux_loss_terms).mean(), local_aux_job_count, True

    def _build_target_aux_targets() -> Tuple[Optional[List[int]], Optional[torch.Tensor], int, bool]:
        main_mask = batch.get("target_aux_main_mask", None)
        if not args.enable_target_loss or args.target_loss_beta == 0 or main_mask is None:
            return None, None, 0, False

        main_positions, main_targets, main_count = _get_selected_positions_and_targets(
            input_ids,
            main_mask,
        )
        local_aux_job_count = main_count // 3 if main_count > 0 else 0
        global_aux_job_flag = torch.tensor(
            1 if local_aux_job_count > 0 else 0,
            device=input_ids.device,
            dtype=torch.int32,
        )
        dist.all_reduce(global_aux_job_flag, op=dist.ReduceOp.MAX)
        return (
            main_positions.tolist() if main_positions is not None else None,
            main_targets,
            local_aux_job_count,
            bool(global_aux_job_flag.item()),
        )

    def _compute_target_aux_loss_from_selected_logits(
        selected_logits: torch.Tensor,
        selected_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if selected_targets is None or selected_targets.numel() == 0:
            return selected_logits.sum() * 0.0
        return args.target_loss_beta * _compute_selected_token_nll_mean_from_logits(
            selected_logits,
            selected_targets,
        )

    think_aux_loss = _zero_metric()
    target_aux_loss = _zero_metric()
    local_think_aux_jobs = 0
    local_target_aux_jobs = 0
    has_global_think_aux = False
    has_global_target_aux = False
    aux_loss_specs = None

    if args.use_chunked_loss_computer:
        # ── Inline aux mode: precompute alt_log_probs, build specs ──
        # Aux loss is computed inline in the chunked loss loop, avoiding
        # separate aux forward passes (saves 2 forwards: target + think-main).
        alt_log_probs_flat = None
        think_main_targets_flat = None
        if args.enable_think_loss and args.think_loss_beta != 0:
            with Timer("think_aux_precompute"):
                (
                    alt_log_probs_flat,
                    think_main_targets_flat,
                    local_think_aux_jobs,
                    has_global_think_aux,
                ) = _precompute_think_alt_log_probs()

        target_main_positions_list = None
        target_main_targets_t = None
        if args.enable_target_loss and args.target_loss_beta != 0:
            with Timer("target_aux"):
                (
                    target_main_positions_list,
                    target_main_targets_t,
                    local_target_aux_jobs,
                    has_global_target_aux,
                ) = _build_target_aux_targets()

        aux_loss_specs = []
        if has_global_target_aux and target_main_targets_t is not None:
            target_mask = batch.get("target_aux_main_mask", None)
            if target_mask is not None:
                aux_loss_specs.append({
                    "name": "target",
                    "mask": target_mask,
                    "targets": target_main_targets_t,
                    "beta": args.target_loss_beta,
                })
        if has_global_think_aux and alt_log_probs_flat is not None:
            think_mask = batch.get("think_aux_main_mask", None)
            if think_mask is not None:
                # CRITICAL: alt_log_probs_flat and think_main_targets_flat must
                # be aligned 1:1 with the nonzero positions of think_mask.
                # If any sample was skipped in _precompute_think_alt_log_probs
                # (because aux_count != main_count, e.g. due to _cut_sample
                # truncating the main path but not the alt path), the lengths
                # will mismatch and the chunked loss loop will index wrong
                # targets / alt_log_probs, corrupting the gap loss gradient.
                mask_nonzero = int(think_mask.sum().item())
                n_alt = int(alt_log_probs_flat.shape[0])
                n_tgt = int(think_main_targets_flat.shape[0])
                assert mask_nonzero == n_alt == n_tgt, (
                    f"[think aux] length mismatch: think_mask.sum()={mask_nonzero}, "
                    f"alt_log_probs_flat={n_alt}, main_targets_flat={n_tgt}. "
                    f"Some samples were skipped in _precompute_think_alt_log_probs "
                    f"(aux_count != main_count), but their positions are still "
                    f"nonzero in think_aux_main_mask. This causes the chunked "
                    f"gap-loss to index WRONG targets/alt_log_probs."
                )
                aux_loss_specs.append({
                    "name": "think",
                    "mask": think_mask,
                    "targets": think_main_targets_flat,
                    "alt_log_probs": alt_log_probs_flat,
                    "beta": args.think_loss_beta,
                })
    else:
        # ── Non-chunked mode: existing flow (separate aux computation) ──
        if args.enable_think_loss and args.think_loss_beta != 0:
            with Timer("think_aux"):
                think_aux_loss, local_think_aux_jobs, has_global_think_aux = _build_think_aux_loss()

        target_main_positions: Optional[List[int]] = None
        target_main_targets_nc: Optional[torch.Tensor] = None
        if args.enable_target_loss and args.target_loss_beta != 0:
            with Timer("target_aux"):
                (
                    target_main_positions,
                    target_main_targets_nc,
                    local_target_aux_jobs,
                    has_global_target_aux,
                ) = _build_target_aux_targets()

    # Forward pass
    with Timer("Fwd"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
        )

        logits = output.logits

        # Shift labels for next token prediction
        pad = torch.full(
            (input_ids.shape[0], 1),
            loss_fn.ignore_index,
            dtype=input_ids.dtype
        ).to(device=input_ids.device, non_blocking=True)
        labels = torch.cat([input_ids[:, 1:], pad], dim=-1)
        # Update labels: use input_ids where loss_mask==1, ignore_index where loss_mask==0
        labels = labels * loss_mask + loss_fn.ignore_index * (1 - loss_mask)

        if args.use_chunked_loss_computer:
            # Inline aux: single forward + backward, aux loss computed in chunks
            # forward_and_backward does ALL backward (main CE + aux) in one pass
            loss, per_token_loss, aux_loss_by_spec = compute_loss_fn(
                logits, labels=labels,
                main_loss_beta=args.main_loss_beta,
                aux_loss_specs=aux_loss_specs if aux_loss_specs else None,
            )
            per_token_loss = per_token_loss.to(loss.device)
            # aux_loss_by_spec is a dict {name: mean_aux_loss} (detached)
            # Split into target vs think for reporting
            if aux_loss_by_spec:
                if "target" in aux_loss_by_spec:
                    target_aux_loss = aux_loss_by_spec["target"]
                if "think" in aux_loss_by_spec:
                    think_aux_loss = aux_loss_by_spec["think"]
            # Add aux losses to total loss for reporting (already backward'd, just for logging)
            loss = loss + think_aux_loss + target_aux_loss
        else:
            loss, per_token_loss = compute_loss_fn(logits, labels=labels)
            loss = args.main_loss_beta * loss
            per_token_loss = per_token_loss.to(loss.device)

            # Target aux (non-chunked: index existing logits, no extra forward)
            if has_global_target_aux:
                if target_main_positions:
                    main_logits_union = logits[:, target_main_positions, :]
                    target_aux_loss = _compute_target_aux_loss_from_selected_logits(
                        selected_logits=main_logits_union,
                        selected_targets=target_main_targets_nc,
                    )
                else:
                    target_aux_loss = logits.sum() * 0.0

            loss = loss + think_aux_loss + target_aux_loss

    # Backward pass
    with Timer("bwd"):
        # Chunked mode: backward already done inside forward_and_backward
        # Non-chunked mode: need explicit backward
        if not args.use_chunked_loss_computer:
            loss.backward()

        # Apply gradient mask for embedding layers if needed
        if args.start_optimize_embedding_index > 0 and embedding_masker is not None:
            embedding_masker.apply_gradient_mask(optimizer)

        if args.max_grad_norm and args.max_grad_norm > 0:
            clip_grad_norm(model, args.max_grad_norm)

    aux_metrics = {}
    if args.enable_think_loss:
        aux_metrics.update({
            "think_aux_loss": think_aux_loss.detach(),
            "think_aux_jobs": torch.tensor(
                float(local_think_aux_jobs),
                device=input_ids.device,
                dtype=torch.float32,
            ),
        })
    if args.enable_target_loss:
        aux_metrics.update({
            "target_aux_loss": target_aux_loss.detach(),
            "target_aux_jobs": torch.tensor(
                float(local_target_aux_jobs),
                device=input_ids.device,
                dtype=torch.float32,
            ),
        })
    return loss, per_token_loss, aux_metrics


def compute_metrics(
    batch: Dict,
    loss: torch.Tensor,
    per_token_loss: torch.Tensor,
    aux_metrics: Dict[str, torch.Tensor],
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

    def _reduce_aux_metric_means(names: List[str]) -> Dict[str, float]:
        values = []
        nonfinite_flags = []
        for name in names:
            value = aux_metrics.get(name)
            if value is None:
                value = torch.tensor(0.0, device=loss.device, dtype=torch.float32)
            value = value.detach().to(device=loss.device)
            value = value.reshape(()).clone().to(dtype=torch.float32)

            local_nonfinite = (~torch.isfinite(value)).to(dtype=torch.float32)
            if local_nonfinite.item() != 0:
                value.zero_()
            values.append(value)
            nonfinite_flags.append(local_nonfinite)

        value_tensor = torch.stack(values)
        nonfinite_tensor = torch.stack(nonfinite_flags)
        dist.all_reduce(nonfinite_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        reduced_values = value_tensor / dist.get_world_size()

        reduced = {}
        for idx, name in enumerate(names):
            reduced_value = reduced_values[idx]
            nonfinite_count = int(nonfinite_tensor[idx].item())
            if nonfinite_count > 0 and dist.get_rank() == 0:
                logger.warning(
                    "Aux metric %s had %d non-finite rank-local values; "
                    "sanitized to 0 for TensorBoard aggregation.",
                    name,
                    nonfinite_count,
                )
            if not torch.isfinite(reduced_value):
                if dist.get_rank() == 0:
                    logger.warning("Aux metric %s reduced to non-finite; logging 0.", name)
                reduced[name] = 0.0
            else:
                reduced[name] = float(reduced_value.item())
        return reduced

    def _reduce_aux_metric_mean(name: str) -> float:
        return _reduce_aux_metric_means([name])[name]

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

    aux_metric_names = []
    if args.enable_think_loss:
        aux_metric_names.extend(["think_aux_loss", "think_aux_jobs"])
    if args.enable_target_loss:
        aux_metric_names.extend(["target_aux_loss", "target_aux_jobs"])
    if aux_metric_names:
        reduced_aux_metrics = _reduce_aux_metric_means(aux_metric_names)
        if args.enable_think_loss:
            metrics.period_sum_think_aux_loss += reduced_aux_metrics["think_aux_loss"]
            metrics.period_sum_think_aux_jobs += reduced_aux_metrics["think_aux_jobs"]
        if args.enable_target_loss:
            metrics.period_sum_target_aux_loss += reduced_aux_metrics["target_aux_loss"]
            metrics.period_sum_target_aux_jobs += reduced_aux_metrics["target_aux_jobs"]
    
    # Compute itemic and text token losses
    if itemic_id_mask is not None:
        itemic_id_mask = itemic_id_mask.view(per_token_loss.shape)
        avg_itemic_token_loss = (itemic_id_mask * per_token_loss).sum() / (itemic_id_mask.sum() + 1e-6)
        avg_text_token_loss = ((1 - itemic_id_mask) * per_token_loss).sum() / ((1 - itemic_id_mask).sum() + 1e-6)
        dist.all_reduce(avg_itemic_token_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_text_token_loss, op=dist.ReduceOp.SUM)
        avg_itemic_token_loss = avg_itemic_token_loss.item() / dist.get_world_size()
        avg_text_token_loss = avg_text_token_loss.item() / dist.get_world_size()
    else:
        avg_itemic_token_loss = 0.0
        avg_text_token_loss = avg_loss
    
    metrics.period_sum_itemic_token_loss += avg_itemic_token_loss
    metrics.period_sum_text_token_loss += avg_text_token_loss
    
    # Monitor data source metrics
    if args.monitor_datasource_loss and data_source is not None:
        local_sample_idx = sample_idx.squeeze()
        unique_sample_idx = local_sample_idx.unique()
        # Get local loss mask for valid token counting
        local_loss_mask = loss_mask.squeeze()
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
    avg_think_aux_loss = metrics.period_sum_think_aux_loss / period_num_steps
    avg_target_aux_loss = metrics.period_sum_target_aux_loss / period_num_steps
    avg_think_aux_jobs = metrics.period_sum_think_aux_jobs / period_num_steps
    avg_target_aux_jobs = metrics.period_sum_target_aux_jobs / period_num_steps
    avg_itemic_token_loss = metrics.period_sum_itemic_token_loss / period_num_steps
    avg_text_token_loss = metrics.period_sum_text_token_loss / period_num_steps
    
    # Reduce data source metrics across all ranks
    period_data_source_loss = dist_reduce_dict(metrics.period_data_source_loss)
    period_data_source_tokens = dist_reduce_dict(metrics.period_data_source_tokens)
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
    if args.enable_think_loss:
        log_dict["training/think_aux_loss"] = avg_think_aux_loss
        log_dict["training/think_aux_jobs_per_rank"] = avg_think_aux_jobs
    if args.enable_target_loss:
        log_dict["training/target_aux_loss"] = avg_target_aux_loss
        log_dict["training/target_aux_jobs_per_rank"] = avg_target_aux_jobs
    
    # Get ticker statistics
    ticker_stats = {}
    for t in [step_time_tracker, iteration_time_tracker]:
        ticker_stats.update(t.stat())
    
    # Log to TensorBoard
    tb_logger.log(
        global_step,
        log_dict,
        ticker_stats,
        period_data_source_loss if args.monitor_datasource_loss else {},
        period_data_source_tokens if args.monitor_datasource_cnt else {},
        total_data_source_samples if args.monitor_datasource_cnt else {},
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
    debug_loss_mask_enabled = args.debug_loss_mask_step > 0
    
    # Validate arguments
    assert args.learning_rate > 0.0, "Learning rate must be positive"
    assert args.save_checkpoint_per_step > 0, "save_checkpoint_per_step must be positive"
    
    # Initialize distributed training
    rank, world_size, local_rank = initialize_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
    
    set_random_seed(args.seed)
    
    # Load dataset configuration
    logger.info(f"Loading dataset config from: {args.dataset_config}")
    with open(args.dataset_config, encoding="utf-8") as f:
        dataset_config = json.loads(f.read())
    dataset_config["think_loss_sources_name"] = (
        args.think_loss_sources_name if args.enable_think_loss else ""
    )
    dataset_config["target_loss_sources_name"] = (
        args.target_loss_sources_name if args.enable_target_loss else ""
    )
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
    torch_profiler = _init_profiler(
        output_dir=os.path.join(args.output_dir, "torch_profile"),
        enable=args.enable_profiler
    )
    
    # Initialize loss function
    CrossEntropyLoss, ChunkedLossComputer = get_loss_components(args.loss_impl)
    print_rank_0(f"Using loss_impl={args.loss_impl}")
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
    
    # Initialize training state
    training_start_time = time.time()
    period_start_time = training_start_time
    remaining_debug_samples = 1  # Number of sample batches to print for debugging
    loss_mask_debug_dumped = False
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
    get_next_batch = lambda: next(data_iter)
    
    # Training loop
    while True:
        with contextlib.ExitStack() as ctx:
            if torch_profiler:
                ctx.enter_context(torch_profiler)
            
            step_time_tracker.tick("enter_context(torch_profiler)")
            try:
                batch = get_next_batch()
            except StopIteration:
                break
            step_time_tracker.tick("next_batch")
            current_step = global_step + 1
            if (
                debug_loss_mask_enabled and
                not loss_mask_debug_dumped and
                current_step == args.debug_loss_mask_step
            ):
                export_loss_mask_debug_csv(
                    batch=batch,
                    tokenizer=tokenizer,
                    output_dir=args.output_dir,
                    step=current_step,
                )
                loss_mask_debug_dumped = True
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
            step_time_tracker.tick("to_cuda(batch)")
            
            # Update MFU stats
            token_count = batch["input_ids"].numel()
            num_samples = len(batch.get("cu_seqlens", [0, 1])) - 1
            mfu_stats.set(num_tokens=token_count, num_samples=num_samples)
            
            # Forward and backward pass
            loss, per_token_loss, aux_metrics = compute_forward_backward(
                model, batch, compute_loss_fn, loss_fn, args,
                embedding_masker, optimizer
            )
            
            # Compute metrics
            epoch_idx = batch.get("epoch_idx", torch.tensor([0])).cpu().item()
            avg_loss, avg_itemic_token_loss, avg_text_token_loss, num_tokens, num_samples, num_valid_tokens = compute_metrics(
                batch, loss, per_token_loss, aux_metrics, batch["loss_mask"], loss_fn, args, metrics
            )
            
            step_time_tracker.tick("compute_metrics")
            
            # Optimizer step
            grad_norm = compute_fsdp_zero2_grad_norm(model)
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