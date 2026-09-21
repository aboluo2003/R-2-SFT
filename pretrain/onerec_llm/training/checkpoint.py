from typing import Dict, Any, Union, Optional, Protocol, Callable, Tuple
import re
import os
import gc
import glob
import time
import datetime
import logging
from pathlib import Path
from concurrent.futures import Future

import torch
import torch.distributed as dist
from safetensors import safe_open

from torch.distributed.checkpoint import (
    async_save,
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
)
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from safetensors.torch import load_file
from tqdm import tqdm

from onerec_llm.utils.distributed import get_world_size_and_rank
from onerec_llm.utils.common import print_rank_0, print_rank_n

def load_safetensors(path: Union[Path, str]) -> Dict[str, torch.Tensor]:
    """Load safetensors file and return a dictionary of tensors.
    
    Args:
        path: Path to the safetensors file.
        
    Returns:
        Dictionary mapping tensor names to tensors.
    """
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def safe_torch_load(
    checkpoint_path: Union[Path, str], 
    weights_only: bool = True, 
    mmap: bool = True
) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. 
    Provides separate handling for safetensors files.

    Args:
        checkpoint_path: Path to the checkpoint file.
        weights_only: Whether to load only tensors, primitive types, and dictionaries
            (passthrough to torch.load). Default: True
        mmap: Whether to mmap from disk into CPU memory. Default: True

    Returns:
        State dict from the checkpoint file.

    Raises:
        ValueError: If the checkpoint file is not found or cannot be loaded.
    """
    try:
        checkpoint_path_str = str(checkpoint_path)
        if checkpoint_path_str.endswith(".safetensors"):
            return load_safetensors(checkpoint_path)
        else:
            return torch.load(
                checkpoint_path_str,
                map_location="cpu",
                mmap=mmap,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}") from e

def load_hf_checkpoint(
    model_dir: str, 
    output_keys_file: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """Load HuggingFace format checkpoint from a directory.
    
    Args:
        model_dir: Directory containing checkpoint files (.safetensors or .bin).
        output_keys_file: Optional path to write checkpoint keys for debugging.
            If None, keys are not written. Default: None.
    
    Returns:
        Merged state dictionary containing all checkpoint weights.
    
    Raises:
        ValueError: If checkpoint files are not found or contain non-tensor values.
    """
    merged_state_dict: Dict[str, torch.Tensor] = {}
    
    # Try to find safetensors files first, fall back to .bin files
    ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not ckpt_paths:
        ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
    
    if not ckpt_paths:
        raise ValueError(f"No checkpoint files found in {model_dir}")
    
    for cpt_idx, cpt_path in enumerate(ckpt_paths):
        print_rank_0(f"Loading checkpoint {cpt_idx + 1}/{len(ckpt_paths)}: {cpt_path}")
        state_dict = safe_torch_load(cpt_path)
        
        # Validate that all values are tensors
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {key}={type(value)} instead."
                )
        
        merged_state_dict.update(state_dict)
        
        # Free memory
        del state_dict
        gc.collect()
    
    # Optionally write keys to file for debugging
    if output_keys_file:
        with open(output_keys_file, "w", encoding="utf-8") as f:
            f.write("# Checkpoint file paths:\n")
            for path in ckpt_paths:
                f.write(f"{path}\n")
            f.write("\n# State dict keys:\n")
            for key in merged_state_dict.keys():
                f.write(f"{key}\n")
    
    return merged_state_dict


def load_checkpoint_to_state_dict(checkpoint_path: Union[str, os.PathLike]) -> Dict[str, torch.Tensor]:
    """Load checkpoint file or directory and return state_dict.
    
    Supports multiple checkpoint formats:
    - .pth or .pt files (PyTorch format)
    - .safetensors files (SafeTensors format)
    - Directories containing .safetensors files (HuggingFace format)
    - .distcp format directories (Distributed checkpoint format)
    
    Args:
        checkpoint_path: Path to checkpoint file or directory.
            Can be:
            - .pth, .pt file path
            - .safetensors file path
            - Directory containing .safetensors files
            - .distcp format directory
    
    Returns:
        state_dict: Dictionary containing model weights
    
    Raises:
        FileNotFoundError: If checkpoint path does not exist
        ValueError: If checkpoint format is unsupported or invalid
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Check if path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # If it's a file
    if os.path.isfile(checkpoint_path):
        # Handle .pth files
        if checkpoint_path.endswith(".pth") or checkpoint_path.endswith(".pt"):
            print_rank_0(f"Loading PyTorch checkpoint from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # If state_dict contains nested 'model' or 'app.model' keys, extract them
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            elif "app" in state_dict and "model" in state_dict["app"]:
                state_dict = state_dict["app"]["model"]
            return state_dict
        
        # Handle .safetensors files
        elif checkpoint_path.endswith(".safetensors"):
            print_rank_0(f"Loading SafeTensors checkpoint from {checkpoint_path}...")
            return load_file(checkpoint_path)
        
        else:
            raise ValueError(f"Unsupported file format: {checkpoint_path}")
    
    # If it's a directory
    elif os.path.isdir(checkpoint_path):
        # Check if it's a .distcp format directory
        if any(file.endswith(".distcp") for file in os.listdir(checkpoint_path)) or \
           os.path.exists(os.path.join(checkpoint_path, "checkpoint.json")):
            print_rank_0(f"Loading DCP checkpoint from {checkpoint_path}...")
            # Use PyTorch's FileSystemReader to load DCP format
            sd: STATE_DICT_TYPE = {}
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
            _load_state_dict(
                sd,
                storage_reader=FileSystemReader(checkpoint_path),
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )
            # Extract model weights section
            if "app" in sd and "model" in sd["app"]:
                return sd["app"]["model"]
            return sd
        
        # Check if it's a directory containing .safetensors files
        safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        
        if safetensors_files:
            # Directly merge all .safetensors files
            print_rank_0(f"Loading and merging all SafeTensors files from {checkpoint_path}...")
            state_dict = {}
            for safetensors_file in tqdm(safetensors_files, desc="Loading safetensors"):
                file_path = os.path.join(checkpoint_path, safetensors_file)
                shard_state_dict = load_file(file_path)
                # Update state_dict, merging all file contents
                state_dict.update(shard_state_dict)
            return state_dict
        
        else:
            raise ValueError(f"No supported checkpoint files found in directory: {checkpoint_path}")
    
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
      
class CheckpointerInterface(Protocol):
    """Protocol interface for checkpoint loaders and savers."""
    
    def load_checkpoint(self, **kwargs) -> Dict[str, Any]:
        """Load checkpoint from storage."""
        ...
    
    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None:
        """Save checkpoint to storage."""
        ...

class DistributedCheckpointer(CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in the DistributedCheckpointing format.

    Args:
        process_group: Optional process group to use for distributed saving/loading.
            If None, the default process group will be used.
            For checkpointing, gloo CPU-based backend is needed.
    """
    
    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup] = None
    ) -> None:
        self._checkpoint_future: Optional[Future] = None
        self._checkpoint_dir_prefix = "global_step"
        _, self._rank = get_world_size_and_rank()
        self._process_group: Optional[dist.ProcessGroup] = process_group
    
    def get_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Get the latest checkpoint directory path.
        
        Args:
            checkpoint_dir: Directory containing checkpoint subdirectories.
            
        Returns:
            Path to the latest checkpoint directory, or None if no checkpoints found.
        """
        checkpoint_dir_pattern = re.compile(f"{self._checkpoint_dir_prefix}(\\d+)")
        checkpoint_paths = []
        
        if not os.path.isdir(checkpoint_dir):
            return None
        
        for name in os.listdir(checkpoint_dir):
            if re.match(checkpoint_dir_pattern, name):
                checkpoint_path = os.path.join(checkpoint_dir, name)
                if os.path.isdir(checkpoint_path):
                    checkpoint_paths.append(name)
        
        if checkpoint_paths:
            latest_checkpoint_dir = sorted(
                checkpoint_paths, 
                key=lambda x: int(x.split("_")[-1])
            )[-1]
            return os.path.join(checkpoint_dir, latest_checkpoint_dir)
        return None

    def load_checkpoint(
        self,
        state_dict: STATE_DICT_TYPE,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        tag: Union[str, int] = "latest"
    ) -> Dict[str, Any]:
        """
        Load a Distributed checkpoint.
        
        Args:
            state_dict: State dictionary to load into.
            checkpoint_path: Direct path to checkpoint. If provided, this takes precedence.
            checkpoint_dir: Directory containing checkpoints.
            tag: Checkpoint tag (e.g., "latest" or step number). Default: "latest".
        
        Returns:
            Loaded state dictionary.
        
        Raises:
            ValueError: If no checkpoint path can be determined.
        """
        if not checkpoint_path:
            if not checkpoint_dir:
                raise ValueError("Either checkpoint_path or checkpoint_dir must be provided")
            
            if tag == "latest":
                checkpoint_path = self.get_latest_checkpoint(checkpoint_dir)
                if not checkpoint_path:
                    raise ValueError(f"No checkpoint found in {checkpoint_dir}")
            else:
                checkpoint_path = str(Path(checkpoint_dir) / str(tag))
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        print_rank_0(f"Loading checkpoint from {checkpoint_path}")
        
        dcp.load(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_path),
            process_group=self._process_group,
        )
        
        return state_dict

    def save_checkpoint(
        self,
        state_dict: STATE_DICT_TYPE,
        output_dir: Union[str, Path],
        tag: Optional[Union[str, int]] = None,
        save_async: bool = False
    ) -> None:
        """
        Save a distributed checkpoint to storage.
        
        If ``save_async`` is True, the save happens asynchronously unblocking the GPUs sooner.
        This should only be used for intermediate checkpoints. Final checkpoint must be synchronous
        as the training job cannot terminate until the checkpoint is persisted.

        Args:
            state_dict: Checkpoint state dict to be written out to file.
            output_dir: Directory to save the checkpoint.
            tag: Checkpoint tag. Used to create the checkpoint directory name, generally step number.
            save_async: If True, save the checkpoint asynchronously. Default: False.
        """
        checkpoint_path = Path(output_dir)
        if tag is not None:
            checkpoint_path = checkpoint_path / f"{self._checkpoint_dir_prefix}{tag}"
        
        checkpoint_path_str = str(checkpoint_path)
        print_rank_0(f"Saving checkpoint to {checkpoint_path_str}")
        
        # Wait for previous checkpoint to finish if still in progress
        if self._checkpoint_future and not self._checkpoint_future.done():
            wait_start = time.perf_counter()
            print_rank_n(
                f"Rank {self._rank}: previous checkpoint has not finished. "
                f"Checkpointing frequency is too high. Waiting...",
                rank=self._rank
            )
            self._checkpoint_future.result()
            wait_time = time.perf_counter() - wait_start
            print_rank_n(
                f"Rank {self._rank}: waited {wait_time:.2f} seconds "
                f"for previous checkpoint to finish",
                rank=self._rank
            )
            self._checkpoint_future = None
        
        cp_start = time.perf_counter()
        
        if save_async:
            def callback(f: Future) -> None:
                if f.exception() is None:
                    print_rank_n(
                        f"Rank {self._rank}: Checkpoint saved asynchronously "
                        f"to {checkpoint_path_str} successfully.",
                        rank=self._rank
                    )
                else:
                    print_rank_n(
                        f"Rank {self._rank}: Checkpoint failed to save asynchronously "
                        f"to {checkpoint_path_str} with exception: {f.exception()}",
                        rank=self._rank
                    )
            
            self._checkpoint_future = async_save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(
                    checkpoint_path_str,
                    thread_count=16
                ),
                process_group=self._process_group,
            )
            
            blocked_time = time.perf_counter() - cp_start
            print_rank_n(
                f"Rank {self._rank}: Trainer was blocked for {blocked_time:.2f} seconds "
                "for checkpointing to start...",
                rank=self._rank
            )
            
            self._checkpoint_future.add_done_callback(callback)
        else:
            print_rank_0(f"Saving model checkpoint synchronously to {checkpoint_path_str}")
            save(
                state_dict=state_dict,
                storage_writer=FileSystemWriter(
                    checkpoint_path_str,
                    thread_count=4
                ),
                process_group=self._process_group,
            )
            print_rank_0(
                "The full model checkpoint, including all the weights and "
                "configurations, has been saved successfully by the "
                "DistributedCheckpointer. "
                "You can now use this checkpoint for further training."
            )

class AppState(Stateful):
  """This is a useful wrapper for checkpointing the Application State. 
     Since this object is compliant with the Stateful protocol, DCP will 
     automatically call state_dict/load_stat_dict as needed in the 
     dcp.save/load APIs.

  Note: We take advantage of this wrapper to hande calling distributed 
    state dict methods on the model and optimizer.
  """

  def __init__(self, model, optimizer=None, call_back=None):
    self.model = model
    self.call_back = call_back

  def set_call_back(self, cb):
    self.call_back = cb
    return self

  def state_dict(self):
    # this line automatically manages FSDP FQN's, as well as sets the 
    # default state dict type to FSDP.SHARDED_STATE_DICT
    model_state_dict = \
      get_model_state_dict(self.model)
    if self.call_back is not None:
      model_state_dict = self.call_back(model_state_dict)
    return {
      "model": model_state_dict
    }

  def load_state_dict(self, state_dict):
    # sets our state dicts on the model and optimizer, now that we've loaded
    set_model_state_dict(
      self.model,
      model_state_dict=state_dict["model"],
    )


logger = logging.getLogger(__name__)

PROCESS_GROUP_TIMEOUT = datetime.timedelta(minutes=60 * 24)


def initialize_distributed() -> tuple:
    """Initialize distributed training environment via MPI env vars.

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


def init_profiler(output_dir: str, enable: bool = False) -> Optional[torch.profiler.profile]:
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

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=50, warmup=1, active=10, repeat=1),
        on_trace_ready=trace_handler,
    )


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

    Saves model state (via DistributedCheckpointer), optimizer/scheduler state,
    and optionally dataloader state to the specified directory.

    Args:
        save_dir: Save directory
        tag: Checkpoint tag (e.g. "step1000")
        global_step: Global training step number
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


def load_model_checkpoint(
    args,
    app_state: AppState,
    dist_checkpointer: DistributedCheckpointer,
    converter,
) -> None:
    """Load model checkpoint from distributed checkpoint.

    Args:
        args: Training arguments (must have resume_from and resume_from_tag)
        app_state: Application state
        dist_checkpointer: Distributed checkpointer
        converter: State dict converter
    """
    ckpt_path = args.resume_from
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")


    # DCP saves under {output_dir}/{tag}/global_step{N}/, so we need to
    # point checkpoint_dir to {resume_from}/{tag} and pass the numeric step
    # as tag so DistributedCheckpointer reconstructs the correct sub-path
    # "global_step{N}" via its _checkpoint_dir_prefix.
    # global_step = int(args.resume_from_tag.split("step")[-1])
    state_dict = {"app": app_state.set_call_back(converter.convert)}
    dist_checkpointer.load_checkpoint(
        state_dict=state_dict,
        checkpoint_dir=ckpt_path,
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
        args: Training arguments (must have resume_from)
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
        args: Training arguments (must have resume_from)

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
    converter,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
) -> Tuple[Optional[Dict], int]:
    """Load checkpoint if resuming training.

    Orchestrates loading of model, optimizer, scheduler, and dataloader checkpoints.

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

        load_model_checkpoint(args, app_state, dist_checkpointer, converter)

        if args.resume_training_state:
            load_optimizer_checkpoint(args, optimizer, lr_scheduler)
            dataloader_state_dict = load_dataloader_checkpoint(args)

    return dataloader_state_dict, global_step
