#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0b: 用 vLLM + Qwen3-8B 对 caption 做摘要（压缩到 75 字以下）。

参考：reason/mpi/cross_reco/ablation/0701_xxx/collect_shared_device_yes.py

部署策略：
  - MPI 多机多卡，每 rank 绑定 1 GPU（CUDA_VISIBLE_DEVICES=local_rank）
  - 每 rank 内 vLLM TP=1，文件级 round-robin 分配
  - 批量推理 llm.generate() 离线模式（非 server 模式），效率最高

输入：caption_shards/part-NNNNNN.parquet (pid, dense_caption)
输出：caption_summary/part-NNNNNN.parquet (pid, dense_caption, summary)

特性：
  - enable_thinking=False（Qwen3-8B 支持非思考模式，效率高）
  - 抑制 vLLM "Processing prompts" tqdm 噪音（use_tqdm=False + 日志级别）
  - 进度 + ETA：每 batch 输出已处理行数、速率、预计剩余时间
  - 断点续跑：跳过已完成的分片（输出文件存在且行数匹配）

用法：
  mpirun -np 8 python3 summary.py \
      --input_dir .../caption_shards \
      --output_dir .../caption_summary \
      --model_path .../Qwen3-8B
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 必须在 import vllm 之前设置，抑制其日志噪音
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.distributed as dist

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREFIX = _SCRIPT_DIR.parents[4]  # OpenOneRec/ root
DEFAULT_INPUT_DIR = str(_PREFIX / "data" / "RecIF-cot" / "caption_shards")
DEFAULT_OUTPUT_DIR = str(_PREFIX / "data" / "RecIF-cot" / "caption_summary")
DEFAULT_MODEL_PATH = str(_PREFIX / "models" / "Qwen3-8B")

SUMMARY_SYSTEM_PROMPT = "你是一个视频内容摘要助手，擅长将冗长的视频描述压缩为简洁的中文摘要。"

SUMMARY_USER_TEMPLATE = """请将以下视频描述压缩为不超过75个中文字符的摘要，保留关键信息（主题、内容、受众兴趣点）。
只输出摘要文本，不要输出任何其他内容、解释或前缀。

视频描述：
{caption}"""


# -----------------------------------------------------------------------------
# 分布式初始化（参考 collect_shared_device_yes.py）
# -----------------------------------------------------------------------------
def init_distributed() -> Tuple[int, int, int, Optional[Any], str]:
    """优先复用 MPI 做进程编排；仅在 torchrun 下初始化 torch.distributed。"""
    if MPI is not None and "OMPI_COMM_WORLD_RANK" in os.environ:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        return rank, world_size, local_rank, comm, "mpi"

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank, None, "torch"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
        return rank, world_size, local_rank, None, "torch"
    return rank, world_size, local_rank, None, "single"


def patch_tokenizer_for_vllm() -> None:
    """兼容 vLLM 0.7.x 与 transformers 5.x 的 tokenizer 属性差异。"""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


def chunk_for_rank(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    return [item for idx, item in enumerate(items) if idx % world_size == rank]


def setup_logging(level_name: str, rank: int) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format=f"%(asctime)s [%(levelname)s] [Rank {rank}] %(message)s",
        force=True,
    )
    # 抑制 vllm 的 INFO 日志（"Processing prompts" 等）
    for name in ("vllm", "vllm.engine", "vllm.core", "vllm.transformers_utils"):
        logging.getLogger(name).setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Prompt 构造
# -----------------------------------------------------------------------------
def build_prompts(
    tokenizer: Any,
    captions: Sequence[str],
) -> List[str]:
    """对每个 caption 构造 chat prompt（enable_thinking=False）。"""
    prompts: List[str] = []
    for cap in captions:
        if not cap or not str(cap).strip():
            prompts.append("")
            continue
        user_prompt = SUMMARY_USER_TEMPLATE.format(caption=str(cap))
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 老版本 tokenizer 不支持 enable_thinking 参数
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(text)
    return prompts


def strip_think(text: str) -> str:
    """如果模型仍然输出了 think，去掉它。"""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# -----------------------------------------------------------------------------
# 文件处理
# -----------------------------------------------------------------------------
def is_shard_done(output_path: Path, expected_rows: int) -> bool:
    """断点续跑：检查输出文件是否已完整写入。"""
    if not output_path.exists():
        return False
    try:
        meta = pq.ParquetFile(output_path).metadata
        return meta.num_rows == expected_rows
    except Exception:
        return False


def process_file(
    input_path: str,
    output_path: Path,
    llm: LLM,
    tokenizer: Any,
    sampling_params: SamplingParams,
    batch_size: int,
    rank: int,
    stats: Dict[str, int],
    total_rows: int,
    t_start: float,
) -> int:
    """处理一个分片文件，返回写入的行数。"""
    # 读取输入
    df = pq.read_table(input_path).to_pandas()
    n_rows = len(df)
    stats["rows_seen"] += n_rows

    captions = df["dense_caption"].tolist()
    pids = df["pid"].tolist()

    # 构造 prompts
    prompts = build_prompts(tokenizer, captions)

    # 过滤空 prompt（caption 为空的行）
    valid_indices = [i for i, p in enumerate(prompts) if p]
    stats["rows_empty_caption"] += n_rows - len(valid_indices)

    summaries: List[str] = [""] * n_rows

    # 批量推理
    for batch_start in range(0, len(valid_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]
        batch_prompts = [prompts[i] for i in batch_indices]

        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        for idx, output in zip(batch_indices, outputs):
            text = output.outputs[0].text
            summaries[idx] = strip_think(text)

        stats["rows_done"] += len(batch_indices)
        _log_progress(rank, stats, total_rows, t_start)

    # 写出
    df_out = df.copy()
    df_out["summary"] = summaries
    table = pa.Table.from_pandas(df_out, preserve_index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    stats["files_done"] += 1
    logger.info(
        "[Rank %s] 写出 %s rows=%s",
        rank, output_path.name, n_rows,
    )
    return n_rows


def _log_progress(rank: int, stats: Dict[str, int], total_rows: int, t_start: float) -> None:
    """输出进度 + 速率 + ETA。"""
    done = stats["rows_done"]
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total_rows - done
    eta_sec = remaining / rate if rate > 0 else 0
    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    logger.info(
        "[Rank %s] 进度: %s/%s rows (%.1f%%), rate=%.0f rows/s, ETA=%s, files_done=%s",
        rank, done, total_rows, 100.0 * done / max(1, total_rows),
        rate, eta_str, stats["files_done"],
    )


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM Qwen3-8B caption 摘要（MPI 并行）")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="caption 分片目录（split.py 的输出）")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="摘要输出目录")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Qwen3-8B 模型路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="每进程 vLLM TP size；MPI 多卡样本并行时保持 1")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="vLLM 最大上下文长度，caption 摘要 2K 足够")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="每次 llm.generate 的 prompt 数")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="摘要最大输出 token 数（75 中文字符约 100-150 token）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_files", type=int, default=0,
                        help="调试用：限制总文件数，0=不限")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="调试用：限制总样本数，0=不限")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--overwrite", action="store_true",
                        help="强制覆盖已完成的输出分片")
    args = parser.parse_args()

    rank, world_size, local_rank, comm, mode = init_distributed()
    setup_logging(args.log_level, rank)
    logger.info(
        "启动: rank=%s world_size=%s local_rank=%s mode=%s model=%s",
        rank, world_size, local_rank, mode, args.model_path,
    )

    patch_tokenizer_for_vllm()

    # 列出所有输入分片
    input_dir = Path(args.input_dir)
    all_files = sorted(input_dir.glob("part-*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"未在 {input_dir} 找到分片，请先运行 split.py")
    if args.max_files > 0:
        all_files = all_files[: args.max_files]

    my_files = chunk_for_rank(all_files, rank, world_size)
    logger.info("文件分配: total=%s my_files=%s", len(all_files), len(my_files))

    # 计算本 rank 负责的总行数（用于 ETA）
    total_rows = 0
    file_row_counts: Dict[str, int] = {}
    for f in my_files:
        try:
            n = pq.ParquetFile(str(f)).metadata.num_rows
            file_row_counts[str(f)] = n
            total_rows += n
        except Exception:
            file_row_counts[str(f)] = 0
    logger.info("[Rank %s] 总行数=%s", rank, total_rows)

    # 初始化 vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 处理
    stats = {
        "rows_seen": 0,
        "rows_done": 0,
        "rows_empty_caption": 0,
        "files_done": 0,
    }
    t_start = time.time()
    # MAX_SAMPLES=0 表示不限制；>0 时作为预算，处理够就停
    remaining_budget: Optional[int] = args.max_samples if args.max_samples > 0 else None

    for file_path in my_files:
        if remaining_budget is not None and remaining_budget <= 0:
            break

        expected_rows = file_row_counts[str(file_path)]
        output_path = Path(args.output_dir) / file_path.name

        # 断点续跑
        if not args.overwrite and is_shard_done(output_path, expected_rows):
            stats["rows_done"] += expected_rows
            stats["files_done"] += 1
            logger.info("[Rank %s] 跳过已完成: %s", rank, output_path.name)
            _log_progress(rank, stats, total_rows, t_start)
            continue

        n_written = process_file(
            input_path=str(file_path),
            output_path=output_path,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            batch_size=args.batch_size,
            rank=rank,
            stats=stats,
            total_rows=total_rows,
            t_start=t_start,
        )

        if remaining_budget is not None:
            remaining_budget -= n_written

    elapsed = time.time() - t_start
    logger.info(
        "[Rank %s] 完成: rows_done=%s files_done=%s elapsed=%.1fs",
        rank, stats["rows_done"], stats["files_done"], elapsed,
    )

    # 汇总
    if comm is not None:
        gathered = comm.gather(stats, root=0)
        comm.Barrier()
        if rank == 0:
            total_done = sum(g["rows_done"] for g in gathered)
            total_files = sum(g["files_done"] for g in gathered)
            total_elapsed = time.time() - t_start
            print("\n================ Caption Summary Done ================")
            print(f"  ranks           : {world_size}")
            print(f"  total files     : {total_files}")
            print(f"  total rows      : {total_done}")
            print(f"  elapsed         : {total_elapsed:.1f}s")
            if total_elapsed > 0:
                print(f"  throughput      : {total_done / total_elapsed:.0f} rows/s")
            print(f"  output_dir      : {args.output_dir}")
            print("================================================================")
    else:
        print("\n================ Caption Summary Done ================")
        print(f"  total files     : {stats['files_done']}")
        print(f"  total rows      : {stats['rows_done']}")
        print(f"  elapsed         : {elapsed:.1f}s")
        print(f"  output_dir      : {args.output_dir}")
        print("================================================================")


if __name__ == "__main__":
    main()
