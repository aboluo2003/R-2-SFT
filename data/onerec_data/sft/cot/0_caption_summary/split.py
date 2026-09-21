#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0a: 把 pid2caption.parquet 拆成多个小 parquet 分片。

输入：OpenOneRec/data/RecIF/pid2caption.parquet（12.66M 行，单 row group ~5.6GB）
输出：每 5000 行一个 parquet 分片，输出到 OpenOneRec/data/RecIF-cot/caption_shards/

单进程运行即可（I/O bound，不需要 GPU）：
    python3 split.py --input ... --output_dir ... --rows_per_shard 5000

也可在 MPI 下运行（rank 0 做拆分，其它 rank 等待）：
    mpirun -np 4 python3 split.py --input ... --output_dir ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREFIX = _SCRIPT_DIR.parents[4]  # OpenOneRec/ root
DEFAULT_INPUT = str(_PREFIX / "data" / "RecIF" / "pid2caption.parquet")
DEFAULT_OUTPUT_DIR = str(_PREFIX / "data" / "RecIF-cot" / "caption_shards")
DEFAULT_ROWS_PER_SHARD = 5000


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        force=True,
    )


def split_parquet(input_path: str, output_dir: str, rows_per_shard: int) -> int:
    """读取大 parquet，按 rows_per_shard 写出分片。返回分片数。"""
    os.makedirs(output_dir, exist_ok=True)
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    logger.info("输入: %s (rows=%s, row_groups=%s)", input_path, total_rows, pf.num_row_groups)

    shard_idx = 0
    buf_pid: List[Any] = []
    buf_cap: List[Any] = []
    rows_seen = 0
    t0 = time.time()

    progress = tqdm(
        pf.iter_batches(batch_size=100000, columns=["pid", "dense_caption"]),
        desc="split",
        total=pf.num_row_groups,
    )
    for batch in progress:
        table = batch.to_pandas()
        for pid, cap in zip(table["pid"].tolist(), table["dense_caption"].tolist()):
            buf_pid.append(pid)
            buf_cap.append(cap)
            rows_seen += 1
            if len(buf_pid) >= rows_per_shard:
                _write_shard(output_dir, shard_idx, buf_pid, buf_cap)
                shard_idx += 1
                buf_pid = []
                buf_cap = []
                if shard_idx % 50 == 0:
                    elapsed = time.time() - t0
                    rate = rows_seen / elapsed if elapsed > 0 else 0
                    logger.info(
                        "已写出 %s 分片, rows=%s, rate=%.0f rows/s, elapsed=%.1fs",
                        shard_idx, rows_seen, rate, elapsed,
                    )

    # 写剩余不足一个 shard 的行
    if buf_pid:
        _write_shard(output_dir, shard_idx, buf_pid, buf_cap)
        shard_idx += 1

    progress.close()
    logger.info("拆分完成: %s 分片, rows=%s, 耗时=%.1fs", shard_idx, rows_seen, time.time() - t0)
    return shard_idx


def _write_shard(output_dir: str, shard_idx: int, pids: List[Any], caps: List[Any]) -> None:
    """写一个 parquet 分片。"""
    table = pa.table({
        "pid": pa.array(pids, type=pa.int64()),
        "dense_caption": pa.array(caps, type=pa.string()),
    })
    path = os.path.join(output_dir, f"part-{shard_idx:06d}.parquet")
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="拆分 pid2caption.parquet 为小分片")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="输入 parquet 路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--rows_per_shard", type=int, default=DEFAULT_ROWS_PER_SHARD,
                        help="每个分片行数，默认 5000")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if MPI is not None and "OMPI_COMM_WORLD_RANK" in os.environ:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0

    if rank == 0:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"输入文件不存在: {args.input}")
        n_shards = split_parquet(args.input, args.output_dir, args.rows_per_shard)
        print(f"\n拆分完成: {n_shards} 个分片 -> {args.output_dir}")
        print(f"每分片 {args.rows_per_shard} 行（最后一个可能更少）")

    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":
    main()
