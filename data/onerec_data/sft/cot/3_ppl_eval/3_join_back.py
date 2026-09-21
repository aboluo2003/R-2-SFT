#!/usr/bin/env python3
"""Step 3a join: 把 step3a PPL eval 输出与 step2 CoT 输出按 sample_key join，
恢复 step5 需要的 history_captions / target_pids / target_captions，
丢弃渲染中间列（system_text / input_text / input_with_cot_text / target_text /
render_meta / history_length / history_values / representation_type）。

MPI 策略: 按 hash(sample_key) % world_size 分片。每个 rank:
  1. 读 step2 全量文件，只保留本 rank 的 sample_key 分片 -> 构造 {sample_key: row} 索引
  2. 读 step3a eval 全量文件，只保留本 rank 的 sample_key 分片
  3. 本地 join，写出 part-{rank:05d}_{chunk:05d}.parquet

输入:
  --cot_input  step2 CoT 输出目录 (sample_key, uid, target_domain, cot,
               history_pids, history_captions, target_pids, target_captions, history_meta, ...)
  --ppl_input  step3a PPL eval 输出目录 (ppl_sample_key, sample_key, target_idx,
               target_pid, logp_y_given_x, logp_y_given_x_cot, ppl_y_given_x,
               ppl_y_given_x_cot, num_target_tokens, eval_meta, + preprocess 渲染中间列)
  --output     输出目录

输出 schema (per-target, 与 step3a 同粒度):
  ppl_sample_key, sample_key, uid, target_domain, target_idx, target_pid,
  cot, history_pids, history_captions, target_pids, target_captions, history_meta,
  logp_y_given_x, logp_y_given_x_cot, ppl_y_given_x, ppl_y_given_x_cot,
  num_target_tokens, eval_meta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = logging.getLogger(__name__)

# step2 要读的列 (维度表; 仅取 join 需要的列, 节省内存)
STEP2_COLUMNS: Tuple[str, ...] = (
    "sample_key",
    "uid",
    "target_domain",
    "cot",
    "history_pids",
    "history_captions",
    "target_pids",
    "target_captions",
    "history_meta",
)

# step3a eval 要读的列 (事实表; 跳过渲染中间列)
PPL_COLUMNS: Tuple[str, ...] = (
    "ppl_sample_key",
    "sample_key",
    "uid",
    "target_domain",
    "target_idx",
    "target_pid",
    "logp_y_given_x",
    "logp_y_given_x_cot",
    "ppl_y_given_x",
    "ppl_y_given_x_cot",
    "num_target_tokens",
    "eval_meta",
)

# 最终输出列顺序
OUTPUT_COLUMNS: Tuple[str, ...] = (
    "ppl_sample_key",
    "sample_key",
    "uid",
    "target_domain",
    "target_idx",
    "target_pid",
    "cot",
    "history_pids",
    "history_captions",
    "target_pids",
    "target_captions",
    "history_meta",
    "logp_y_given_x",
    "logp_y_given_x_cot",
    "ppl_y_given_x",
    "ppl_y_given_x_cot",
    "num_target_tokens",
    "eval_meta",
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def hash_shard(sample_key: Any, world_size: int) -> int:
    """稳定的 sample_key -> rank 分片。sha1 保证跨 step2/step3a、跨 run 一致。"""
    if world_size <= 1:
        return 0
    digest = hashlib.sha1(str(sample_key).encode("utf-8")).hexdigest()
    return int(digest, 16) % world_size


def list_parquet_files(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    if os.path.isfile(path):
        return [path]
    out: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name.endswith(".parquet"):
                out.append(os.path.join(root, name))
    return sorted(out)


def _shard_mask(sample_keys: pd.Series, world_size: int, rank: int) -> pd.Series:
    if world_size <= 1:
        return pd.Series([True] * len(sample_keys), index=sample_keys.index)
    return sample_keys.map(lambda x: hash_shard(x, world_size) == rank)


def load_step2_index(
    cot_input: str, rank: int, world_size: int
) -> Dict[str, Dict[str, Any]]:
    """读 step2 全量文件, 按 sample_key hash 分片, 构造 {sample_key: row} 索引。"""
    files = list_parquet_files(cot_input)
    index: Dict[str, Dict[str, Any]] = {}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=list(STEP2_COLUMNS))
        except Exception as exc:
            logger.warning("[rank %d] 读 step2 文件失败 %s: %s", rank, f, exc)
            continue
        if len(df) == 0:
            continue
        if world_size > 1:
            df = df[_shard_mask(df["sample_key"].astype(str), world_size, rank)]
        for row in df.to_dict(orient="records"):
            sk = str(row.get("sample_key"))
            if sk in index:
                continue  # step2 里 sample_key 全局唯一
            index[sk] = row
    logger.info("[rank %d] step2 索引大小 = %d (扫描 %d 个文件)", rank, len(index), len(files))
    return index


def iter_ppl_shard(
    ppl_input: str, rank: int, world_size: int
) -> Iterable[pd.DataFrame]:
    """按 sample_key hash 分片读 step3a eval 输出。"""
    files = list_parquet_files(ppl_input)
    for f in files:
        try:
            df = pd.read_parquet(f, columns=list(PPL_COLUMNS))
        except Exception as exc:
            logger.warning("[rank %d] 读 ppl 文件失败 %s: %s", rank, f, exc)
            continue
        if len(df) == 0:
            continue
        if world_size > 1:
            df = df[_shard_mask(df["sample_key"].astype(str), world_size, rank)]
        if len(df) > 0:
            yield df


def write_shard(
    records: List[Dict[str, Any]], output_dir: str, rank: int, part_idx: int
) -> str:
    df = pd.DataFrame(records, columns=list(OUTPUT_COLUMNS))
    path = os.path.join(output_dir, f"part-{rank:05d}_{part_idx:05d}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def join_and_write(
    cot_input: str,
    ppl_input: str,
    output: str,
    rank: int,
    world_size: int,
    rows_per_shard: int,
) -> Dict[str, Any]:
    os.makedirs(output, exist_ok=True)
    step2_index = load_step2_index(cot_input, rank, world_size)

    buffer: List[Dict[str, Any]] = []
    part_idx = 0
    total_rows = 0
    missing = 0
    files_seen = 0

    for df in iter_ppl_shard(ppl_input, rank, world_size):
        files_seen += 1
        for row in df.to_dict(orient="records"):
            sk = str(row.get("sample_key"))
            step2_row = step2_index.get(sk)
            if step2_row is None:
                missing += 1
                continue
            merged: Dict[str, Any] = {
                "ppl_sample_key": row.get("ppl_sample_key"),
                "sample_key": sk,
                "uid": step2_row.get("uid", row.get("uid")),
                "target_domain": step2_row.get("target_domain", row.get("target_domain")),
                "target_idx": row.get("target_idx"),
                "target_pid": row.get("target_pid"),
                "cot": step2_row.get("cot"),
                "history_pids": step2_row.get("history_pids"),
                "history_captions": step2_row.get("history_captions"),
                "target_pids": step2_row.get("target_pids"),
                "target_captions": step2_row.get("target_captions"),
                "history_meta": step2_row.get("history_meta"),
                "logp_y_given_x": row.get("logp_y_given_x"),
                "logp_y_given_x_cot": row.get("logp_y_given_x_cot"),
                "ppl_y_given_x": row.get("ppl_y_given_x"),
                "ppl_y_given_x_cot": row.get("ppl_y_given_x_cot"),
                "num_target_tokens": row.get("num_target_tokens"),
                "eval_meta": row.get("eval_meta"),
            }
            buffer.append(merged)
            total_rows += 1
            if len(buffer) >= rows_per_shard:
                part_idx += 1
                write_shard(buffer, output, rank, part_idx)
                buffer = []
    if buffer:
        part_idx += 1
        write_shard(buffer, output, rank, part_idx)
        buffer = []

    stats = {
        "rank": rank,
        "files_scanned": files_seen,
        "rows_joined": total_rows,
        "rows_missing_step2": missing,
        "parts_written": part_idx,
    }
    logger.info(
        "[rank %d] join 完成: joined=%d, missing_step2=%d, parts=%d",
        rank, total_rows, missing, part_idx,
    )
    return stats


def aggregate_summary(local_stats: Dict[str, Any], comm: Any, rank: int, world_size: int) -> Dict[str, Any]:
    if comm is None or world_size <= 1:
        return {**local_stats, "world_size": 1}
    all_stats = comm.allgather(local_stats)
    total_joined = sum(s.get("rows_joined", 0) for s in all_stats)
    total_missing = sum(s.get("rows_missing_step2", 0) for s in all_stats)
    total_parts = sum(s.get("parts_written", 0) for s in all_stats)
    return {
        "world_size": world_size,
        "total_rows_joined": total_joined,
        "total_rows_missing_step2": total_missing,
        "total_parts_written": total_parts,
        "per_rank": all_stats,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step 3a join: 把 step3a PPL eval 输出与 step2 CoT 输出 join, 输出干净的 per-target parquet"
    )
    p.add_argument("--cot_input", required=True, help="step2 CoT 输出目录")
    p.add_argument("--ppl_input", required=True, help="step3a PPL eval 输出目录")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--rows_per_shard", type=int, default=2000, help="每个输出 parquet 行数 (默认 2000)")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.verbose)

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        comm = None
        rank = 0
        world_size = 1
        logger.warning("mpi4py 不可用, 退化为单进程模式。")

    logger.info(
        "[rank %d/%d] cot_input=%s ppl_input=%s output=%s",
        rank, world_size, args.cot_input, args.ppl_input, args.output,
    )

    local_stats = join_and_write(
        cot_input=args.cot_input,
        ppl_input=args.ppl_input,
        output=args.output,
        rank=rank,
        world_size=world_size,
        rows_per_shard=args.rows_per_shard,
    )

    summary = aggregate_summary(local_stats, comm, rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("Step 3a join 完成")
        print("=" * 60)
        print(f"world_size              : {summary.get('world_size')}")
        print(f"total_rows_joined       : {summary.get('total_rows_joined', local_stats['rows_joined'])}")
        print(f"total_rows_missing_step2: {summary.get('total_rows_missing_step2', local_stats['rows_missing_step2'])}")
        print(f"total_parts_written     : {summary.get('total_parts_written', local_stats['parts_written'])}")
        print(f"output_dir              : {args.output}")
        print("=" * 60)
        summary_path = os.path.join(args.output, "join_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("summary 写入 %s", summary_path)


if __name__ == "__main__":
    main()
