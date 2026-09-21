#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 (history_text): 为 RecIF CoT 消融数据抽取 video / ad 历史 + caption 文本。

任务：
  Step 1：从 onerec_bench_release.parquet 抽取每个用户的 video / ad 历史序列与
          target 序列，字段对齐现有 SFT 脚本 video_rec.py / ad_rec.py。
  Step 2：用 caption（dense_caption 或 summary）把 history / target 里的 pid
          映射成文本序列。没有 caption 的 item 在 item 粒度丢弃；若丢完后 history
          为空则整条样本丢弃。
          - --caption_column dense_caption：用 pid2caption.parquet 的原始完整描述
          - --caption_column summary：用 caption_summary/ 分片目录里的 Qwen3-8B 摘要

输出 schema (每行 = 一条样本)：
  uid            : int
  target_domain  : str   ('video' | 'ad')
  history_pids   : list[int]   过滤后保留的 history pid（按原顺序）
  history_captions : list[str] 与 history_pids 一一对应
  target_pids    : list[int]   过滤后保留的 target pid
  target_captions: list[str]   与 target_pids 一一对应
  history_meta   : str (JSON)  原始字段名 / 截断信息等，便于后续 CoT prompt 复现

MPI 策略：方案 A
  - caption dict 较大（~7-10GB 内存），但用户数据只有 16 万行。
  - 每个 rank 各自全量读取 pid2caption.parquet 构建本地 dict（同节点 OS page
    cache 会让磁盘只真正读一次）。
  - 用户按 round-robin 分配给各 rank（chunk_for_rank）。
  - 输出按 rank 分片，每 1000 行一个 parquet。

用法：
  python3 main.py \
      --bench_uri  .../onerec_bench_release.parquet \
      --caption_uri .../pid2caption.parquet \
      --output_uri  .../steps/1_history_text

  mpirun -np 16 python3 main.py --bench_uri ... --caption_uri ... --output_uri ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 常量：与现有 SFT 脚本 video_rec.py / ad_rec.py 对齐
# -----------------------------------------------------------------------------
VIDEO_HIST_MAX_LEN = 512          # video_rec.py: HIST_MAX_LEN
AD_VIDEO_HIST_MAX_LEN = 100      # ad_rec.py: VIDEO_HIST_MAX_LEN
AD_AD_HIST_MAX_LEN = 200          # ad_rec.py: AD_HIST_MAX_LEN
TARGET_MAX_LEN = 10                # video_rec.py / ad_rec.py: TARGET_MAX_LEN
PRODUCT_VIDEO_HIST_MAX_LEN = 100  # product_rec.py: VIDEO_HIST_MAX_LEN
PRODUCT_HIST_MAX_LEN = 100        # product_rec.py: PRODUCT_HIST_MAX_LEN

OUTPUT_COLUMNS = (
    "uid",
    "target_domain",
    "history_pids",
    "history_captions",
    "target_pids",
    "target_captions",
    "history_meta",
)
ROWS_PER_SHARD = 1000


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        force=True,
    )


def chunk_for_rank(items: List[Any], rank: int, world_size: int) -> List[Any]:
    """round-robin 分片，与 reason/mpi 项目内其它脚本保持一致。"""
    return [item for idx, item in enumerate(items) if idx % world_size == rank]


def to_int_list(value: Any) -> List[int]:
    """把 parquet 里的 list[float] / list[int] / ndarray 安全转成 list[int]。

    - 兼容 hist_ad_pid / hist_goods_pid / hist_longview_video_list 的 float64
      表示（存在 NaN 的情况会被过滤）。
    - 兼容 hist_video_pid 的 int64 表示。
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    try:
        arr = np.asarray(value)
    except Exception:
        return []
    if arr.dtype.kind == "f":
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    return [int(x) for x in arr.tolist()]


def filter_pids_by_caption(
    pids: List[int],
    pid2caption: Dict[int, str],
) -> Tuple[List[int], List[str]]:
    """item-level drop：没有 caption 的 pid 直接跳过。"""
    kept_pids: List[int] = []
    kept_caps: List[str] = []
    for pid in pids:
        cap = pid2caption.get(pid)
        if not cap:
            continue
        kept_pids.append(pid)
        kept_caps.append(cap)
    return kept_pids, kept_caps


def keep_pids_with_optional_caption(
    pids: List[int],
    pid2caption: Dict[int, str],
) -> Tuple[List[int], List[str]]:
    """保留全部 pid，无 caption 的对应位置填空串。

    与 filter_pids_by_caption 的差异：不丢弃无 caption 的 pid，只把 caption 置空。
    用于 target 粒度筛选场景：无 caption 的 target 也要进 PPL 评估 / 训练池，
    由下游（pid2sid 映射 / PPL diff 门控）决定是否最终保留。

    history 不用此函数（无 caption 的 history pid 对 CoT prompt 生成无用，仍丢弃）。
    """
    kept_pids: List[int] = []
    kept_caps: List[str] = []
    for pid in pids:
        cap = pid2caption.get(pid) or ""
        kept_pids.append(pid)
        kept_caps.append(cap)
    return kept_pids, kept_caps


# -----------------------------------------------------------------------------
# caption 字典构建（方案 A：每 rank 全量读取）
# -----------------------------------------------------------------------------
def build_caption_dict(
    caption_uri: str,
    rank: int,
    caption_column: str,
    pid_column: str = "pid",
) -> Dict[int, str]:
    """构建 {pid: caption_text} 字典。

    支持两种输入：
    - 单个 parquet 文件（如 pid2caption.parquet，列 pid + dense_caption）
    - 目录（如 caption_summary/，含多个 part-*.parquet 分片，列 pid + dense_caption + summary）

    caption_column 指定用哪一列作为 caption 文本：
    - 'dense_caption'：原始完整描述（默认，prompt 长但信息全）
    - 'summary'：Qwen3-8B 生成的摘要（短，省 API 费用）
    - 'caption'：电商域 goods_pid2caption 的列名（内容即 dense caption，较短）

    pid_column 指定用哪一列作为 pid 键：
    - 'pid'（默认）：video/ad 域 pid2caption
    - 'new_pid'：电商域 goods_pid2caption（不要用 old_pid）
    """
    t0 = time.time()
    logger.info(
        "rank=%s 开始加载 caption: %s (pid_column=%s caption_column=%s)",
        rank, caption_uri, pid_column, caption_column,
    )

    caption_path = Path(caption_uri)
    if caption_path.is_dir():
        files = sorted(caption_path.glob("part-*.parquet"))
        if not files:
            raise FileNotFoundError(f"未在 {caption_uri} 找到 part-*.parquet 分片")
        logger.info("rank=%s 检测到目录输入: %s 个分片", rank, len(files))
    elif caption_path.is_file():
        files = [caption_path]
    else:
        raise FileNotFoundError(f"caption 路径不存在: {caption_uri}")

    pid2caption: Dict[int, str] = {}
    n_rows = 0
    for fpath in files:
        pf = pq.ParquetFile(str(fpath))
        # 校验列存在
        schema_names = pf.schema.names
        if caption_column not in schema_names:
            raise ValueError(
                f"列 '{caption_column}' 不存在于 {fpath}，可用列: {schema_names}"
            )
        if pid_column not in schema_names:
            raise ValueError(
                f"pid 列 '{pid_column}' 不存在于 {fpath}，可用列: {schema_names}"
            )
        for batch in pf.iter_batches(batch_size=200000, columns=[pid_column, caption_column]):
            table = batch.to_pandas()
            pids = table[pid_column].tolist()
            caps = table[caption_column].tolist()
            for pid, cap in zip(pids, caps):
                # skip None / NaN / empty caption
                if cap is None:
                    continue
                if isinstance(cap, float) and pd.isna(cap):
                    continue
                if isinstance(cap, str) and not cap.strip():
                    continue
                pid2caption[int(pid)] = cap
            n_rows += len(table)

    logger.info(
        "rank=%s 加载 caption 完成: entries=%s rows=%s files=%s "
        "pid_column=%s caption_column=%s 耗时=%.1fs",
        rank, len(pid2caption), n_rows, len(files),
        pid_column, caption_column, time.time() - t0,
    )
    return pid2caption


# -----------------------------------------------------------------------------
# Step1+2: 单行处理
# -----------------------------------------------------------------------------
def process_user_row(
    row: pd.Series,
    pid2caption: Dict[int, str],
    min_history_len: int,
    keep_no_caption_target: bool = False,
    target_max_len: int = 0,
    keep_no_caption_history: bool = False,
    goods_pid2caption: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """一个用户 → 最多 2 条样本（video / ad）。

    Args:
      keep_no_caption_target: True 时保留无 caption 的 target pid（target_captions
                              对应位置填空串）。默认 False（原行为：丢弃无 caption target）。
      keep_no_caption_history: True 时保留无 caption 的 history pid（history_captions
                               对应位置填空串），使 history pids 与 bench 完全一致
                               （仅做长度截断，不按 caption 丢 pid）。
                               默认 False（原行为：丢弃无 caption history pid）。
                               ★ SID PPL / 训练时 history 用 SID 渲染，无 caption 但
                                 有 SID 的 pid 也能提供上下文，开启此项可让 SID 路径
                                 看到更完整的 history。
      target_max_len: target 截断上限。0 = 不截断（保留全部，0717 新默认），
                      >0 = 取前 N 个（原行为 TARGET_MAX_LEN=10）。
    """
    uid = int(row["uid"])
    results: List[Dict[str, Any]] = []

    # ---------------- video ----------------
    # 字段对齐 video_rec.py:69-80
    hist_video_pids = to_int_list(row.get("hist_video_pid"))
    target_video_pids = to_int_list(row.get("target_video_pid"))
    # 截断：history 取最近 512，target 按 target_max_len 截断（0 = 不截断）
    hist_video_pids = hist_video_pids[-VIDEO_HIST_MAX_LEN:]
    if target_max_len > 0:
        target_video_pids = target_video_pids[:target_max_len]

    if keep_no_caption_history:
        hist_pids, hist_caps = keep_pids_with_optional_caption(
            hist_video_pids, pid2caption)
    else:
        hist_pids, hist_caps = filter_pids_by_caption(hist_video_pids, pid2caption)
    if keep_no_caption_target:
        tgt_pids, tgt_caps = keep_pids_with_optional_caption(
            target_video_pids, pid2caption)
    else:
        tgt_pids, tgt_caps = filter_pids_by_caption(target_video_pids, pid2caption)
    # target 没有任何 caption 的样本直接丢（无法构成监督信号）
    # ★ 但 keep_no_caption_target=True 时，tgt_pids 可能非空但 tgt_caps 含空串，
    #   仍需保留（下游 PID 评估 / 训练时 SID 渲染不依赖 caption）
    if hist_pids and tgt_pids and len(hist_pids) >= min_history_len:
        results.append({
            "uid": uid,
            "target_domain": "video",
            "history_pids": hist_pids,
            "history_captions": hist_caps,
            "target_pids": tgt_pids,
            "target_captions": tgt_caps,
            "history_meta": json.dumps({
                "field": "hist_video_pid",
                "hist_max_len": VIDEO_HIST_MAX_LEN,
                "target_max_len": target_max_len,
                "raw_hist_len": len(hist_video_pids),
                "raw_target_len": len(target_video_pids),
                "kept_hist_len": len(hist_pids),
                "kept_target_len": len(tgt_pids),
                "keep_no_caption_target": keep_no_caption_target,
                "keep_no_caption_history": keep_no_caption_history,
            }, ensure_ascii=False),
        })

    # ---------------- ad ----------------
    # 字段对齐 ad_rec.py:86-122
    # history = hist_longview_video_list[-100:] + hist_ad_pid[-200:]
    hist_longview = to_int_list(row.get("hist_longview_video_list"))
    hist_ad_pids = to_int_list(row.get("hist_ad_pid"))
    target_ad_pids = to_int_list(row.get("target_ad_pid"))

    hist_longview = hist_longview[-AD_VIDEO_HIST_MAX_LEN:]
    hist_ad_pids = hist_ad_pids[-AD_AD_HIST_MAX_LEN:]
    if target_max_len > 0:
        target_ad_pids = target_ad_pids[:target_max_len]

    # 两段 history 分别处理 caption，保留分段结构信息
    if keep_no_caption_history:
        lv_pids, lv_caps = keep_pids_with_optional_caption(
            hist_longview, pid2caption)
        ad_pids, ad_caps = keep_pids_with_optional_caption(
            hist_ad_pids, pid2caption)
    else:
        lv_pids, lv_caps = filter_pids_by_caption(hist_longview, pid2caption)
        ad_pids, ad_caps = filter_pids_by_caption(hist_ad_pids, pid2caption)
    ad_hist_pids = lv_pids + ad_pids
    ad_hist_caps = lv_caps + ad_caps
    if keep_no_caption_target:
        ad_tgt_pids, ad_tgt_caps = keep_pids_with_optional_caption(
            target_ad_pids, pid2caption)
    else:
        ad_tgt_pids, ad_tgt_caps = filter_pids_by_caption(target_ad_pids, pid2caption)

    if ad_hist_pids and ad_tgt_pids and len(ad_hist_pids) >= min_history_len:
        results.append({
            "uid": uid,
            "target_domain": "ad",
            "history_pids": ad_hist_pids,
            "history_captions": ad_hist_caps,
            "target_pids": ad_tgt_pids,
            "target_captions": ad_tgt_caps,
            "history_meta": json.dumps({
                "field": "hist_longview_video_list+hist_ad_pid",
                "video_hist_max_len": AD_VIDEO_HIST_MAX_LEN,
                "ad_hist_max_len": AD_AD_HIST_MAX_LEN,
                "target_max_len": target_max_len,
                "raw_video_hist_len": len(hist_longview),
                "raw_ad_hist_len": len(hist_ad_pids),
                "raw_target_len": len(target_ad_pids),
                "kept_video_hist_len": len(lv_pids),
                "kept_ad_hist_len": len(ad_pids),
                "kept_hist_len": len(ad_hist_pids),
                "kept_target_len": len(ad_tgt_pids),
                "keep_no_caption_target": keep_no_caption_target,
                "keep_no_caption_history": keep_no_caption_history,
            }, ensure_ascii=False),
        })

    # ---------------- product ----------------
    # 字段对齐 product_rec.py:85-123
    # history = hist_longview_video_list[-100:] + hist_goods_pid[-100:]
    # 视频段 caption 用 pid2caption（video/ad 域 caption），
    # 商品段 + target caption 用 goods_pid2caption（电商域 caption，key 为 new_pid）。
    # 若 goods_pid2caption 为 None（未启用 product 域），退化为空 dict → 商品段全被过滤。
    goods_cap: Dict[int, str] = goods_pid2caption if goods_pid2caption is not None else {}
    hist_longview_prod = to_int_list(row.get("hist_longview_video_list"))
    hist_goods_pids = to_int_list(row.get("hist_goods_pid"))
    target_goods_pids = to_int_list(row.get("target_goods_pid"))

    hist_longview_prod = hist_longview_prod[-PRODUCT_VIDEO_HIST_MAX_LEN:]
    hist_goods_pids = hist_goods_pids[-PRODUCT_HIST_MAX_LEN:]
    if target_max_len > 0:
        target_goods_pids = target_goods_pids[:target_max_len]

    # 两段 history 分别处理 caption：视频段用 pid2caption，商品段用 goods_pid2caption
    if keep_no_caption_history:
        lv_pids_p, lv_caps_p = keep_pids_with_optional_caption(hist_longview_prod, pid2caption)
        goods_pids, goods_caps = keep_pids_with_optional_caption(hist_goods_pids, goods_cap)
    else:
        lv_pids_p, lv_caps_p = filter_pids_by_caption(hist_longview_prod, pid2caption)
        goods_pids, goods_caps = filter_pids_by_caption(hist_goods_pids, goods_cap)
    prod_hist_pids = lv_pids_p + goods_pids
    prod_hist_caps = lv_caps_p + goods_caps
    if keep_no_caption_target:
        prod_tgt_pids, prod_tgt_caps = keep_pids_with_optional_caption(target_goods_pids, goods_cap)
    else:
        prod_tgt_pids, prod_tgt_caps = filter_pids_by_caption(target_goods_pids, goods_cap)

    if prod_hist_pids and prod_tgt_pids and len(prod_hist_pids) >= min_history_len:
        results.append({
            "uid": uid,
            "target_domain": "product",
            "history_pids": prod_hist_pids,
            "history_captions": prod_hist_caps,
            "target_pids": prod_tgt_pids,
            "target_captions": prod_tgt_caps,
            "history_meta": json.dumps({
                "field": "hist_longview_video_list+hist_goods_pid",
                "video_hist_max_len": PRODUCT_VIDEO_HIST_MAX_LEN,
                "product_hist_max_len": PRODUCT_HIST_MAX_LEN,
                "target_max_len": target_max_len,
                "raw_video_hist_len": len(hist_longview_prod),
                "raw_product_hist_len": len(hist_goods_pids),
                "raw_target_len": len(target_goods_pids),
                "kept_video_hist_len": len(lv_pids_p),
                "kept_product_hist_len": len(goods_pids),
                "kept_hist_len": len(prod_hist_pids),
                "kept_target_len": len(prod_tgt_pids),
                "keep_no_caption_target": keep_no_caption_target,
                "keep_no_caption_history": keep_no_caption_history,
            }, ensure_ascii=False),
        })

    return results


# -----------------------------------------------------------------------------
# 输出
# -----------------------------------------------------------------------------
def write_shards(
    records: List[Dict[str, Any]],
    output_dir: str,
    rank: int,
    start_part: int,
    rows_per_shard: int,
) -> int:
    """把 records 按 rows_per_shard 切片写出，返回下一个 part 编号。"""
    if not records:
        return start_part
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records)
    # 列顺序固定
    df = df.reindex(columns=list(OUTPUT_COLUMNS))
    n = len(df)
    n_shards = (n + rows_per_shard - 1) // rows_per_shard
    for i in range(n_shards):
        lo = i * rows_per_shard
        hi = min((i + 1) * rows_per_shard, n)
        shard = df.iloc[lo:hi]
        part_index = start_part + i
        # 跨 rank 全局唯一编号：part-{rank:05d}-{part:05d}.parquet
        fname = f"part-rank{rank:05d}-{part_index:05d}.parquet"
        fpath = os.path.join(output_dir, fname)
        table = pa.Table.from_pandas(shard, preserve_index=False)
        pq.write_table(table, fpath)
        logger.info("rank=%s 写出 %s rows=%s", rank, fpath, len(shard))
    return start_part + n_shards


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def run(
    bench_uri: str,
    caption_uri: str,
    caption_column: str,
    output_uri: str,
    rows_per_shard: int,
    min_history_len: int,
    verbose: bool,
    keep_no_caption_target: bool = False,
    target_max_len: int = 0,
    keep_no_caption_history: bool = False,
    goods_caption_uri: str = "",
    goods_caption_column: str = "caption",
    goods_pid_column: str = "new_pid",
    target_domain_filter: str = "",
) -> Dict[str, Any]:
    setup_logging(verbose)

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        comm = None
        rank = 0
        world_size = 1
        logger.warning("mpi4py 不可用，退化为单进程模式。")

    # 1) 每 rank 全量加载 caption dict（方案 A）
    pid2caption = build_caption_dict(caption_uri, rank, caption_column)

    # 1b) 电商域 caption（可选）：goods_pid2caption，key 为 new_pid（不要用 old_pid）
    goods_pid2caption: Dict[int, str] = {}
    if goods_caption_uri:
        goods_pid2caption = build_caption_dict(
            goods_caption_uri, rank, goods_caption_column, pid_column=goods_pid_column,
        )
    else:
        logger.info("rank=%s 未提供 goods_caption_uri，product 域 caption 过滤将退化为空（商品段全丢）", rank)

    # 2) 读 bench：单文件、162K 行，每 rank 全量读再 chunk（开销可忽略）
    t0 = time.time()
    logger.info("rank=%s 开始加载 bench: %s", rank, bench_uri)
    bench_df = pq.read_table(bench_uri).to_pandas()
    logger.info(
        "rank=%s 加载 bench 完成: rows=%s cols=%s 耗时=%.1fs",
        rank, len(bench_df), len(bench_df.columns), time.time() - t0,
    )

    # 3) 用户分片（round-robin）—— bench 一行 = 一个 uid 的完整多域序列
    user_indices = list(range(len(bench_df)))
    my_indices = chunk_for_rank(user_indices, rank, world_size)
    logger.info("rank=%s 分配用户数=%s / 总用户数=%s", rank, len(my_indices), len(user_indices))

    # 4) 处理
    local_records: List[Dict[str, Any]] = []
    local_stats = {"users": 0, "video_samples": 0, "ad_samples": 0, "product_samples": 0, "skipped_no_target": 0}
    logger.info(
        "rank=%s 配置: keep_no_caption_target=%s keep_no_caption_history=%s target_max_len=%s (0=不截断) "
        "target_domain_filter=%s goods_caption=%s",
        rank, keep_no_caption_target, keep_no_caption_history, target_max_len,
        target_domain_filter or "<all>", bool(goods_pid2caption),
    )
    progress = tqdm(my_indices, desc=f"rank{rank} process", disable=rank != 0)
    for idx in progress:
        row = bench_df.iloc[idx]
        local_stats["users"] += 1
        samples = process_user_row(
            row, pid2caption, min_history_len,
            keep_no_caption_target=keep_no_caption_target,
            target_max_len=target_max_len,
            keep_no_caption_history=keep_no_caption_history,
            goods_pid2caption=goods_pid2caption,
        )
        if target_domain_filter:
            samples = [s for s in samples if s["target_domain"] == target_domain_filter]
        for s in samples:
            if s["target_domain"] == "video":
                local_stats["video_samples"] += 1
            elif s["target_domain"] == "ad":
                local_stats["ad_samples"] += 1
            elif s["target_domain"] == "product":
                local_stats["product_samples"] += 1
        if not samples:
            local_stats["skipped_no_target"] += 1
        local_records.extend(samples)

    # 5) 写出
    t_write = time.time()
    n_written = write_shards(
        local_records,
        output_uri,
        rank=rank,
        start_part=0,
        rows_per_shard=rows_per_shard,
    )
    logger.info(
        "rank=%s 写出完成: shards=%s records=%s 耗时=%.1fs stats=%s",
        rank, n_written, len(local_records), time.time() - t_write, local_stats,
    )

    # 6) 汇总
    if comm is not None:
        gathered = comm.gather(local_stats, root=0)
        comm.Barrier()
        if rank == 0:
            agg = {"users": 0, "video_samples": 0, "ad_samples": 0, "product_samples": 0, "skipped_no_target": 0}
            for s in gathered:
                for k in agg:
                    agg[k] += s.get(k, 0)
            logger.info("汇总: %s", agg)
            print("\n================ Step 1 (history_text) Summary ================")
            print(f"  users            : {agg['users']}")
            print(f"  video_samples    : {agg['video_samples']}")
            print(f"  ad_samples       : {agg['ad_samples']}")
            print(f"  product_samples  : {agg['product_samples']}")
            print(f"  skipped_no_target: {agg['skipped_no_target']}")
            print(f"  output_dir       : {output_uri}")
            print("================================================================")
        else:
            return local_stats
    else:
        logger.info("单进程汇总: %s", local_stats)
        print("\n================ Step 1 (history_text) Summary ================")
        print(f"  users            : {local_stats['users']}")
        print(f"  video_samples    : {local_stats['video_samples']}")
        print(f"  ad_samples       : {local_stats['ad_samples']}")
        print(f"  product_samples  : {local_stats['product_samples']}")
        print(f"  skipped_no_target: {local_stats['skipped_no_target']}")
        print(f"  output_dir       : {output_uri}")
        print("================================================================")

    return local_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 1: 抽取 video/ad 历史 + caption 文本（MPI 并行）",
    )
    parser.add_argument(
        "--bench_uri",
        default=os.environ.get("BENCH_URI", ""),
        help="onerec_bench_release.parquet 路径（本地或 viewfs://）。默认读 BENCH_URI 环境变量。",
    )
    parser.add_argument(
        "--caption_uri",
        default=os.environ.get("CAPTION_URI", ""),
        help="pid2caption.parquet 路径，或 caption_summary/ 分片目录。默认读 CAPTION_URI 环境变量。",
    )
    parser.add_argument(
        "--caption_column",
        default=os.environ.get("CAPTION_COLUMN", "dense_caption"),
        help="caption 文本列名：'dense_caption'(原始完整描述，默认) 或 'summary'(Qwen3-8B 摘要)。"
             "默认读 CAPTION_COLUMN 环境变量。",
    )
    parser.add_argument(
        "--output_uri",
        default=os.environ.get("OUTPUT_URI", ""),
        help="输出目录。默认读 OUTPUT_URI 环境变量。",
    )
    parser.add_argument(
        "--rows_per_shard",
        type=int,
        default=int(os.environ.get("ROWS_PER_SHARD", ROWS_PER_SHARD)),
        help="每个输出 parquet 文件的行数，默认 1000。",
    )
    parser.add_argument(
        "--min_history_len",
        type=int,
        default=int(os.environ.get("MIN_HISTORY_LEN", 1)),
        help="history 保留 caption 后的最小长度，低于此值丢弃该样本。默认 1。",
    )
    parser.add_argument(
        "--keep_no_caption_target",
        type=int,
        default=int(os.environ.get("KEEP_NO_CAPTION_TARGET", 0)),
        help="target 粒度 caption 处理开关。"
             "0（默认）= 丢弃无 caption 的 target pid（原行为）。"
             "1 = 保留全部 target pid，无 caption 的对应位置填空串。"
             "默认读 KEEP_NO_CAPTION_TARGET 环境变量。",
    )
    parser.add_argument(
        "--keep_no_caption_history",
        type=int,
        default=int(os.environ.get("KEEP_NO_CAPTION_HISTORY", 0)),
        help="history 粒度 caption 处理开关。"
             "0（默认）= 丢弃无 caption 的 history pid（原行为，CoT prompt 只描述有 caption 的历史）。"
             "1 = 保留全部 history pid（仅做长度截断，不按 caption 丢），使 history pids 与 bench 完全一致。"
             "无 caption 的 history pid 在 CoT prompt 里隐形（caption 渲染跳过空串），"
             "但在 SID PPL / 训练里会以 SID 形式提供上下文（有 SID 的 pid）。"
             "默认读 KEEP_NO_CAPTION_HISTORY 环境变量。",
    )
    parser.add_argument(
        "--target_max_len",
        type=int,
        default=int(os.environ.get("TARGET_MAX_LEN_STEP1", 0)),
        help="target 截断上限。"
             "0（默认）= 不截断，保留 bench 全部 target（0717 新默认，配合 per-target PPL 筛选）。"
             ">0 = 取前 N 个（原行为 N=10，只评 / 只训前 10 个 target）。"
             "默认读 TARGET_MAX_LEN_STEP1 环境变量（与 5_data_format 的 TARGET_MAX_LEN 区分）。",
    )
    parser.add_argument(
        "--target_domain_filter",
        type=str,
        default=os.environ.get("TARGET_DOMAIN_FILTER", ""),
        choices=["", "video", "ad", "product"],
        help="仅处理指定目标域的样本（用于 product 域单独抽取，避免重复产出 video/ad）。"
             "空字符串=不过滤（处理 video+ad+product），"
             "'video'/'ad'/'product'=仅对应域。默认读 TARGET_DOMAIN_FILTER 环境变量。",
    )
    parser.add_argument(
        "--goods_caption_uri",
        default=os.environ.get("GOODS_CAPTION_URI", ""),
        help="电商域 goods_pid2caption 路径（目录或单文件），列含 new_pid + caption。"
             "启用 product 域（--target_domain_filter product 或全域）时必填。"
             "默认读 GOODS_CAPTION_URI 环境变量。",
    )
    parser.add_argument(
        "--goods_caption_column",
        default=os.environ.get("GOODS_CAPTION_COLUMN", "caption"),
        help="goods_pid2caption 中 caption 文本列名，默认 'caption'（内容即 dense caption，较短）。",
    )
    parser.add_argument(
        "--goods_pid_column",
        default=os.environ.get("GOODS_PID_COLUMN", "new_pid"),
        help="goods_pid2caption 中 pid 列名，默认 'new_pid'（★ 不要用 old_pid）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细的日志。",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    missing = []
    if not args.bench_uri:
        missing.append("--bench_uri / BENCH_URI")
    if not args.caption_uri:
        missing.append("--caption_uri / CAPTION_URI")
    if not args.output_uri:
        missing.append("--output_uri / OUTPUT_URI")
    if missing:
        raise ValueError("缺少必填参数: " + ", ".join(missing))
    # product 域依赖 goods caption：若 filter 选了 product 或未 filter（含 product）
    # 但没提供 goods_caption_uri，提示用户（不强制报错——退化为商品段全丢，但 product 样本数会为 0）
    need_goods = (args.target_domain_filter in ("", "product"))
    if need_goods and not args.goods_caption_uri:
        logger.warning(
            "包含 product 域但未提供 --goods_caption_uri，product 域商品段 / target "
            "caption 过滤将退化为全丢（product 样本数为 0）。若仅处理 video/ad 请用 "
            "--target_domain_filter video/ad。"
        )
    run(
        bench_uri=args.bench_uri,
        caption_uri=args.caption_uri,
        caption_column=args.caption_column,
        output_uri=args.output_uri,
        rows_per_shard=args.rows_per_shard,
        min_history_len=args.min_history_len,
        verbose=args.verbose,
        keep_no_caption_target=bool(args.keep_no_caption_target),
        target_max_len=args.target_max_len,
        keep_no_caption_history=bool(args.keep_no_caption_history),
        goods_caption_uri=args.goods_caption_uri,
        goods_caption_column=args.goods_caption_column,
        goods_pid_column=args.goods_pid_column,
        target_domain_filter=args.target_domain_filter,
    )


if __name__ == "__main__":
    main()
