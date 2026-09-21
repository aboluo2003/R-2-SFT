#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a (ppl_eval preprocess): 统一生成 PPL 评估输入（MPI 并行）。

目标：
  把 step 2 (2_cot) 产出的 CoT 样本，整理成后续 PPL 计算可直接消费的统一 schema。
  同一份主脚本支持两种表示形式：
  1. caption: 直接使用 history/target caption
  2. sid:     使用 pid2sid 把 history/target pid 转成 OpenOneRec item token

设计要点：
  - 每个 target 单独 explode 成一行，便于逐 target 计算：
      PPL(target | input)
      PPL(target | input, cot)
  - 直接产出统一字段：
      system_text / input_text / input_with_cot_text / target_text
    这样后续 PPL 主脚本只需要切换输入文件与模型路径即可复用。
  - 支持 MPI：目录输入时按 parquet shard round-robin；单文件输入时按行 round-robin。

输入 schema（来自 step2 / step4）：
  sample_key, uid, target_domain, cot,
  history_pids, history_captions, target_pids, target_captions, history_meta

输出 schema（每行 = 一个 target）：
  ppl_sample_key        : str   sample_key + target_idx
  sample_key            : str
  uid                   : int
  target_domain         : str   ('video' | 'ad')
  target_idx            : int
  target_pid            : int
  representation_type   : str   ('caption' | 'sid')
  system_text           : str
  input_text            : str
  input_with_cot_text   : str
  target_text           : str
  cot                   : str
  history_values        : list[str]
  history_pids          : list[int]
  history_length        : int
  history_meta          : str
  render_meta           : str(JSON)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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

REPRESENTATION_CAPTION = "caption"
REPRESENTATION_SID = "sid"
SID_FORMAT = "<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>"

# ★ history 截断上限：与 5_data_format/main.py 对齐（0717 新增 bench 模式用）
# target 不截断（保留全量，让 PPL eval 覆盖全部 target → 4a 能给所有 target 打标记）
VIDEO_HIST_MAX_LEN = 512
AD_VIDEO_HIST_MAX_LEN = 100
AD_AD_HIST_MAX_LEN = 200
# product 域（0724 ver3 新增，与 5_data_format/main.py + product_rec.py 对齐）
PRODUCT_VIDEO_HIST_MAX_LEN = 100  # product_rec.py: VIDEO_HIST_MAX_LEN
PRODUCT_HIST_MAX_LEN = 100        # product_rec.py: PRODUCT_HIST_MAX_LEN

OUTPUT_COLUMNS = (
    "ppl_sample_key",
    "sample_key",
    "uid",
    "target_domain",
    "target_idx",
    "target_pid",
    "representation_type",
    "system_text",
    "input_text",
    "input_with_cot_text",
    "target_text",
    "cot",
    "history_values",
    "history_pids",
    "history_length",
    "history_meta",
    "render_meta",
)

SYSTEM_TEXTS = {
    # 与训练侧 video_rec.py / 5_data_format VIDEO_SYSTEM_PROMPTS[0] 对齐（固定，不随机）
    ("video", REPRESENTATION_CAPTION): (
        "你是一个智能推荐助手，能够根据用户的浏览历史预测用户可能感兴趣的下一个内容。"
    ),
    # 与训练侧 ad_rec.py / 5_data_format AD_SYSTEM_PROMPTS[0] 对齐
    ("ad", REPRESENTATION_CAPTION): (
        "你是一个智能广告推荐助手，能够根据用户的视频观看历史和广告点击行为，预测用户接下来可能点击的广告视频。"
    ),
    ("video", REPRESENTATION_SID): (
        "你是一个智能推荐助手，能够根据用户的浏览历史预测用户可能感兴趣的下一个内容。"
    ),
    ("ad", REPRESENTATION_SID): (
        "你是一个智能广告推荐助手，能够根据用户的视频观看历史和广告点击行为，预测用户接下来可能点击的广告视频。"
    ),
    # 0724 ver3 新增 product 域（与 5_data_format/main.py PRODUCT_SYSTEM_PROMPTS[0] 对齐）
    ("product", REPRESENTATION_CAPTION): (
        "你是一个智能跨域推荐助手，能够根据用户观看的视频内容和历史购物行为，预测用户接下来可能点击的商品。"
    ),
    ("product", REPRESENTATION_SID): (
        "你是一个智能跨域推荐助手，能够根据用户观看的视频内容和历史购物行为，预测用户接下来可能点击的商品。"
    ),
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        force=True,
    )


def chunk_for_rank(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    return [item for idx, item in enumerate(items) if idx % world_size == rank]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def to_int_list(value: Any) -> List[int]:
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


def to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not isinstance(value, (list, tuple)):
        value = [value]
    result: List[str] = []
    for item in value:
        text = safe_str(item)
        if text:
            result.append(text)
    return result


def parse_json_dict(value: Any) -> Dict[str, Any]:
    text = safe_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def list_parquet_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet"))
        if not files:
            files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"未在 {path} 找到 parquet 文件")
        return files
    raise FileNotFoundError(f"输入路径不存在: {path}")


def load_pid2sid(path: str, rank: int) -> Dict[int, Any]:
    t0 = time.time()
    logger.info("rank=%s 加载 pid2sid: %s", rank, path)
    df = pd.read_parquet(path, columns=["pid", "sid"])
    mapping = dict(zip(df["pid"].tolist(), df["sid"].tolist()))
    logger.info(
        "rank=%s 加载 pid2sid 完成: entries=%s 耗时=%.1fs",
        rank, len(mapping), time.time() - t0,
    )
    return mapping


def pid_to_sid_str(pid: int, pid2sid: Dict[int, Any]) -> str:
    if pid not in pid2sid:
        return ""
    code = pid2sid[pid]
    try:
        return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
    except (IndexError, TypeError, KeyError):
        return ""


def load_bench(bench_uri: str, rank: int) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """加载 bench parquet，构建 uid→row_idx 映射（0717 新增 bench 模式用）。

    与 5_data_format/main.py 的 load_bench_df 对齐，用于 target_source=bench 时
    重建 history/target pids（与训练完全一致，不丢无 caption pid，仅长度截断）。
    """
    t0 = time.time()
    logger.info("rank=%s 加载 bench (full): %s", rank, bench_uri)
    bench_df = pq.read_table(bench_uri).to_pandas()
    uid_list = bench_df["uid"].astype(int).tolist()
    uid_to_idx = {uid: idx for idx, uid in enumerate(uid_list)}
    logger.info(
        "rank=%s 加载 bench 完成: rows=%s cols=%s 耗时=%.1fs",
        rank, len(bench_df), len(bench_df.columns), time.time() - t0,
    )
    return bench_df, uid_to_idx


def rebuild_pids_from_bench(
    bench_row: pd.Series,
    target_domain: str,
) -> Tuple[List[int], List[int], Dict[str, Any], bool]:
    """从 bench row 重建 history/target pids，与 5_data_format 训练对齐。

    - history: 仅长度截断（VIDEO_HIST_MAX_LEN / AD_*_HIST_MAX_LEN），不按 caption 丢 pid。
    - target: 不截断（全量，让 PPL eval 覆盖全部 target → 4a 能给所有 target 打标记）。
    - ad 域返回分段长度，供下游 split_ad_history 用。

    返回 (history_pids, target_pids, history_meta, ok)。
    bench_row 为 None 或字段缺失时 ok=False，调用方应回退到 upstream（2_cot pids）。
    """
    if bench_row is None:
        return [], [], {}, False
    if target_domain == "video":
        hist_pids = to_int_list(bench_row.get("hist_video_pid"))[-VIDEO_HIST_MAX_LEN:]
        tgt_pids = to_int_list(bench_row.get("target_video_pid"))
        meta = {
            "field": "hist_video_pid",
            "kept_video_hist_len": len(hist_pids),
            "kept_ad_hist_len": 0,
        }
        return hist_pids, tgt_pids, meta, True
    elif target_domain == "ad":
        video_pids = to_int_list(bench_row.get("hist_longview_video_list"))[-AD_VIDEO_HIST_MAX_LEN:]
        ad_pids = to_int_list(bench_row.get("hist_ad_pid"))[-AD_AD_HIST_MAX_LEN:]
        hist_pids = video_pids + ad_pids
        tgt_pids = to_int_list(bench_row.get("target_ad_pid"))
        meta = {
            "field": "hist_longview_video_list+hist_ad_pid",
            "kept_video_hist_len": len(video_pids),
            "kept_ad_hist_len": len(ad_pids),
        }
        return hist_pids, tgt_pids, meta, True
    elif target_domain == "product":
        # 0724 ver3 新增：product 域，与 5_data_format/main.py get_orig_history_pids 对齐
        # history = hist_longview_video_list[-100:] + hist_goods_pid[-100:]
        # target  = target_goods_pid
        video_pids = to_int_list(bench_row.get("hist_longview_video_list"))[-PRODUCT_VIDEO_HIST_MAX_LEN:]
        goods_pids = to_int_list(bench_row.get("hist_goods_pid"))[-PRODUCT_HIST_MAX_LEN:]
        hist_pids = video_pids + goods_pids
        tgt_pids = to_int_list(bench_row.get("target_goods_pid"))
        meta = {
            "field": "hist_longview_video_list+hist_goods_pid",
            "kept_video_hist_len": len(video_pids),
            "kept_product_hist_len": len(goods_pids),
        }
        return hist_pids, tgt_pids, meta, True
    return [], [], {}, False


def build_representation_values(
    representation_type: str,
    pids: List[int],
    captions: List[str],
    pid2sid: Optional[Dict[int, Any]],
) -> Tuple[List[str], List[int], int]:
    if representation_type == REPRESENTATION_CAPTION:
        values: List[str] = []
        kept_pids: List[int] = []
        source_len = min(len(pids), len(captions))
        for pid, caption in zip(pids[:source_len], captions[:source_len]):
            text = safe_str(caption)
            if not text:
                continue
            kept_pids.append(int(pid))
            values.append(text)
        dropped = len(pids) - len(kept_pids)
        return values, kept_pids, dropped

    if pid2sid is None:
        raise ValueError("sid 模式要求提供 pid2sid 映射")

    values = []
    kept_pids = []
    for pid in pids:
        sid = pid_to_sid_str(int(pid), pid2sid)
        if not sid:
            continue
        kept_pids.append(int(pid))
        values.append(sid)
    dropped = len(pids) - len(kept_pids)
    return values, kept_pids, dropped


def split_ad_history(
    history_values: List[str],
    history_pids: List[int],
    history_meta: Dict[str, Any],
) -> Dict[str, Dict[str, List[Any]]]:
    video_len = int(history_meta.get("kept_video_hist_len", 0) or 0)
    ad_len = int(history_meta.get("kept_ad_hist_len", 0) or 0)
    if video_len < 0:
        video_len = 0
    if ad_len < 0:
        ad_len = 0

    if video_len + ad_len > len(history_values):
        video_len = min(video_len, len(history_values))
        ad_len = max(0, len(history_values) - video_len)

    if video_len == 0 and ad_len == 0:
        ad_len = len(history_values)

    return {
        "video": {
            "values": history_values[:video_len],
            "pids": history_pids[:video_len],
        },
        "ad": {
            "values": history_values[video_len:video_len + ad_len],
            "pids": history_pids[video_len:video_len + ad_len],
        },
    }


def split_product_history(
    history_values: List[str],
    history_pids: List[int],
    history_meta: Dict[str, Any],
) -> Dict[str, Dict[str, List[Any]]]:
    """product 域两段式 history 切分（video 段 + goods 段）。

    与 split_ad_history 结构一致，但 meta key 用 kept_product_hist_len（与
    1_history_text/main.py product 分支对齐），第二段 key 为 "product"。
    """
    video_len = int(history_meta.get("kept_video_hist_len", 0) or 0)
    product_len = int(history_meta.get("kept_product_hist_len", 0) or 0)
    if video_len < 0:
        video_len = 0
    if product_len < 0:
        product_len = 0

    if video_len + product_len > len(history_values):
        video_len = min(video_len, len(history_values))
        product_len = max(0, len(history_values) - video_len)

    if video_len == 0 and product_len == 0:
        product_len = len(history_values)

    return {
        "video": {
            "values": history_values[:video_len],
            "pids": history_pids[:video_len],
        },
        "product": {
            "values": history_values[video_len:video_len + product_len],
            "pids": history_pids[video_len:video_len + product_len],
        },
    }


def render_history_block(values: Sequence[str], representation_type: str) -> str:
    if not values:
        return "（空）"
    if representation_type == REPRESENTATION_CAPTION:
        lines = [f"{idx}. {safe_str(value)}" for idx, value in enumerate(values, start=1) if safe_str(value)]
        return "\n".join(lines) if lines else "（空）"
    return "".join(safe_str(value) for value in values if safe_str(value)) or "（空）"


def build_video_input_text(
    representation_type: str,
    history_values: List[str],
    cot_text: str,
) -> Tuple[str, str]:
    """video 域 user message 文本，与训练侧 video_rec.py / 5_data_format 对齐（固定 prompt，不随机）。

    训练侧 USER_PROMPTS[0]: "根据以下用户浏览记录，请预测用户接下来可能观看的内容：\\n{query}"
    结构: {intro}:\\n{sids}

    ★ cot 不放在 user message（旧设计错误）。cot 由 2_eval_ppl.py 放到 assistant 的
      <think>...</think> 块里，与训练侧 build_cot_answer 对齐。因此 input_text ==
      input_with_cot_text（二者相同，cot 字段单独保存原 cot 文本）。
    """
    history_block = render_history_block(history_values, representation_type)
    # 与训练侧 video_rec.py USER_PROMPTS[0] 结构对齐：intro 独占一行，sids 在下一行
    base = f"根据以下用户浏览记录，请预测用户接下来可能观看的内容：\n{history_block}"
    # cot 不插入 user message；input_with_cot_text 与 input_text 相同
    return base, base


def build_ad_input_text(
    representation_type: str,
    history_values: List[str],
    history_pids: List[int],
    history_meta: Dict[str, Any],
    cot_text: str,
) -> Tuple[str, str, Dict[str, int]]:
    """ad 域 input 文本，与训练侧 ad_rec.py / 5_data_format 对齐（固定 prompt，不随机）。

    训练侧结构（ad_rec.py process_row / 5_data_format build_cot_record ad 分支）:
      user_parts = [f"{AD_VIDEO_WATCH_PROMPTS[i]}{video_sids}", f"{AD_AD_CLICK_PROMPTS[i]}{ad_sids}"]
      user_content = "\\n".join(user_parts)
      user_prompt = user_content + "\\n" + AD_TASK_PROMPTS[i]
    即: {label}{sids}\\n{label}{sids}\\n{task}  (label 和 sids 同行，段间单换行)
    """
    segments = split_ad_history(history_values, history_pids, history_meta)
    video_values = segments["video"]["values"]
    ad_values = segments["ad"]["values"]

    video_block = render_history_block(video_values, representation_type)
    ad_block = render_history_block(ad_values, representation_type)
    # 与训练侧 AD_VIDEO_WATCH_PROMPTS[0] / AD_AD_CLICK_PROMPTS[0] 对齐（label 与 sids 同行）
    video_part = f"用户观看过的视频：{video_block}" if video_block else ""
    ad_part = f"用户点击过的广告视频：{ad_block}" if ad_block else ""
    user_content = "\n".join(p for p in [video_part, ad_part] if p)
    # 与训练侧 AD_TASK_PROMPTS[0] 对齐
    task_prompt = "请根据用户的观看和广告点击历史，预测用户接下来可能点击的广告视频。"
    base = f"{user_content}\n{task_prompt}"
    # cot 不插入 user message；input_with_cot_text 与 input_text 相同
    return base, base, {
        "video_history_len": len(video_values),
        "ad_history_len": len(ad_values),
    }


def build_product_input_text(
    representation_type: str,
    history_values: List[str],
    history_pids: List[int],
    history_meta: Dict[str, Any],
    cot_text: str,
) -> Tuple[str, str, Dict[str, int]]:
    """product 域 input 文本（两段式：用户观看视频 + 用户点击商品），与训练侧
    product_rec.py / 5_data_format 对齐（固定 prompt，不随机）。

    训练侧结构（product_rec.py process_row / 5_data_format build_cot_record product 分支）:
      user_parts = [f"{PRODUCT_VIDEO_WATCH_PROMPTS[i]}{video_sids}", f"{PRODUCT_CLICK_PROMPTS[i]}{goods_sids}"]
      user_content = "\\n".join(user_parts)
      user_prompt = user_content + "\\n" + PRODUCT_TASK_PROMPTS[i]
    即: {label}{sids}\\n{label}{sids}\\n{task}  (label 和 sids 同行，段间单换行)
    """
    segments = split_product_history(history_values, history_pids, history_meta)
    video_values = segments["video"]["values"]
    goods_values = segments["product"]["values"]

    video_block = render_history_block(video_values, representation_type)
    goods_block = render_history_block(goods_values, representation_type)
    # 与训练侧 PRODUCT_VIDEO_WATCH_PROMPTS[0] / PRODUCT_CLICK_PROMPTS[0] 对齐（label 与 sids 同行）
    video_part = f"用户观看过的视频：{video_block}" if video_block else ""
    goods_part = f"用户点击过的商品：{goods_block}" if goods_block else ""
    user_content = "\n".join(p for p in [video_part, goods_part] if p)
    # 与训练侧 PRODUCT_TASK_PROMPTS[0] 对齐
    task_prompt = "请根据用户的观看和购物历史，预测用户接下来可能点击的商品。"
    base = f"{user_content}\n{task_prompt}"
    # cot 不插入 user message；input_with_cot_text 与 input_text 相同
    return base, base, {
        "video_history_len": len(video_values),
        "product_history_len": len(goods_values),
    }


def build_prompt_pair(
    target_domain: str,
    representation_type: str,
    history_values: List[str],
    history_pids: List[int],
    history_meta: Dict[str, Any],
    cot_text: str,
) -> Tuple[str, str, Dict[str, int]]:
    if target_domain == "video":
        base, with_cot = build_video_input_text(
            representation_type=representation_type,
            history_values=history_values,
            cot_text=cot_text,
        )
        return base, with_cot, {
            "video_history_len": len(history_values),
            "ad_history_len": 0,
        }

    if target_domain == "ad":
        return build_ad_input_text(
            representation_type=representation_type,
            history_values=history_values,
            history_pids=history_pids,
            history_meta=history_meta,
            cot_text=cot_text,
        )

    if target_domain == "product":
        return build_product_input_text(
            representation_type=representation_type,
            history_values=history_values,
            history_pids=history_pids,
            history_meta=history_meta,
            cot_text=cot_text,
        )

    raise ValueError(f"不支持的 target_domain: {target_domain}")


def iter_my_frames(
    input_path: Path,
    rank: int,
    world_size: int,
) -> Iterator[pd.DataFrame]:
    columns = [
        "sample_key",
        "uid",
        "target_domain",
        "cot",
        "history_pids",
        "history_captions",
        "target_pids",
        "target_captions",
        "history_meta",
    ]

    if input_path.is_file():
        logger.info("rank=%s 读取单文件输入: %s", rank, input_path)
        df = pq.read_table(input_path, columns=columns).to_pandas()
        my_indices = chunk_for_rank(list(range(len(df))), rank, world_size)
        if my_indices:
            yield df.iloc[my_indices].reset_index(drop=True)
        return

    files = list_parquet_files(input_path)
    my_files = chunk_for_rank(files, rank, world_size)
    logger.info("rank=%s 分配输入分片=%s / 总分片=%s", rank, len(my_files), len(files))
    for path in my_files:
        yield pq.read_table(path, columns=columns).to_pandas()


def iter_records_from_frame(
    frame: pd.DataFrame,
    representation_type: str,
    pid2sid: Optional[Dict[int, Any]],
    product_pid2sid: Optional[Dict[int, Any]] = None,
    bench_df: Optional[pd.DataFrame] = None,
    uid_to_idx: Optional[Dict[int, int]] = None,
) -> Iterable[Dict[str, Any]]:
    """逐行生成 PPL eval 记录。

    Args:
      bench_df / uid_to_idx: target_source=bench 时传入，用于从 bench row 重建
                             history/target pids（与 5_data_format 训练对齐，不丢
                             无 caption pid，仅长度截断）。None 时用上游 2_cot pids。
      product_pid2sid: product 域 goods pid → sid 映射（0724 ver3 新增）。
                       product 域商品段 history + target 用此映射，视频段仍用 pid2sid。
    """
    use_bench = bench_df is not None and uid_to_idx is not None
    for row in frame.to_dict(orient="records"):
        sample_key = safe_str(row.get("sample_key"))
        uid = int(row["uid"])
        target_domain = safe_str(row.get("target_domain"))
        cot_text = safe_str(row.get("cot"))
        history_meta_json = parse_json_dict(row.get("history_meta"))

        # ---- pids 来源选择 ----
        bench_row: Optional[pd.Series] = None
        if use_bench:
            bench_row = bench_df.iloc[uid_to_idx[uid]] if uid in uid_to_idx else None

        if use_bench and bench_row is not None:
            # ★ bench 模式：从 bench 重建 pids（与训练对齐，不丢无 caption pid）
            hist_pids, tgt_pids, rebuilt_meta, ok = rebuild_pids_from_bench(
                bench_row, target_domain)
            if not ok:
                # bench 字段缺失 → 回退到上游（2_cot pids）
                hist_pids = to_int_list(row.get("history_pids"))
                tgt_pids = to_int_list(row.get("target_pids"))
            else:
                history_meta_json = rebuilt_meta
            # SID 模式不读 captions，填空即可
            history_captions: List[str] = []
            target_captions: List[str] = []
        else:
            # upstream 模式（或 bench 缺失回退）：用 2_cot 原始 pids（caption-filtered）
            history_captions = to_str_list(row.get("history_captions"))
            target_captions = to_str_list(row.get("target_captions"))
            if not use_bench:
                hist_pids = to_int_list(row.get("history_pids"))
                tgt_pids = to_int_list(row.get("target_pids"))
            # use_bench=True 但 bench_row=None 的回退已在上面处理

        # ---- 构造 history / target 的 SID 表示 ----
        if use_bench and bench_row is not None and target_domain == "ad":
            # ★ ad bench 模式：video / ad 两段分别 SID 化，保留分段长度给 split_ad_history
            video_pids_raw = to_int_list(bench_row.get("hist_longview_video_list"))[-AD_VIDEO_HIST_MAX_LEN:]
            ad_pids_raw = to_int_list(bench_row.get("hist_ad_pid"))[-AD_AD_HIST_MAX_LEN:]
            video_values, video_kept_pids, video_dropped = build_representation_values(
                representation_type=representation_type,
                pids=video_pids_raw, captions=[], pid2sid=pid2sid,
            )
            ad_values, ad_kept_pids, ad_dropped = build_representation_values(
                representation_type=representation_type,
                pids=ad_pids_raw, captions=[], pid2sid=pid2sid,
            )
            history_values = video_values + ad_values
            kept_history_pids = video_kept_pids + ad_kept_pids
            dropped_history = video_dropped + ad_dropped
            history_meta_json = {
                "kept_video_hist_len": len(video_values),
                "kept_ad_hist_len": len(ad_values),
            }
        elif use_bench and bench_row is not None and target_domain == "product":
            # ★ product bench 模式（0724 ver3 新增）：video / goods 两段分别 SID 化
            #   video 段用 pid2sid（video/ad sid），goods 段用 product_pid2sid（电商 sid）
            video_pids_raw = to_int_list(bench_row.get("hist_longview_video_list"))[-PRODUCT_VIDEO_HIST_MAX_LEN:]
            goods_pids_raw = to_int_list(bench_row.get("hist_goods_pid"))[-PRODUCT_HIST_MAX_LEN:]
            video_values, video_kept_pids, video_dropped = build_representation_values(
                representation_type=representation_type,
                pids=video_pids_raw, captions=[], pid2sid=pid2sid,
            )
            goods_values, goods_kept_pids, goods_dropped = build_representation_values(
                representation_type=representation_type,
                pids=goods_pids_raw, captions=[], pid2sid=product_pid2sid,
            )
            history_values = video_values + goods_values
            kept_history_pids = video_kept_pids + goods_kept_pids
            dropped_history = video_dropped + goods_dropped
            history_meta_json = {
                "kept_video_hist_len": len(video_values),
                "kept_product_hist_len": len(goods_values),
            }
        elif target_domain == "product":
            # ★ product upstream（bench_row=None 或 use_bench=False）：从 2_cot pids 按 meta 切分
            #   - sid 模式：goods 段必须用 product_pid2sid，否则 goods pid 会被 video_ad pid2sid 丢掉
            #   - caption 模式（0726 新增）：按 meta 把 history_captions 切成 video 段 / goods 段，
            #     分别传给 build_representation_values。video 段 caption 来自 pid2caption summary
            #     （1_history_text 用 CAPTION_COLUMN=summary），goods 段 caption 来自 goods_pid2caption
            #     （1_history_text run_product_only.sh 用 --goods_caption_column caption）。
            #     两者在 2_cot_merged 里已按 [video_seg, goods_seg] 顺序拼到 history_captions，
            #     与 history_pids 一一对应，故切分方式与 pids 完全一致。
            meta_video_len = int(history_meta_json.get("kept_video_hist_len", 0) or 0)
            meta_product_len = int(history_meta_json.get("kept_product_hist_len", 0) or 0)
            if meta_video_len + meta_product_len > 0 and meta_video_len + meta_product_len <= len(hist_pids):
                video_pids_raw = hist_pids[:meta_video_len]
                goods_pids_raw = hist_pids[meta_video_len:meta_video_len + meta_product_len]
            else:
                # meta 缺失 → 假设全是商品段
                video_pids_raw = []
                goods_pids_raw = hist_pids
            # ★ caption 模式：按 meta 切分 history_captions（与 hist_pids 一一对应）；
            #   sid 模式：captions=[]（pid2sid 不读 caption，保持原逻辑）
            if representation_type == REPRESENTATION_CAPTION:
                n_caps = len(history_captions)
                if meta_video_len + meta_product_len > 0 and meta_video_len + meta_product_len <= n_caps:
                    video_caps = history_captions[:meta_video_len]
                    goods_caps = history_captions[meta_video_len:meta_video_len + meta_product_len]
                else:
                    # meta 缺失 → 假设全是商品段
                    video_caps = []
                    goods_caps = history_captions
            else:
                video_caps = []
                goods_caps = []
            video_values, video_kept_pids, video_dropped = build_representation_values(
                representation_type=representation_type,
                pids=video_pids_raw, captions=video_caps, pid2sid=pid2sid,
            )
            goods_values, goods_kept_pids, goods_dropped = build_representation_values(
                representation_type=representation_type,
                pids=goods_pids_raw, captions=goods_caps, pid2sid=product_pid2sid,
            )
            history_values = video_values + goods_values
            kept_history_pids = video_kept_pids + goods_kept_pids
            dropped_history = video_dropped + goods_dropped
            history_meta_json = {
                "kept_video_hist_len": len(video_values),
                "kept_product_hist_len": len(goods_values),
            }
        else:
            history_values, kept_history_pids, dropped_history = build_representation_values(
                representation_type=representation_type,
                pids=hist_pids,
                captions=history_captions,
                pid2sid=pid2sid,
            )
        if not history_values:
            continue

        # ★ product 域 target 必须用 product_pid2sid（电商 sid），无 fallback（0728 修复，
        #   对齐 5_data_format/main.py 0726 的 fix bug: fallback when pid2sid(product)）。
        #   旧 bug：`product_pid2sid if (target_domain == "product" and product_pid2sid) else pid2sid`
        #   在 product_pid2sid 为 None/{} 时静默 fallback 到 video/ad pid2sid，因 pid 数值
        #   重叠会让 product target 拿到错误 sid（样本不丢但 sid 100% 错，极难发现）。
        if target_domain == "product":
            if not product_pid2sid:
                raise ValueError(
                    "product 域 PPL eval 样本要求 product_pid2sid 非空（电商 sid 映射），"
                    "当前为 None/{}。请检查 1_run_preprocess.sh 是否传入 "
                    "--product_pid2sid_uri，以及路径是否存在。"
                    "（旧 bug：fallback 到 video/ad pid2sid，因 pid 数值重叠得到错误 sid）"
                )
            target_pid2sid_map = product_pid2sid
        else:
            target_pid2sid_map = pid2sid
        target_values, kept_target_pids, dropped_target = build_representation_values(
            representation_type=representation_type,
            pids=tgt_pids,
            captions=target_captions,
            pid2sid=target_pid2sid_map,
        )
        if not target_values:
            continue

        input_text, input_with_cot_text, segment_lengths = build_prompt_pair(
            target_domain=target_domain,
            representation_type=representation_type,
            history_values=history_values,
            history_pids=kept_history_pids,
            history_meta=history_meta_json,
            cot_text=cot_text,
        )

        system_text = SYSTEM_TEXTS[(target_domain, representation_type)]
        for target_idx, (target_pid, target_text) in enumerate(zip(kept_target_pids, target_values)):
            ppl_sample_key = f"{sample_key}#target{target_idx:02d}"
            render_meta = {
                "representation_type": representation_type,
                "source_target_count": len(tgt_pids),
                "kept_target_count": len(target_values),
                "dropped_history_count": dropped_history,
                "dropped_target_count": dropped_target,
                "segment_lengths": segment_lengths,
                "target_source": "bench" if (use_bench and bench_row is not None) else "upstream",
            }
            yield {
                "ppl_sample_key": ppl_sample_key,
                "sample_key": sample_key,
                "uid": uid,
                "target_domain": target_domain,
                "target_idx": target_idx,
                "target_pid": int(target_pid),
                "representation_type": representation_type,
                "system_text": system_text,
                "input_text": input_text,
                "input_with_cot_text": input_with_cot_text,
                "target_text": target_text,
                "cot": cot_text,
                "history_values": history_values,
                "history_pids": kept_history_pids,
                "history_length": len(history_values),
                "history_meta": json.dumps(history_meta_json, ensure_ascii=False),
                "render_meta": json.dumps(render_meta, ensure_ascii=False),
            }


def write_shards(
    records: List[Dict[str, Any]],
    output_dir: str,
    rank: int,
    start_part: int,
    rows_per_shard: int,
) -> int:
    if not records:
        return start_part

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records).reindex(columns=list(OUTPUT_COLUMNS))
    n_rows = len(df)
    n_shards = (n_rows + rows_per_shard - 1) // rows_per_shard

    for shard_idx in range(n_shards):
        lo = shard_idx * rows_per_shard
        hi = min((shard_idx + 1) * rows_per_shard, n_rows)
        shard_df = df.iloc[lo:hi]
        part_idx = start_part + shard_idx
        file_name = f"part-rank{rank:05d}-{part_idx:05d}.parquet"
        file_path = os.path.join(output_dir, file_name)
        table = pa.Table.from_pandas(shard_df, preserve_index=False)
        pq.write_table(table, file_path)
        logger.info("rank=%s 写出 %s rows=%s", rank, file_path, len(shard_df))

    return start_part + n_shards


def run(args: argparse.Namespace) -> Dict[str, Any]:
    setup_logging(args.verbose)

    # ★ 与 2_eval_ppl.py init_distributed 一致：mpirun 下不 fallback。
    #   若 mpi4py 缺失，每个进程都当自己是 rank 0 跑全集，输出互相覆盖。
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        if MPI is None:
            raise RuntimeError(
                "检测到 mpirun 环境（OMPI_COMM_WORLD_RANK 已设置）但 mpi4py 未安装。"
                "多进程下若 fallback 到单进程，每个进程都会当自己是 rank 0 跑全集，"
                "浪费算力且输出互相覆盖。请安装 mpi4py：\n"
                "  pip install mpi4py\n"
                "并确保 mpi4py 链接的 MPI 库与 mpirun 使用的 OpenMPI 版本一致。"
            )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        comm = None
        rank = 0
        world_size = 1
        logger.warning("未检测到 mpirun 环境（OMPI_COMM_WORLD_RANK 未设置），退化为单进程模式。")

    input_path = Path(args.input_uri)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    pid2sid: Optional[Dict[int, Any]] = None
    if args.representation_type == REPRESENTATION_SID:
        if not args.pid2sid_uri:
            raise ValueError("sid 模式必须提供 --pid2sid_uri")
        pid2sid = load_pid2sid(args.pid2sid_uri, rank)

    # ★ product_pid2sid（0724 ver3 新增）：product 域商品段 history + target 用此映射
    product_pid2sid: Optional[Dict[int, Any]] = None
    if args.representation_type == REPRESENTATION_SID and args.product_pid2sid_uri:
        product_pid2sid = load_pid2sid(args.product_pid2sid_uri, rank)

    # ★ bench 模式：加载 bench 并构建 uid→row_idx 映射（与 5_data_format 训练对齐）
    bench_df: Optional[pd.DataFrame] = None
    uid_to_idx: Optional[Dict[int, int]] = None
    if getattr(args, "target_source", "upstream") == "bench":
        if not args.bench_uri:
            raise ValueError("target_source=bench 必须提供 --bench_uri")
        bench_df, uid_to_idx = load_bench(args.bench_uri, rank)

    total_records = 0
    total_samples = 0
    buffer: List[Dict[str, Any]] = []
    part_idx = 0

    frame_iter = iter_my_frames(input_path, rank, world_size)
    for frame in tqdm(frame_iter, desc=f"rank{rank:05d}"):
        total_samples += len(frame)
        for record in iter_records_from_frame(
            frame=frame,
            representation_type=args.representation_type,
            pid2sid=pid2sid,
            product_pid2sid=product_pid2sid,
            bench_df=bench_df,
            uid_to_idx=uid_to_idx,
        ):
            buffer.append(record)
            total_records += 1
            if len(buffer) >= args.rows_per_shard:
                part_idx = write_shards(
                    records=buffer,
                    output_dir=args.output_uri,
                    rank=rank,
                    start_part=part_idx,
                    rows_per_shard=args.rows_per_shard,
                )
                buffer = []

    if buffer:
        part_idx = write_shards(
            records=buffer,
            output_dir=args.output_uri,
            rank=rank,
            start_part=part_idx,
            rows_per_shard=args.rows_per_shard,
        )

    summary = {
        "rank": rank,
        "world_size": world_size,
        "representation_type": args.representation_type,
        "input_uri": args.input_uri,
        "output_uri": args.output_uri,
        "total_input_rows_seen": total_samples,
        "total_output_rows_written": total_records,
        "parts_written": part_idx,
    }
    logger.info("rank=%s 完成: %s", rank, summary)

    if comm is not None:
        gathered = comm.gather(summary, root=0)
        if rank == 0:
            merged = {
                "representation_type": args.representation_type,
                "input_uri": args.input_uri,
                "output_uri": args.output_uri,
                "world_size": world_size,
                "total_input_rows_seen": int(sum(x["total_input_rows_seen"] for x in gathered)),
                "total_output_rows_written": int(sum(x["total_output_rows_written"] for x in gathered)),
                "parts_written": int(sum(x["parts_written"] for x in gathered)),
                "rank_summaries": gathered,
            }
            logger.info("全局汇总: %s", json.dumps(merged, ensure_ascii=False))
            return merged
        return summary

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare unified inputs for PPL evaluation.")
    parser.add_argument(
        "--input_uri",
        type=str,
        required=True,
        help="step2/step4 CoT 输出目录或单个 parquet 文件",
    )
    parser.add_argument(
        "--output_uri",
        type=str,
        required=True,
        help="预处理输出目录",
    )
    parser.add_argument(
        "--representation_type",
        type=str,
        choices=[REPRESENTATION_CAPTION, REPRESENTATION_SID],
        required=True,
        help="caption=使用 target_caption；sid=使用 pid2sid 转 sid token",
    )
    parser.add_argument(
        "--pid2sid_uri",
        type=str,
        default="",
        help="video/ad pid2sid parquet，仅 sid 模式需要",
    )
    parser.add_argument(
        "--product_pid2sid_uri",
        type=str,
        default="",
        help="product_pid2sid parquet 路径（0724 ver3 新增），sid 模式下 2_cot 含 "
             "product 域样本时必填（product 域商品段 history + target 用此映射）",
    )
    parser.add_argument(
        "--target_source",
        type=str,
        choices=["upstream", "bench"],
        default="upstream",
        help="history/target pids 来源（0717 新增，仅 sid 模式有意义）。"
             "'upstream'（默认）= 用 2_cot 上游的 pids（caption-filtered + [:10] 截断，原行为）。"
             "'bench' = 从 bench parquet 重建 pids，与 5_data_format 训练完全对齐"
             "（history 仅长度截断不丢无 caption pid，target 不截断全量覆盖）。"
             "★ bench 模式下 CoT 文本仍来自 2_cot（不重跑 LLM），只是 input/target 的 pid 集合改用 bench。",
    )
    parser.add_argument(
        "--bench_uri",
        type=str,
        default="",
        help="onerec_bench_release.parquet 路径，target_source=bench 时必填",
    )
    parser.add_argument(
        "--rows_per_shard",
        type=int,
        default=1000,
        help="每个输出 parquet 的行数",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更详细日志",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
