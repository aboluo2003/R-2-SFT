#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 (data_format): 合成 RecIF CoT SFT 数据集（MPI 并行）。

任务：
  把 step 2 (2_cot) 或 step 4 (4_filter) 产出的 CoT 样本，与从
  onerec_bench_release.parquet 提取的非 CoT 推荐样本、以及 caption2sid /
  sid2caption 任务样本，拼成统一的 SFT 训练格式，并做全局 shuffle + 分片，
  最后生成 file_list.json。

输入：
  - CoT 目录（step2 / step4）：part-*.parquet，列
      sample_key, uid, target_domain, cot,
      history_pids, history_captions, target_pids, target_captions, history_meta
    target_domain ∈ {video, ad}
  - bench parquet：onerec_bench_release.parquet（非 CoT 推荐样本来源 + split 过滤）
  - caption 目录：pid + dense_caption 列的 parquet 或分片目录
  - pid2sid parquet（video_ad_pid2sid / product_pid2sid）

输出：
  - 输出目录下的分片 parquet：part-{shard_idx:05d}.parquet
  - OpenOneRec/pretrain/examples/blended/<blended_json_name>  file_list.json
    （文件内为输出分片的绝对路径）

训练格式（messages JSON）：
  - cot 样本 assistant：  {cot_body}\n\n{target_sids}
    训练侧 add_think_pattern=True 时会识别  tag 并给 user 加 /think
  - 非 cot 样本 assistant：{target_sids}
    训练侧会自动包空  并给 user 加 /no_think
  - prompt / system message 与 video_rec.py / ad_rec.py / product_rec.py /
    caption_to_sid.py / item_understand.py 对齐

MPI 策略（两阶段全局 shuffle）：
  阶段1（per-rank 处理 + 临时文件）：
    - caption / pid2sid dict 每 rank 全量加载（同节点 page cache 让磁盘只读一次）
    - bench parquet 每 rank 全量读再按 uid 索引 round-robin 分片
    - 各 rank 把构造的样本写 per-rank 临时文件 temp_rank{r:05d}.parquet
  Allgather：交换 (temp_path, row_count) 元数据，每 rank 都知道全局布局
  阶段2（确定性置换 + 并行 shard 写出）：
    - 每 rank 用相同 seed 生成全局置换 perm[0..N_total)
    - 每 rank 拥有连续的一段输出分片 [r*k, (r+1)*k)
    - 通过 perm 反查所需的全局行索引 → 映射到 (temp_file, local_idx)
    - 从共享 FS 读取所需行 → 写出 part-{shard:05d}.parquet

用法：
  # 单进程
  python3 main.py --cot_dir <step2 输出> --bench_parquet <bench> \
      --caption_dir <caption shards> --pid2sid <video_ad pid2sid> \
      --product_pid2sid <product pid2sid> --output_dir <输出> \
      --blended_json_path <file_list.json>

  # MPI 并行
  mpirun -np 16 python3 main.py ... [同上参数]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


# =============================================================================
# 常量：与 video_rec.py / ad_rec.py / product_rec.py 对齐
# =============================================================================
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

VIDEO_HIST_MAX_LEN = 512          # video_rec.py: HIST_MAX_LEN
AD_VIDEO_HIST_MAX_LEN = 100       # ad_rec.py: VIDEO_HIST_MAX_LEN
AD_AD_HIST_MAX_LEN = 200          # ad_rec.py: AD_HIST_MAX_LEN
PRODUCT_VIDEO_HIST_MAX_LEN = 100  # product_rec.py: VIDEO_HIST_MAX_LEN
PRODUCT_HIST_MAX_LEN = 100        # product_rec.py: PRODUCT_HIST_MAX_LEN
# target 截断上限：每条推荐样本最多保留前 3 个 target。
# - split_target=0（不拆分）：每条样本的 target_pids 截断到前 3 个，target_sids 拼接 3 个 sid
# - split_target!=0（拆分）：每条原始样本最多拆成 3 条，每条 1 个 target；
#   原始 target 数 > 3 时用 per-sample 确定性 RNG 随机采样 3 个
TARGET_MAX_LEN = 3

OUTPUT_COLUMNS = ["uuid", "source", "messages", "metadata"]
ROWS_PER_SHARD = 1000

# Reasoning tags recognized by training-side qwen3_dataset.py regex
# r'<think_open>(.*?)<think_close>'. Built via concatenation so the literal
# tags are never mis-parsed by tooling.
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"

# 来源标识
# CoT 样本统一使用 RecIF_RecCOT（不区分 video/ad 域），
# 训练侧 think_loss_sources_name 只需列一个 source 即可让所有 CoT 参与 think loss。
# 非 CoT 推荐样本仍按域区分，便于训练侧按域加权。
SOURCE_REC_COT = "RecIF_RecCOT"
SOURCE_VIDEO_REC = "RecIF_VideoRec"
SOURCE_AD_REC = "RecIF_AdRec"
SOURCE_PRODUCT_REC = "RecIF_ProductRec"
SOURCE_CAPTION_TO_SID = "RecIF_CaptionToSid"
SOURCE_ITEM_UNDERSTAND = "RecIF_ItemUnderstand"


# =============================================================================
# Prompt 模板（直接拷贝自 sft/*.py，保证训练对齐）
# =============================================================================
# ---- video_rec.py ----
VIDEO_SYSTEM_PROMPTS = [
    "你是一个智能推荐助手，能够根据用户的浏览历史预测用户可能感兴趣的下一个内容。",
    "你是一名内容推荐专家，擅长分析用户浏览行为并预测用户偏好。",
    "作为推荐系统助手，你需要根据用户历史浏览记录推荐合适的内容。",
    "你具备理解用户浏览模式并生成个性化推荐的能力。",
    "你是一个专业的内容推荐助手，能够根据用户过往浏览记录推荐相关内容。",
]
VIDEO_USER_PROMPTS = [
    "根据以下用户浏览记录，请预测用户接下来可能观看的内容：\n{query}",
    "用户浏览了以下内容：\n{query}\n请预测用户的下一个观看意向。",
    "以下是用户的浏览历史：\n{query}\n请推荐用户可能感兴趣的下一个内容。",
    "用户历史浏览记录如下：\n{query}\n分析并预测用户接下来会观看什么内容。",
    "{query}\n根据上述浏览记录，推测用户的下一个观看目标。",
]

# ---- ad_rec.py ----
AD_SYSTEM_PROMPTS = [
    "你是一个智能广告推荐助手，能够根据用户的视频观看历史和广告点击行为，预测用户接下来可能点击的广告视频。",
    "你是一个广告点击预测专家，擅长分析用户的观看习惯和广告点击偏好，预测用户的广告兴趣。",
    "你是一个个性化广告推荐系统，能够基于用户的视频观看历史和广告点击记录，预测用户未来可能点击的广告。",
    "你是一个用户行为分析助手，专注于理解用户的内容偏好和广告兴趣，推荐相关广告视频。",
    "你是一个广告推荐引擎，通过学习用户的视频观看和广告点击历史，预测用户对广告的兴趣。",
]
AD_VIDEO_WATCH_PROMPTS = [
    "用户观看过的视频：",
    "用户浏览过的视频内容：",
    "用户长时间观看的视频：",
    "用户感兴趣的视频：",
]
AD_AD_CLICK_PROMPTS = [
    "用户点击过的广告视频：",
    "用户浏览过的广告视频：",
    "用户感兴趣的广告视频：",
    "用户历史广告点击记录：",
]
AD_TASK_PROMPTS = [
    "请根据用户的观看和广告点击历史，预测用户接下来可能点击的广告视频。",
    "基于以上记录，推荐用户可能感兴趣并点击的广告视频。",
    "分析用户的行为偏好，预测用户下一步会点击哪些广告视频。",
    "根据用户的视频观看和广告点击习惯，推荐用户可能点击的广告视频。",
    "请推荐用户接下来可能感兴趣并点击的广告视频。",
]

# ---- product_rec.py ----
PRODUCT_SYSTEM_PROMPTS = [
    "你是一个智能跨域推荐助手，能够根据用户观看的视频内容和历史购物行为，预测用户接下来可能点击的商品。",
    "你是一个跨域推荐专家，擅长分析用户的观看习惯和购物偏好，预测用户的商品兴趣。",
    "你是一个个性化推荐系统，能够基于用户的视频观看历史和购物记录，预测用户未来可能购买的商品。",
    "你是一个用户行为分析助手，专注于理解用户的内容偏好和购物兴趣，推荐相关商品。",
    "你是一个跨域推荐引擎，通过学习用户的视频观看和购物历史，预测用户对商品的兴趣。",
]
PRODUCT_VIDEO_WATCH_PROMPTS = [
    "用户观看过的视频：",
    "用户浏览过的视频内容：",
    "用户长时间观看的视频：",
    "用户感兴趣的视频：",
]
PRODUCT_CLICK_PROMPTS = [
    "用户点击过的商品：",
    "用户浏览过的商品：",
    "用户感兴趣的商品：",
    "用户历史购物记录：",
]
PRODUCT_TASK_PROMPTS = [
    "请根据用户的观看和购物历史，预测用户接下来可能点击的商品。",
    "基于以上记录，推荐用户可能感兴趣并点击的商品。",
    "分析用户的行为偏好，预测用户下一步会点击哪些商品。",
    "根据用户的视频观看和购物习惯，推荐用户可能点击的商品。",
    "请推荐用户接下来可能感兴趣并点击的商品。",
]

# ---- caption_to_sid.py ----
CAPTION2SID_SYSTEM_PROMPTS = [
    "你是一名视频检索助手，需要根据视频描述找出对应的视频token。",
    "你是一个专业的视频内容定位助手，能够根据文字描述匹配正确的视频token。",
    "你擅长根据视频内容描述识别对应的视频token，请直接输出最匹配的视频token。",
    "作为视频内容理解助手，你需要把给定的视频描述映射为对应的视频token。",
    "你是一个视频标识预测专家，请根据描述返回准确的视频token。",
]
CAPTION2SID_USER_PROMPTS = [
    "请根据下面的视频描述，输出对应的视频token：\n{caption}",
    "这段视频的内容描述如下，请给出它对应的视频token：\n{caption}",
    "请阅读以下视频描述，并返回唯一对应的视频token：\n{caption}",
    "下面是一个视频的完整描述，请输出该视频的视频token：\n{caption}",
    "根据这段视频描述，识别并输出对应的视频token：\n{caption}",
    "请将以下视频内容说明转换为对应的视频token：\n{caption}",
]

# ---- item_understand.py ----
SID2CAPTION_SYSTEM_PROMPTS = [
    "你是一名视频描述生成器，请根据下面的视频token生成视频描述。",
    "你是一个专业的视频内容分析助手，能够理解视频token并生成准确的描述。",
    "你是一位视频理解专家，擅长将视频token转换为详细的文字描述。",
    "作为视频内容解析助手，你需要根据视频token提供精准的内容描述。",
    "你是一个智能视频解说员，可以根据视频token创建生动的描述。",
    "你具备理解视频token并生成高质量描述的能力。",
    "你是视频内容描述专家，能够将视频token转化为易懂的文字说明。",
    "作为AI视频分析助手，你可以根据视频token生成详细准确的描述。",
]
SID2CAPTION_USER_PROMPTS = [
    "请描述 {sid} 的内容",
    "这段视频 {sid} 展示了什么？",
    "请解释 {sid} 中的内容",
    "能否说明 {sid} 里发生了什么？",
    "请分析 {sid} 的具体内容",
    "{sid} 这个视频讲的是什么？",
    "请详细描述 {sid}",
    "告诉我 {sid} 的内容是什么",
    "请为 {sid} 生成描述",
    "{sid} 包含哪些内容？",
    "请说明视频 {sid} 的主要内容",
    "描述一下 {sid} 中展现的场景",
    "{sid} 这段内容是关于什么的？",
    "请解读 {sid} 的视频内容",
    "能描述下 {sid} 吗？",
    "{sid} 里面有什么？",
    "请对 {sid} 进行内容说明",
    "这个 {sid} 是什么内容？",
    "分析 {sid} 并给出描述",
    "请阐述 {sid} 的内容细节",
]

# ---- product 域 caption 任务（0728 新增，修复 prompt 与 content 不匹配 bug） ----
# 旧 bug：product sid2caption / caption2sid 复用 video/ad 的"视频"prompt，但
#   assistant 输出的是商品描述（goods_pid2caption）→ prompt-content mismatch，
#   污染 product sid 语义空间，可能是 product 跑不过 uncot-only baseline 的原因之一。
# 修复：product 域用独立的"商品"主题 prompt，与 goods_pid2caption 内容对齐。
PRODUCT_SID2CAPTION_SYSTEM_PROMPTS = [
    "你是一名商品描述生成器，请根据下面的商品token生成商品描述。",
    "你是一个专业的商品内容分析助手，能够理解商品token并生成准确的描述。",
    "你是一位商品理解专家，擅长将商品token转换为详细的文字描述。",
    "作为商品内容解析助手，你需要根据商品token提供精准的商品描述。",
    "你是一个智能商品解说员，可以根据商品token创建生动的描述。",
    "你具备理解商品token并生成高质量描述的能力。",
    "你是商品内容描述专家，能够将商品token转化为易懂的文字说明。",
    "作为AI商品分析助手，你可以根据商品token生成详细准确的描述。",
]
PRODUCT_SID2CAPTION_USER_PROMPTS = [
    "请描述 {sid} 对应的商品",
    "这个商品 {sid} 是什么？",
    "请解释 {sid} 对应的商品信息",
    "能否说明 {sid} 这个商品的情况？",
    "请分析 {sid} 对应商品的具体信息",
    "{sid} 这个商品是什么？",
    "请详细描述 {sid} 对应的商品",
    "告诉我 {sid} 对应商品的信息",
    "请为 {sid} 生成商品描述",
    "{sid} 对应的商品包含哪些信息？",
    "请说明商品 {sid} 的主要内容",
    "描述一下 {sid} 对应的商品",
    "{sid} 这个商品是关于什么的？",
    "请解读 {sid} 的商品信息",
    "能描述下 {sid} 对应的商品吗？",
    "{sid} 这个商品有什么特点？",
    "请对 {sid} 进行商品信息说明",
    "这个 {sid} 对应什么商品？",
    "分析 {sid} 并给出商品描述",
    "请阐述 {sid} 对应商品的细节",
]
PRODUCT_CAPTION2SID_SYSTEM_PROMPTS = [
    "你是一名商品检索助手，需要根据商品描述找出对应的商品token。",
    "你是一个专业的商品定位助手，能够根据文字描述匹配正确的商品token。",
    "你擅长根据商品描述识别对应的商品token，请直接输出最匹配的商品token。",
    "作为商品内容理解助手，你需要把给定的商品描述映射为对应的商品token。",
    "你是一个商品标识预测专家，请根据描述返回准确的商品token。",
]
PRODUCT_CAPTION2SID_USER_PROMPTS = [
    "请根据下面的商品描述，输出对应的商品token：\n{caption}",
    "这个商品的内容描述如下，请给出它对应的商品token：\n{caption}",
    "请阅读以下商品描述，并返回唯一对应的商品token：\n{caption}",
    "下面是一个商品的完整描述，请输出该商品的商品token：\n{caption}",
    "根据这段商品描述，识别并输出对应的商品token：\n{caption}",
    "请将以下商品内容说明转换为对应的商品token：\n{caption}",
]


# =============================================================================
# siddoc mode: 把用户历史序列从 user message 挪到 assistant 输出，
# 让 only_assistant_loss=true 时对历史序列也算 CE loss（对用户序列做自回归）。
#
# 新设计（0713 起）：
#   - siddoc 样本的 assistant 只放「带分段标签的历史序列」，不放 target、不放
#     prediction prompt。避免 ad 域里「ad 历史 + ad target」拼在一起造成 sid
#     边界混淆（ad target 紧跟 ad 历史，模型分不清历史结束 / target 开始）。
#   - 各域 history 在 assistant 中保留与非 CoT 一致的分段标签：
#       video  : "用户观看过的视频：<sids>"
#       ad     : "用户观看过的视频：<video_sids>\n用户点击过的广告视频：<ad_sids>"
#       product: "用户观看过的视频：<video_sids>\n用户点击过的商品：<product_sids>"
#   - target 预测仍由同用户的原格式 rec 样本承担（SIDD0C_MODE=2 mix 模式下
#     每个 user 每域同时产出一条 rec + 一条 siddoc）。
#
# SIDD0C_MODE 语义（仅作用于非 CoT rec 样本，caption 任务不受影响）：
#   0 = 原格式（rec 样本，assistant = target）
#   1 = 纯 siddoc 替换（rec 样本全部变成 siddoc，assistant = 历史标签序列）
#   2 = mix：每个用户每域同时造一条原格式 rec + 一条 siddoc，混在一起训练
# =============================================================================
SIDD0C_TASK_INSTRUCTION = (
    "请输出该用户在广告、直播、视频、电商场景中的行为历史。"
)


# =============================================================================
# 通用工具
# =============================================================================
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        force=True,
    )


def chunk_for_rank(items: List[Any], rank: int, world_size: int) -> List[Any]:
    """round-robin 分片，与 reason/mpi / 1_history_text 保持一致。"""
    return [item for idx, item in enumerate(items) if idx % world_size == rank]


def to_int_list(value: Any) -> List[int]:
    """安全转 list[int]，兼容 float64 / NaN / ndarray。"""
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


def normalize_caption(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def pids_to_sids(pids: Sequence[int], pid2sid: Dict[int, Any]) -> str:
    """把 list[pid] 转成 SID 拼接字符串，与 video_rec.py 的逻辑一致。"""
    if pids is None:
        return ""
    sids: List[str] = []
    for pid in pids:
        if pid in pid2sid:
            code = pid2sid[pid]
            try:
                sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            except (IndexError, TypeError, KeyError):
                continue
            sids.append(sid)
    return "".join(sids)


def pids_to_sids_with_count(
    pids: Sequence[int], pid2sid: Dict[int, Any]
) -> Tuple[str, int]:
    """同 pids_to_sids，但额外返回命中的 sid 个数（用于 metadata 标记分段位置）。"""
    if pids is None:
        return "", 0
    sids: List[str] = []
    for pid in pids:
        if pid in pid2sid:
            code = pid2sid[pid]
            try:
                sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            except (IndexError, TypeError, KeyError):
                continue
            sids.append(sid)
    return "".join(sids), len(sids)


def pid_to_sid_str(pid: int, pid2sid: Dict[int, Any]) -> str:
    if pid not in pid2sid:
        return ""
    code = pid2sid[pid]
    try:
        return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
    except (IndexError, TypeError, KeyError):
        return ""


def build_messages_json(
    system_prompt: str,
    user_prompt: str,
    assistant_text: str,
) -> str:
    """构造与 sft/*.py 一致的 messages JSON 字符串。"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    return json.dumps(messages, ensure_ascii=False)


def build_cot_answer(cot_text: str, target_sids: str) -> str:
    """Construct the assistant text for a CoT sample.

    Format (aligned with qwen3_dataset.py add_think_pattern):
        THINK_OPEN + "\n" + cot_body + THINK_CLOSE + "\n\n" + target_sids

    where THINK_OPEN / THINK_CLOSE are the reasoning tags recognized by the
    training-side regex. Training adds ``/think`` to the user message when the
    reasoning body is non-empty.
    """
    cot_body = safe_str(cot_text)
    if cot_body and not cot_body.endswith("\n"):
        cot_body += "\n"
    return THINK_OPEN + "\n" + cot_body + THINK_CLOSE + "\n\n" + target_sids


# =============================================================================
# 加载 pid2sid 映射
# =============================================================================
def load_pid2sid(path: str, rank: int, label: str) -> Dict[int, Any]:
    t0 = time.time()
    logger.info("rank=%s 加载 %s: %s", rank, label, path)
    df = pd.read_parquet(path, columns=["pid", "sid"])
    pid2sid = dict(zip(df["pid"].tolist(), df["sid"].tolist()))
    logger.info(
        "rank=%s 加载 %s 完成: entries=%s 耗时=%.1fs",
        rank, label, len(pid2sid), time.time() - t0,
    )
    return pid2sid


# =============================================================================
# 加载 CoT 样本（step2 / step4）
# =============================================================================
PART_FILE_RE = re.compile(r"^part-.*\.parquet$")


def list_parquet_shards(path: Path) -> List[Path]:
    """列出目录下的 part-*.parquet，或单个文件。"""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and PART_FILE_RE.match(p.name))
        if files:
            return files
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"未在 {path} 找到 parquet 分片")


def load_uid_split_map(bench_parquet: str, rank: int) -> Dict[int, int]:
    """从 bench parquet 读取 uid + split 列，构造 {uid: split} 映射。

    CoT 样本本身不含 split 字段（step1 1_history_text 不按 split 过滤），
    通过 uid join bench 的 split 列，可以过滤掉 split!=0 的 CoT 样本，
    与非 CoT 推荐样本（video_rec.py 等只处理 split=0）保持一致。

    只读 uid + split 两列，开销很小（~10MB）。
    """
    t0 = time.time()
    logger.info("rank=%s 加载 bench uid+split: %s", rank, bench_parquet)
    try:
        table = pq.read_table(bench_parquet, columns=["uid", "split"])
    except Exception as exc:
        logger.warning(
            "rank=%s 读取 bench uid+split 失败: %s，CoT 将不做 split 过滤", rank, exc,
        )
        return {}
    df = table.to_pandas()
    uid_to_split = dict(zip(df["uid"].astype(int).tolist(), df["split"].astype(int).tolist()))
    logger.info(
        "rank=%s 加载 bench uid+split 完成: entries=%s 耗时=%.1fs",
        rank, len(uid_to_split), time.time() - t0,
    )
    return uid_to_split


def load_bench_df(
    bench_parquet: str,
    rank: int,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """加载完整 bench parquet，构建 {uid: row_idx} 与 {uid: split} 映射。

    用于：
      1. CoT 样本按 uid 查找原始 history / target pid（修复 CoT 与非 CoT 历史不一致）
      2. CoT 按 split 过滤（替代单独的 load_uid_split_map 调用）
      3. 传给 build_non_cot_rec_records 复用，避免重复加载

    返回:
      bench_df: 完整 bench DataFrame
      uid_to_idx: {uid: row_index}，用 bench_df.iloc[idx] 取行
      uid_to_split: {uid: split}，split 列缺失时为空 dict
    """
    t0 = time.time()
    logger.info("rank=%s 加载 bench (full): %s", rank, bench_parquet)
    bench_df = pq.read_table(bench_parquet).to_pandas()
    logger.info(
        "rank=%s 加载 bench 完成: rows=%s cols=%s 耗时=%.1fs",
        rank, len(bench_df), len(bench_df.columns), time.time() - t0,
    )

    uid_list = bench_df["uid"].astype(int).tolist()
    uid_to_idx = {uid: idx for idx, uid in enumerate(uid_list)}

    uid_to_split: Dict[int, int] = {}
    if "split" in bench_df.columns:
        split_list = bench_df["split"].astype(int).tolist()
        uid_to_split = dict(zip(uid_list, split_list))
    else:
        logger.warning("rank=%s bench 无 split 列，CoT 将不做 split 过滤", rank)

    return bench_df, uid_to_idx, uid_to_split


def load_cot_samples(
    cot_dir: Path,
    rank: int,
    world_size: int,
    uid_to_split: Optional[Dict[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], Set[Tuple[int, str]]]:
    """读取 step2/step4 输出，按文件 round-robin 分给各 rank。

    Args:
      uid_to_split: 可选的 {uid: split} 映射。若提供，跳过 split!=0 的 CoT 样本
                    （通过 uid join bench 的 split 列）。

    返回:
      samples: list[dict]，每条 dict 包含 uid/target_domain/cot/history_pids/
               target_pids/history_meta/filter_meta 等字段。
               filter_meta 为 4a_filter_ppl 写入的 per-target 标记 JSON
               （annotate_all 模式下每个 sample 都有；threshold 模式下只有
               survived sample 有）。若文件无此列则为 None。
      cot_user_keys: 全局 {(uid, target_domain)} 集合（用于非 cot 重叠判断）
    """
    t0 = time.time()
    try:
        files = list_parquet_shards(cot_dir)
    except FileNotFoundError:
        # 空目录视为"无 CoT"（用于 siddoc 等无 CoT 消融实验：gen_siddoc.sh 把
        # COT_DIR 指向 _empty_cot）。返回空结果，下游 cot_records=[] 且
        # cot_user_keys=set()，配合 ALLOW_COT_NON_COT_USER_OVERLAP=1 时
        # 非 CoT 样本使用全部用户 —— 与 ver0 中"CoT 为空时 overlap=0/1 等价"一致。
        logger.info("rank=%s CoT 目录为空，跳过 CoT 加载: %s", rank, cot_dir)
        return [], set()
    logger.info("rank=%s 找到 CoT 分片 %s 个: %s", rank, len(files), cot_dir)
    my_files = chunk_for_rank(files, rank, world_size)
    logger.info("rank=%s 分配 CoT 分片=%s / 总分片=%s", rank, len(my_files), len(files))

    local_samples: List[Dict[str, Any]] = []
    local_user_keys: Set[Tuple[int, str]] = set()
    skipped_split = 0
    need_cols = [
        "sample_key", "uid", "target_domain", "cot",
        "history_pids", "history_captions", "target_pids", "target_captions",
        "history_meta",
    ]
    for fpath in tqdm(my_files, desc=f"rank{rank} load cot", disable=rank != 0):
        try:
            table = pq.read_table(fpath)
        except Exception as exc:
            logger.warning("rank=%s 读取 %s 失败: %s", rank, fpath, exc)
            continue
        # 只保留存在的列
        available = set(table.schema.names)
        cols = [c for c in need_cols if c in available]
        df = table.select(cols).to_pandas()
        for _, row in df.iterrows():
            uid = int(row["uid"])
            target_domain = safe_str(row.get("target_domain"))
            if target_domain not in ("video", "ad", "product"):
                continue
            # 通过 uid join bench 的 split 列过滤
            if uid_to_split is not None and uid_to_split.get(uid, 0) != 0:
                skipped_split += 1
                continue
            hist_caps = row.get("history_captions")
            if hist_caps is None or (isinstance(hist_caps, float) and pd.isna(hist_caps)):
                hist_caps = []
            tgt_caps = row.get("target_captions")
            if tgt_caps is None or (isinstance(tgt_caps, float) and pd.isna(tgt_caps)):
                tgt_caps = []
            sample = {
                "sample_key": safe_str(row.get("sample_key")) or f"{uid}_{target_domain}",
                "uid": uid,
                "target_domain": target_domain,
                "cot": safe_str(row.get("cot")),
                "history_pids": to_int_list(row.get("history_pids")),
                "history_captions": [safe_str(c) for c in hist_caps],
                "target_pids": to_int_list(row.get("target_pids")),
                "target_captions": [safe_str(c) for c in tgt_caps],
                "history_meta": safe_str(row.get("history_meta")),
            }
            local_samples.append(sample)
            local_user_keys.add((uid, target_domain))

    logger.info(
        "rank=%s 加载 CoT 完成: samples=%s user_keys=%s skipped_split=%s 耗时=%.1fs",
        rank, len(local_samples), len(local_user_keys), skipped_split, time.time() - t0,
    )
    return local_samples, local_user_keys


def parse_history_meta(meta_str: str) -> Dict[str, Any]:
    """解析 history_meta JSON，失败返回空 dict。"""
    if not meta_str:
        return {}
    try:
        obj = json.loads(meta_str)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _stable_seed(sample_key: str, uid: int, target_domain: str, base_seed: int) -> int:
    """根据 sample_key / (uid, domain) 生成稳定的 per-sample seed。

    保证同一条样本无论被哪个 rank 处理、无论 world_size 多少，拆分时的随机
    采样结果都一致 —— 便于复现和跨配置对比。取 md5 前 64 bit 作为整数 seed。
    """
    key = f"{base_seed}:{sample_key or f'{uid}_{target_domain}'}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def compute_target_subsets(
    target_pids: List[int],
    split_target: int,
    max_len: int,
    rng: random.Random,
) -> List[List[int]]:
    """计算 target 拆分子集列表。

    - ``split_target == 0``（不拆分）：返回 ``[target_pids[:max_len]]``，
      即单条样本保留前 ``max_len`` 个 target（与原截断行为一致）。
    - ``split_target != 0``（拆分）：返回 ``[[p] for p in sampled]``，
      每个子集是单个 target，最多 ``max_len`` 个：
        - ``len(target_pids) > max_len`` 时用 ``rng`` 随机采样 ``max_len`` 个
          （per-sample 确定性 RNG，复现稳定）
        - 否则用全部 ``target_pids``

    Returns:
      list[list[int]]，每个子集是一条样本的 target_pids。空 target 返回 []。
    """
    if not target_pids:
        return []
    if split_target == 0:
        return [target_pids[:max_len]]
    if len(target_pids) > max_len:
        sampled = rng.sample(target_pids, max_len)
    else:
        sampled = list(target_pids)
    return [[p] for p in sampled]


# =============================================================================
# 从 bench row 提取原始历史 / target pid（与非 CoT 脚本对齐）
# =============================================================================
def get_orig_history_pids(
    bench_row: Optional[pd.Series],
    target_domain: str,
) -> Optional[Dict[str, List[int]]]:
    """从 bench row 读取原始（未经 caption 过滤）的 history pid。

    返回 dict:
      video 域: {"hist_pids": list[int]}  ← hist_video_pid[-VIDEO_HIST_MAX_LEN:]
      ad    域: {"video_pids": list[int], "ad_pids": list[int]}
                ← hist_longview_video_list[-AD_VIDEO_HIST_MAX_LEN:]
                  + hist_ad_pid[-AD_AD_HIST_MAX_LEN:]

    bench_row 为 None 时返回 None（调用方需退化为 caption-filtered 路径）。
    """
    if bench_row is None:
        return None
    if target_domain == "video":
        hist_pids = to_int_list(bench_row.get("hist_video_pid"))
        hist_pids = hist_pids[-VIDEO_HIST_MAX_LEN:]
        return {"hist_pids": hist_pids}
    elif target_domain == "ad":
        hist_longview = to_int_list(bench_row.get("hist_longview_video_list"))
        hist_ad_pids = to_int_list(bench_row.get("hist_ad_pid"))
        hist_longview = hist_longview[-AD_VIDEO_HIST_MAX_LEN:]
        hist_ad_pids = hist_ad_pids[-AD_AD_HIST_MAX_LEN:]
        return {"video_pids": hist_longview, "ad_pids": hist_ad_pids}
    elif target_domain == "product":
        hist_longview = to_int_list(bench_row.get("hist_longview_video_list"))
        hist_goods_pids = to_int_list(bench_row.get("hist_goods_pid"))
        hist_longview = hist_longview[-PRODUCT_VIDEO_HIST_MAX_LEN:]
        hist_goods_pids = hist_goods_pids[-PRODUCT_HIST_MAX_LEN:]
        return {"video_pids": hist_longview, "product_pids": hist_goods_pids}
    return None


def get_orig_target_pids(
    bench_row: Optional[pd.Series],
    target_domain: str,
    fallback: List[int],
) -> List[int]:
    """从 bench row 读取原始 target pid（与非 CoT 脚本对齐）。

    bench_row 为 None 或字段缺失时退化为 fallback（caption-filtered target）。
    """
    if bench_row is None:
        return fallback
    if target_domain == "video":
        return to_int_list(bench_row.get("target_video_pid")) or fallback
    elif target_domain == "ad":
        return to_int_list(bench_row.get("target_ad_pid")) or fallback
    elif target_domain == "product":
        return to_int_list(bench_row.get("target_goods_pid")) or fallback
    return fallback


def build_cot_record(
    sample: Dict[str, Any],
    pid2sid: Dict[int, Any],
    product_pid2sid: Optional[Dict[int, Any]] = None,
    bench_row: Optional[pd.Series] = None,
    siddoc_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """把一条 CoT 样本构造成训练格式。

    历史序列对齐策略（修复 bug：CoT 与非 CoT 历史不一致）：
      - 若传入 bench_row（推荐）：从 bench row 读取**原始** history / target pid
        （未经 caption 过滤），再做与 video_rec.py / ad_rec.py 一致的截断 +
        pid2sid 过滤。这样 CoT 样本的 input 与非 CoT 样本完全一致。
      - 若 bench_row 为 None（退化）：使用 step1/step2 输出的 caption-filtered
        history_pids（旧逻辑，仅用于无 bench 时的兼容路径）。

    ad 域两段式 history：
      - input 中显式分为「用户观看过的视频」+「用户点击过的广告视频」两段
      - metadata 中记录 video_hist_sid_count / ad_hist_sid_count，标明分段位置

    siddoc_mode=True 时：
      - user message = SIDD0C_TASK_INSTRUCTION（任务指令）
      - assistant = history_part + prediction_prompt + answer
      - 历史序列挪到 assistant，让 only_assistant_loss=true 时对历史也算 CE loss

    ★ sid 映射（无 fallback，必须正确）：
      - video / ad 域：用 pid2sid（video/ad pid 空间）
      - product 域：target + goods history 段必须用 product_pid2sid（电商 pid 空间）。
        若 product_pid2sid 为 None 或空 dict，直接抛 ValueError —— 不再静默 fallback
        到 pid2sid（旧 bug：product pid 与 video/ad pid 数值重叠但 sid 编码不同，
        fallback 会让 product 样本拿到 video/ad 的错误 sid，样本数/token 数不变但
        sid 标签 100% 错，只有 eval 才能暴露，排查极困难）。
    """
    target_domain = sample["target_domain"]
    history_pids_caption_filtered = sample["history_pids"]
    target_pids = sample["target_pids"]
    cot_text = sample["cot"]

    if not target_pids:
        return None
    # ★ 严格 sid 映射：product 域必须用 product_pid2sid，无 fallback。
    if target_domain == "product":
        if not product_pid2sid:
            raise ValueError(
                "product 域 CoT 样本要求 product_pid2sid 非空（电商 sid 映射），"
                "当前为 None/{}。请检查 run_mpi.sh 是否传入 --product_pid2sid，"
                "以及 run() 是否在 CoT 含 product 域时加载了该映射。"
                "（旧 bug：product_pid2sid 为空时 fallback 到 video/ad pid2sid，"
                "因 pid 数值重叠会得到错误 sid 但样本不被丢弃，极难发现。）"
            )
        target_sid_map = product_pid2sid
    else:
        target_sid_map = pid2sid
    target_sids = pids_to_sids(target_pids[:TARGET_MAX_LEN], target_sid_map)
    if not target_sids:
        return None

    meta = parse_history_meta(sample.get("history_meta"))
    field = safe_str(meta.get("field"))

    answer = build_cot_answer(cot_text, target_sids)
    history_part = ""

    # 分段统计（写入 metadata，标明 video / ad 段的分界位置）
    video_hist_sid_count = 0
    ad_hist_sid_count = 0
    history_source = "bench_original" if bench_row is not None else "caption_filtered"

    if target_domain == "video":
        # 优先从 bench row 读取原始 history（与非 CoT 对齐）
        orig = get_orig_history_pids(bench_row, "video")
        if orig is not None:
            hist_pids = orig["hist_pids"]
        else:
            hist_pids = history_pids_caption_filtered[-VIDEO_HIST_MAX_LEN:]
        hist_sids, video_hist_sid_count = pids_to_sids_with_count(hist_pids, pid2sid)
        if not hist_sids:
            return None
        system_prompt = random.choice(VIDEO_SYSTEM_PROMPTS)
        if siddoc_mode:
            history_part = random.choice(AD_VIDEO_WATCH_PROMPTS) + hist_sids
            user_prompt = SIDD0C_TASK_INSTRUCTION
        else:
            user_prompt = random.choice(VIDEO_USER_PROMPTS).format(query=hist_sids)
        source = SOURCE_REC_COT
    elif target_domain == "ad":
        # ad_rec.py：两段式 prompt（用户观看视频 + 用户点击广告）
        # 优先从 bench row 读取原始 history（与非 CoT 对齐）
        orig = get_orig_history_pids(bench_row, "ad")
        if orig is not None:
            video_pids = orig["video_pids"]
            ad_pids = orig["ad_pids"]
        else:
            # 退化：从 caption-filtered history_pids 按 meta 长度切分（旧逻辑）
            meta_video_len = int(meta.get("kept_video_hist_len", 0) or 0)
            meta_ad_len = int(meta.get("kept_ad_hist_len", 0) or 0)
            if field and "hist_longview_video_list+hist_ad_pid" in field and (meta_video_len or meta_ad_len):
                video_pids = history_pids_caption_filtered[:meta_video_len]
                ad_pids = history_pids_caption_filtered[meta_video_len:meta_video_len + meta_ad_len]
            else:
                video_pids = []
                ad_pids = history_pids_caption_filtered
            # step1 已截断过，这里再截断是 no-op，但保持与旧逻辑 / 非 CoT 路径一致
            video_pids = list(video_pids)[-AD_VIDEO_HIST_MAX_LEN:]
            ad_pids = list(ad_pids)[-AD_AD_HIST_MAX_LEN:]

        user_parts: List[str] = []
        if video_pids:
            video_sids, video_hist_sid_count = pids_to_sids_with_count(video_pids, pid2sid)
            if video_sids:
                user_parts.append(f"{random.choice(AD_VIDEO_WATCH_PROMPTS)}{video_sids}")
        if ad_pids:
            ad_sids, ad_hist_sid_count = pids_to_sids_with_count(ad_pids, pid2sid)
            if ad_sids:
                user_parts.append(f"{random.choice(AD_AD_CLICK_PROMPTS)}{ad_sids}")
        if not user_parts:
            return None

        user_content = "\n".join(user_parts)
        system_prompt = random.choice(AD_SYSTEM_PROMPTS)
        if siddoc_mode:
            history_part = user_content
            user_prompt = SIDD0C_TASK_INSTRUCTION
        else:
            task_prompt = random.choice(AD_TASK_PROMPTS)
            user_prompt = user_content + "\n" + task_prompt
        source = SOURCE_REC_COT
    elif target_domain == "product":
        # product_rec.py：两段式 prompt（用户观看视频 + 用户点击商品）
        # 优先从 bench row 读取原始 history（与非 CoT 对齐）
        orig = get_orig_history_pids(bench_row, "product")
        if orig is not None:
            video_pids = orig["video_pids"]
            goods_pids = orig["product_pids"]
        else:
            # 退化：从 caption-filtered history_pids 按 meta 长度切分
            meta_video_len = int(meta.get("kept_video_hist_len", 0) or 0)
            meta_product_len = int(meta.get("kept_product_hist_len", 0) or 0)
            if field and "hist_longview_video_list+hist_goods_pid" in field and (meta_video_len or meta_product_len):
                video_pids = history_pids_caption_filtered[:meta_video_len]
                goods_pids = history_pids_caption_filtered[meta_video_len:meta_video_len + meta_product_len]
            else:
                video_pids = []
                goods_pids = history_pids_caption_filtered
            video_pids = list(video_pids)[-PRODUCT_VIDEO_HIST_MAX_LEN:]
            goods_pids = list(goods_pids)[-PRODUCT_HIST_MAX_LEN:]

        user_parts: List[str] = []
        if video_pids:
            # 视频段用 pid2sid（video/ad sid），与 build_non_cot_product_record 一致
            video_sids, video_hist_sid_count = pids_to_sids_with_count(video_pids, pid2sid)
            if video_sids:
                user_parts.append(f"{random.choice(PRODUCT_VIDEO_WATCH_PROMPTS)}{video_sids}")
        if goods_pids:
            # ★ 商品段必须用 product_pid2sid（电商 sid），无 fallback。
            #   旧 bug：`product_pid2sid or {}` 在 product_pid2sid 为 None/{} 时
            #   退化为 {}，goods pid 全查不到 → goods_sids="" → 样本丢弃；
            #   或更糟的 fallback 到 video/ad sid（pid 数值重叠）→ 错误 sid。
            #   现在直接抛错，让缺失立刻暴露。
            if not product_pid2sid:
                raise ValueError(
                    "product 域 CoT 样本的 goods history 段要求 product_pid2sid 非空，"
                    "当前为 None/{}。请检查 --product_pid2sid 是否传入并加载。"
                )
            goods_sids, ad_hist_sid_count = pids_to_sids_with_count(
                goods_pids, product_pid2sid)
            if goods_sids:
                user_parts.append(f"{random.choice(PRODUCT_CLICK_PROMPTS)}{goods_sids}")
        if not user_parts:
            return None

        user_content = "\n".join(user_parts)
        system_prompt = random.choice(PRODUCT_SYSTEM_PROMPTS)
        if siddoc_mode:
            history_part = user_content
            user_prompt = SIDD0C_TASK_INSTRUCTION
        else:
            task_prompt = random.choice(PRODUCT_TASK_PROMPTS)
            user_prompt = user_content + "\n" + task_prompt
        source = SOURCE_REC_COT
    else:
        return None

    # 组装 assistant_text
    if siddoc_mode:
        # CoT siddoc：历史标签 + CoT answer（历史挪到 assistant 算 CE loss）。
        # 不放 prediction prompt（已移除），历史段与 CoT answer 之间用空行分隔。
        # 注意：0713 siddoc 实验无 CoT（COT_DIR 空目录），此分支仅在 siddoc_mode==1
        # 且有 CoT 时触发；siddoc_mode==2（mix）只作用于非 CoT rec 样本。
        assistant_text = (history_part + "\n\n" + answer) if history_part else answer
    else:
        assistant_text = answer

    metadata = {
        "task_type": "cot_rec",
        "uid": int(sample["uid"]),
        "target_domain": target_domain,
        "sample_key": sample.get("sample_key"),
        "target_pids": target_pids[:TARGET_MAX_LEN],
        "history_len": video_hist_sid_count + ad_hist_sid_count,
        # ad 域两段式 history 的分段标记：input 中前 video_hist_sid_count 个 sid
        # 属于「用户观看过的视频」，后续 ad_hist_sid_count 个属于「用户点击过的广告视频」。
        # video 域 history 全为视频，ad_hist_sid_count=0。
        "video_hist_sid_count": video_hist_sid_count,
        "ad_hist_sid_count": ad_hist_sid_count,
        # 标明 history 来源，便于排查对齐问题
        "history_source": history_source,
    }
    return {
        "uuid": str(uuid.uuid4()),
        "source": source,
        "messages": build_messages_json(system_prompt, user_prompt, assistant_text),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


# =============================================================================
# 从 bench 构造非 CoT 推荐样本（video / ad / product）
# =============================================================================
def build_non_cot_video_record(
    row,
    pid2sid: Dict[int, Any],
    target_pids_override: Optional[List[int]] = None,
    siddoc_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """对齐 video_rec.py 的 process_row。

    Args:
      target_pids_override: 拆分模式下传入的单 target 子集（[pid]）。
        None 时从 row 读完整 target_video_pid 再截断到 TARGET_MAX_LEN。
      siddoc_mode: True 时历史序列挪到 assistant 输出。
    """
    hist_pids = to_int_list(row.get("hist_video_pid"))
    if target_pids_override is not None:
        target_pids = list(target_pids_override)
    else:
        target_pids = to_int_list(row.get("target_video_pid"))
    hist_pids = hist_pids[-VIDEO_HIST_MAX_LEN:]
    target_pids = target_pids[:TARGET_MAX_LEN]

    query, video_hist_sid_count = pids_to_sids_with_count(hist_pids, pid2sid)
    answer = pids_to_sids(target_pids, pid2sid)
    if not query or not answer:
        return None

    system_prompt = random.choice(VIDEO_SYSTEM_PROMPTS)
    if siddoc_mode:
        # siddoc：assistant 只放带标签的历史（video 域单段），不放 target / prediction
        # prompt。target 预测由同用户的原格式 rec 样本承担（mix 模式）。
        history_part = random.choice(AD_VIDEO_WATCH_PROMPTS) + query
        user_prompt = SIDD0C_TASK_INSTRUCTION
        assistant_text = history_part
    else:
        user_prompt = random.choice(VIDEO_USER_PROMPTS).format(query=query)
        assistant_text = answer
    metadata = {
        "task_type": "siddoc" if siddoc_mode else "rec",
        "uid": int(row["uid"]),
        "target_domain": "video",
        "history_len": video_hist_sid_count,
        "video_hist_sid_count": video_hist_sid_count,
        "ad_hist_sid_count": 0,
        "history_source": "bench_original",
    }
    return {
        "uuid": str(uuid.uuid4()),
        "source": SOURCE_VIDEO_REC,
        "messages": build_messages_json(system_prompt, user_prompt, assistant_text),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def build_non_cot_ad_record(
    row,
    pid2sid: Dict[int, Any],
    target_pids_override: Optional[List[int]] = None,
    siddoc_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """对齐 ad_rec.py 的 process_row。

    Args:
      target_pids_override: 拆分模式下传入的单 target 子集（[pid]）。
        None 时从 row 读完整 target_ad_pid 再截断到 TARGET_MAX_LEN。
      siddoc_mode: True 时历史序列挪到 assistant 输出。
    """
    hist_longview = to_int_list(row.get("hist_longview_video_list"))
    hist_ad_pids = to_int_list(row.get("hist_ad_pid"))
    if target_pids_override is not None:
        target_ad_pids = list(target_pids_override)
    else:
        target_ad_pids = to_int_list(row.get("target_ad_pid"))

    if not target_ad_pids:
        return None

    user_parts: List[str] = []
    video_hist_sid_count = 0
    ad_hist_sid_count = 0
    if hist_longview:
        video_sids, video_hist_sid_count = pids_to_sids_with_count(
            hist_longview[-AD_VIDEO_HIST_MAX_LEN:], pid2sid)
        if video_sids:
            user_parts.append(f"{random.choice(AD_VIDEO_WATCH_PROMPTS)}{video_sids}")
    if hist_ad_pids:
        ad_sids, ad_hist_sid_count = pids_to_sids_with_count(
            hist_ad_pids[-AD_AD_HIST_MAX_LEN:], pid2sid)
        if ad_sids:
            user_parts.append(f"{random.choice(AD_AD_CLICK_PROMPTS)}{ad_sids}")
    if not user_parts:
        return None

    answer = pids_to_sids(target_ad_pids[:TARGET_MAX_LEN], pid2sid)
    if not answer:
        return None

    user_content = "\n".join(user_parts)
    system_prompt = random.choice(AD_SYSTEM_PROMPTS)
    if siddoc_mode:
        # siddoc：assistant 只放两段式历史（视频观看 + 广告点击，各带标签），
        # 不放 target / prediction prompt。避免 ad target 紧跟 ad 历史造成 sid
        # 边界混淆（ad target 与 ad 历史同域 sid，模型分不清）。
        user_prompt = SIDD0C_TASK_INSTRUCTION
        assistant_text = user_content
    else:
        task_prompt = random.choice(AD_TASK_PROMPTS)
        user_prompt = user_content + "\n" + task_prompt
        assistant_text = answer
    metadata = {
        "task_type": "siddoc" if siddoc_mode else "rec",
        "uid": int(row["uid"]),
        "target_domain": "ad",
        "history_len": video_hist_sid_count + ad_hist_sid_count,
        "video_hist_sid_count": video_hist_sid_count,
        "ad_hist_sid_count": ad_hist_sid_count,
        "history_source": "bench_original",
    }
    return {
        "uuid": str(uuid.uuid4()),
        "source": SOURCE_AD_REC,
        "messages": build_messages_json(system_prompt, user_prompt, assistant_text),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def build_non_cot_product_record(
    row,
    video_pid2sid: Dict[int, Any],
    product_pid2sid: Dict[int, Any],
    target_pids_override: Optional[List[int]] = None,
    siddoc_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """对齐 product_rec.py 的 process_row。

    Args:
      target_pids_override: 拆分模式下传入的单 target 子集（[pid]）。
        None 时从 row 读完整 target_goods_pid 再截断到 TARGET_MAX_LEN。
      siddoc_mode: True 时历史序列挪到 assistant 输出。
    """
    hist_longview = to_int_list(row.get("hist_longview_video_list"))
    hist_product_pids = to_int_list(row.get("hist_goods_pid"))
    if target_pids_override is not None:
        target_product_pids = list(target_pids_override)
    else:
        target_product_pids = to_int_list(row.get("target_goods_pid"))

    if not target_product_pids:
        return None

    user_parts: List[str] = []
    video_hist_sid_count = 0
    product_hist_sid_count = 0
    if hist_longview:
        video_sids, video_hist_sid_count = pids_to_sids_with_count(
            hist_longview[-PRODUCT_VIDEO_HIST_MAX_LEN:], video_pid2sid)
        if video_sids:
            user_parts.append(f"{random.choice(PRODUCT_VIDEO_WATCH_PROMPTS)}{video_sids}")
    if hist_product_pids:
        product_sids, product_hist_sid_count = pids_to_sids_with_count(
            hist_product_pids[-PRODUCT_HIST_MAX_LEN:], product_pid2sid)
        if product_sids:
            user_parts.append(f"{random.choice(PRODUCT_CLICK_PROMPTS)}{product_sids}")
    if not user_parts:
        return None

    answer = pids_to_sids(target_product_pids[:TARGET_MAX_LEN], product_pid2sid)
    if not answer:
        return None

    user_content = "\n".join(user_parts)
    system_prompt = random.choice(PRODUCT_SYSTEM_PROMPTS)
    if siddoc_mode:
        # siddoc：assistant 只放两段式历史（视频观看 + 商品点击，各带标签），
        # 不放 target / prediction prompt。
        user_prompt = SIDD0C_TASK_INSTRUCTION
        assistant_text = user_content
    else:
        task_prompt = random.choice(PRODUCT_TASK_PROMPTS)
        user_prompt = user_content + "\n" + task_prompt
        assistant_text = answer
    metadata = {
        "task_type": "siddoc" if siddoc_mode else "rec",
        "uid": int(row["uid"]),
        "target_domain": "product",
        "history_len": video_hist_sid_count + product_hist_sid_count,
        "video_hist_sid_count": video_hist_sid_count,
        "ad_hist_sid_count": product_hist_sid_count,
        "history_source": "bench_original",
    }
    return {
        "uuid": str(uuid.uuid4()),
        "source": SOURCE_PRODUCT_REC,
        "messages": build_messages_json(system_prompt, user_prompt, assistant_text),
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def build_non_cot_rec_records(
    bench_df: pd.DataFrame,
    pid2sid: Dict[int, Any],
    product_pid2sid: Dict[int, Any],
    cot_user_keys: Set[Tuple[int, str]],
    allow_overlap: bool,
    enable_video: bool,
    enable_ad: bool,
    enable_product: bool,
    split_target: int,
    seed: int,
    rank: int,
    world_size: int,
    siddoc_mode: int = 0,
) -> List[Dict[str, Any]]:
    """从 bench 构造非 CoT 推荐样本。

    - 仅处理 split=0 的行（对齐 video_rec.py / ad_rec.py / product_rec.py）
    - bench_df 由调用方传入（与 CoT 侧共享同一份 df，避免重复加载）
    - allow_overlap=False 时，跳过 (uid, domain) ∈ cot_user_keys 的样本
    - split_target!=0 时，每个域的 target 按 TARGET_MAX_LEN 拆成多条单 target 样本
      （per-sample 确定性 RNG，与 CoT 侧一致）
    - siddoc_mode（仅作用于非 CoT rec 样本）：
        0 = 原格式（rec 样本，assistant = target）
        1 = 纯 siddoc 替换（rec 全部变 siddoc，assistant = 历史标签序列，无 target）
        2 = mix：每个用户每域同时造一条原格式 rec + 一条 siddoc，混在一起训练。
            rec 学 target 预测，siddoc 补历史自回归 CE loss。
    """
    build_normal = siddoc_mode in (0, 2)
    build_siddoc = siddoc_mode in (1, 2)
    siddoc_factor = (1 if build_normal else 0) + (1 if build_siddoc else 0)
    t0 = time.time()
    logger.info("rank=%s 使用已加载 bench_df: rows=%s siddoc_mode=%s (normal=%s siddoc=%s)",
                rank, len(bench_df), siddoc_mode, build_normal, build_siddoc)

    user_indices = list(range(len(bench_df)))
    my_indices = chunk_for_rank(user_indices, rank, world_size)
    logger.info("rank=%s 分配用户数=%s / 总用户数=%s", rank, len(my_indices), len(user_indices))

    records: List[Dict[str, Any]] = []
    stats = {"video": 0, "ad": 0, "product": 0, "skipped_split": 0, "skipped_overlap": 0}

    progress = tqdm(my_indices, desc=f"rank{rank} non-cot", disable=rank != 0)
    for idx in progress:
        row = bench_df.iloc[idx]
        if int(row.get("split", 0)) != 0:
            stats["skipped_split"] += 1
            continue
        uid = int(row["uid"])

        if enable_video:
            if (uid, "video") in cot_user_keys and not allow_overlap:
                stats["skipped_overlap"] += 1
            else:
                target_pids_full = to_int_list(row.get("target_video_pid"))
                rng = random.Random(_stable_seed("", uid, "video", seed))
                subsets = compute_target_subsets(
                    target_pids_full, split_target, TARGET_MAX_LEN, rng)
                if build_normal:
                    for subset in subsets:
                        rec = build_non_cot_video_record(row, pid2sid, target_pids_override=subset, siddoc_mode=False)
                        if rec:
                            records.append(rec)
                if build_siddoc:
                    for subset in subsets:
                        rec = build_non_cot_video_record(row, pid2sid, target_pids_override=subset, siddoc_mode=True)
                        if rec:
                            records.append(rec)
                stats["video"] += len(subsets) * siddoc_factor

        if enable_ad:
            if (uid, "ad") in cot_user_keys and not allow_overlap:
                stats["skipped_overlap"] += 1
            else:
                target_pids_full = to_int_list(row.get("target_ad_pid"))
                rng = random.Random(_stable_seed("", uid, "ad", seed))
                subsets = compute_target_subsets(
                    target_pids_full, split_target, TARGET_MAX_LEN, rng)
                if build_normal:
                    for subset in subsets:
                        rec = build_non_cot_ad_record(row, pid2sid, target_pids_override=subset, siddoc_mode=False)
                        if rec:
                            records.append(rec)
                if build_siddoc:
                    for subset in subsets:
                        rec = build_non_cot_ad_record(row, pid2sid, target_pids_override=subset, siddoc_mode=True)
                        if rec:
                            records.append(rec)
                stats["ad"] += len(subsets) * siddoc_factor

        if enable_product:
            # product 域没有 CoT，不受 overlap 限制
            target_pids_full = to_int_list(row.get("target_goods_pid"))
            rng = random.Random(_stable_seed("", uid, "product", seed))
            subsets = compute_target_subsets(
                target_pids_full, split_target, TARGET_MAX_LEN, rng)
            if build_normal:
                for subset in subsets:
                    rec = build_non_cot_product_record(
                        row, pid2sid, product_pid2sid, target_pids_override=subset, siddoc_mode=False)
                    if rec:
                        records.append(rec)
            if build_siddoc:
                for subset in subsets:
                    rec = build_non_cot_product_record(
                        row, pid2sid, product_pid2sid, target_pids_override=subset, siddoc_mode=True)
                    if rec:
                        records.append(rec)
            stats["product"] += len(subsets) * siddoc_factor

    logger.info("rank=%s 非 CoT 构造完成: records=%s stats=%s 耗时=%.1fs",
                rank, len(records), stats, time.time() - t0)
    return records


# =============================================================================
# Caption 任务：sid2caption / caption2sid
# =============================================================================
def resolve_caption_files(caption_dir: Path, shuffle: bool, file_rng: random.Random) -> List[Path]:
    files = list_parquet_shards(caption_dir)
    if shuffle:
        file_rng.shuffle(files)
    return files


def reservoir_sample_disjoint_pids(
    caption_files: List[Path],
    valid_pids: Set[int],
    total_target: int,
    sample_rng: random.Random,
    rank: int,
    pid_column: str = "pid",
    caption_column: str = "dense_caption",
) -> Tuple[List[int], int]:
    """跨 caption 分片做 reservoir sampling，返回互不相交的 pid 列表。

    pid_column / caption_column 参数化以支持不同 schema（如 product 域
    用 new_pid + caption），默认值兼容 video/ad 的 caption_shards。
    """
    if total_target <= 0:
        return [], 0
    sampled: List[int] = []
    seen_unique = 0
    seen_set: Set[int] = set()

    for fpath in tqdm(caption_files, desc=f"rank{rank} sample pids", disable=rank != 0):
        try:
            df = pd.read_parquet(fpath, columns=[pid_column, caption_column])
        except Exception as exc:
            logger.warning("rank=%s 读取 caption 失败 %s: %s", rank, fpath, exc)
            continue
        for row in df.itertuples(index=False):
            pid = int(getattr(row, pid_column))
            if pid not in valid_pids:
                continue
            cap = normalize_caption(getattr(row, caption_column))
            if not cap:
                continue
            if pid in seen_set:
                continue
            seen_set.add(pid)
            seen_unique += 1
            if len(sampled) < total_target:
                sampled.append(pid)
                continue
            replace_idx = sample_rng.randint(0, seen_unique - 1)
            if replace_idx < total_target:
                sampled[replace_idx] = pid

    return sampled, seen_unique


def build_caption_records(
    caption_files: List[Path],
    pid2sid: Dict[int, Any],
    sid2caption_pids: Set[int],
    caption2sid_pids: Set[int],
    rank: int,
    world_size: int,
    pid_column: str = "pid",
    caption_column: str = "dense_caption",
    domain: str = "video",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """遍历 caption 分片，为命中的 pid 构造 sid2caption / caption2sid 样本。

    按 (pid, task) 唯一化：每个 pid 只为它被分配到的那一个任务产出一条样本。
    pid_column / caption_column 参数化以支持不同 schema（如 product 域
    用 new_pid + caption），默认值兼容 video/ad 的 caption_shards。

    ★ domain 参数（0728 新增，修复 prompt 与 content 不匹配 bug）：
      - "video"（默认）：用 video/ad 的"视频"prompt（SID2CAPTION_* / CAPTION2SID_*），
        覆盖 video_ad_pid2sid 的 pid 空间（video + ad 共享 sid/caption）。
      - "product"：用 product 域的"商品"prompt（PRODUCT_SID2CAPTION_* / PRODUCT_CAPTION2SID_*），
        覆盖 product_pid2sid 的 pid 空间（goods_pid2caption）。
      旧 bug：product caption 复用 video "视频"prompt → assistant 输出商品描述但
      prompt 问"这个视频讲的是什么" → prompt-content mismatch，污染 product sid 语义。
      metadata 新增 "domain" 字段便于审计。
    """
    # 按 domain 选 prompt 集（0728 修复 product caption prompt bug）
    if domain == "product":
        s2c_sys_prompts = PRODUCT_SID2CAPTION_SYSTEM_PROMPTS
        s2c_usr_prompts = PRODUCT_SID2CAPTION_USER_PROMPTS
        c2s_sys_prompts = PRODUCT_CAPTION2SID_SYSTEM_PROMPTS
        c2s_usr_prompts = PRODUCT_CAPTION2SID_USER_PROMPTS
    else:
        s2c_sys_prompts = SID2CAPTION_SYSTEM_PROMPTS
        s2c_usr_prompts = SID2CAPTION_USER_PROMPTS
        c2s_sys_prompts = CAPTION2SID_SYSTEM_PROMPTS
        c2s_usr_prompts = CAPTION2SID_USER_PROMPTS

    c2s_records: List[Dict[str, Any]] = []
    s2c_records: List[Dict[str, Any]] = []
    seen_c2s: Set[int] = set()
    seen_s2c: Set[int] = set()

    # round-robin 分片给各 rank
    my_files = chunk_for_rank(caption_files, rank, world_size)
    logger.info("rank=%s 分配 caption 分片=%s / 总分片=%s (domain=%s)", rank, len(my_files), len(caption_files), domain)

    for fpath in tqdm(my_files, desc=f"rank{rank} caption[{domain}]", disable=rank != 0):
        try:
            df = pd.read_parquet(fpath, columns=[pid_column, caption_column])
        except Exception as exc:
            logger.warning("rank=%s 读取 caption 失败 %s: %s", rank, fpath, exc)
            continue
        for row in df.itertuples(index=False):
            pid = int(getattr(row, pid_column))
            dense_caption = normalize_caption(getattr(row, caption_column))
            if not dense_caption:
                continue
            sid = pid_to_sid_str(pid, pid2sid)
            if not sid:
                continue

            if pid in sid2caption_pids and pid not in seen_s2c:
                seen_s2c.add(pid)
                system_prompt = random.choice(s2c_sys_prompts)
                user_prompt = random.choice(s2c_usr_prompts).format(sid=sid)
                metadata = {"task_type": "sid2caption", "pid": pid, "sid": sid, "domain": domain}
                s2c_records.append({
                    "uuid": str(uuid.uuid4()),
                    "source": SOURCE_ITEM_UNDERSTAND,
                    "messages": build_messages_json(system_prompt, user_prompt, dense_caption),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                })

            if pid in caption2sid_pids and pid not in seen_c2s:
                seen_c2s.add(pid)
                system_prompt = random.choice(c2s_sys_prompts)
                user_prompt = random.choice(c2s_usr_prompts).format(caption=dense_caption)
                metadata = {"task_type": "caption2sid", "pid": pid, "sid": sid, "domain": domain}
                c2s_records.append({
                    "uuid": str(uuid.uuid4()),
                    "source": SOURCE_CAPTION_TO_SID,
                    "messages": build_messages_json(system_prompt, user_prompt, sid),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                })

    logger.info("rank=%s caption[%s] 构造完成: s2c=%s c2s=%s", rank, domain, len(s2c_records), len(c2s_records))
    return s2c_records, c2s_records


# =============================================================================
# 阶段1：写 per-rank 临时文件
# =============================================================================
def write_temp_file(records: List[Dict[str, Any]], temp_path: Path) -> int:
    """把 records 写成单个临时 parquet，返回行数。"""
    if not records:
        # 仍写一个空文件，保证布局一致
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, temp_path)
    return len(df)


# =============================================================================
# 阶段2：全局 shuffle + 分片写出
# =============================================================================
def gather_temp_layout(
    comm,
    local_temp_path: str,
    local_rows: int,
) -> Tuple[List[str], List[int], int]:
    """Allgather 临时文件路径和行数，返回全局布局。

    Returns:
      temp_paths: list[str]，各 rank 的临时文件路径
      temp_counts: list[int]，各 rank 临时文件行数
      total_rows: int，全局总行数
    """
    local_info = (local_temp_path, local_rows)
    gathered = comm.allgather(local_info)
    temp_paths = [info[0] for info in gathered]
    temp_counts = [info[1] for info in gathered]
    total_rows = sum(temp_counts)
    return temp_paths, temp_counts, total_rows


def compute_global_permutation(total_rows: int, seed: int, sample_ratio: float) -> np.ndarray:
    """用固定 seed 生成全局置换，并按 sample_ratio 截断。

    所有 rank 用相同 seed → 相同 perm → 一致的全局行序。
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rows)
    if sample_ratio is not None and 0 < sample_ratio < 1.0:
        keep = max(1, int(round(total_rows * sample_ratio)))
        perm = perm[:keep]
    return perm


def assign_shards_to_ranks(n_shards: int, world_size: int) -> List[Tuple[int, int]]:
    """把 n_shards 个分片 round-robin 分给各 rank。

    Returns:
      list[(rank, shard_idx)]
    """
    assignments: List[Tuple[int, int]] = []
    for shard_idx in range(n_shards):
        rank = shard_idx % world_size
        assignments.append((rank, shard_idx))
    return assignments


def read_rows_from_temp_files(
    temp_paths: List[str],
    temp_counts: List[int],
    global_indices: np.ndarray,
) -> pd.DataFrame:
    """从多个临时文件按全局行索引读取所需行，返回 DataFrame。

    输出顺序严格按 ``global_indices`` 的顺序（即 perm 的顺序）。

    实现策略：
    1. 先把每个 global_idx 映射到 (file_idx, local_idx)，按 file_idx 分组。
    2. 对每个临时文件用 ``pyarrow.Table.take(local_indices)`` 一次性读取所需行。
    3. 读完后用 (file_idx, local_idx) → global_idx 的映射，按 global_indices
       的原始顺序重建结果。
    """
    if len(global_indices) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 计算每个临时文件的起始全局行号（starts[i] = 第 i 个临时文件的第一行全局行号）
    starts = [0]
    for cnt in temp_counts[:-1]:
        starts.append(starts[-1] + cnt)

    # 把 global_indices 按 file_idx 分组，同时记录每个 global_idx 在结果中的目标位置
    # file_idx -> list of (local_idx, target_pos)
    file_to_items: Dict[int, List[Tuple[int, int]]] = {}
    for pos, gidx in enumerate(global_indices):
        gidx_int = int(gidx)
        # 找到 gidx 属于哪个临时文件
        file_idx = bisect_right(starts, gidx_int) - 1
        if file_idx < 0:
            file_idx = 0
        local_idx = gidx_int - starts[file_idx]
        file_to_items.setdefault(file_idx, []).append((local_idx, pos))

    # 读取各临时文件所需行，存入按 target_pos 索引的 dict
    result_rows: Dict[int, pd.Series] = {}
    for file_idx, items in file_to_items.items():
        path = temp_paths[file_idx]
        local_indices = [it[0] for it in items]
        try:
            table = pq.read_table(path)
        except Exception as exc:
            logger.error("读取临时文件失败 %s: %s", path, exc)
            raise
        taken = table.take(local_indices).to_pandas()
        for row, (_, pos) in zip(
            [taken.iloc[i] for i in range(len(taken))], items
        ):
            result_rows[pos] = row

    # 按 global_indices 的顺序组装结果
    ordered_rows = [result_rows[pos] for pos in range(len(global_indices))]
    combined = pd.DataFrame(ordered_rows, columns=OUTPUT_COLUMNS)
    return combined


def batch_read_rows_from_temp_files(
    temp_paths: List[str],
    temp_counts: List[int],
    shard_ranges: List[Tuple[int, np.ndarray]],
    rank: int,
) -> Dict[int, Dict[str, Any]]:
    """批量读取本 rank 所有分片需要的行，每个 temp file 只读一次。

    修复性能瓶颈：原 ``read_rows_from_temp_files`` 每个 shard 都会全量读取
    temp file（``pq.read_table`` 全量加载），导致 O(n_shards × n_temp_files)
    次全文件 I/O。以 2.3M 行 / 2330 shards / 16 ranks 为例，每 rank 需读
    ~146 个 shard × 16 个 temp file = 2336 次全文件读（每次 ~145K 行），
    总磁盘 I/O ~1TB/rank，耗时数小时。

    本函数把所有 shard 的 ``global_indices`` 合并，按 temp file 分组后每个
    文件只读一次，用 ``Table.take()`` 批量提取所需行，磁盘 I/O 降低 ~n_shards 倍。
    """
    if not shard_ranges:
        return {}

    # 计算每个临时文件的起始全局行号
    starts = [0]
    for cnt in temp_counts[:-1]:
        starts.append(starts[-1] + cnt)

    # 收集所有 global_indices，按 temp file 分组
    # file_idx -> {local_idx: global_idx}（perm 是置换，每个 global_idx 唯一）
    file_to_needed: Dict[int, Dict[int, int]] = {}
    total_needed = 0
    for _, global_indices in shard_ranges:
        for gidx in global_indices:
            gidx_int = int(gidx)
            file_idx = bisect_right(starts, gidx_int) - 1
            if file_idx < 0:
                file_idx = 0
            local_idx = gidx_int - starts[file_idx]
            file_to_needed.setdefault(file_idx, {})[local_idx] = gidx_int
            total_needed += 1

    # 逐个 temp file 读取，只读一次，批量提取所需行
    global_row_cache: Dict[int, Dict[str, Any]] = {}
    t_batch = time.time()
    for file_idx, local_to_global in file_to_needed.items():
        path = temp_paths[file_idx]
        local_indices = sorted(local_to_global.keys())
        try:
            table = pq.read_table(path)
        except Exception as exc:
            logger.error("rank=%s 读取临时文件失败 %s: %s", rank, path, exc)
            raise
        taken = table.take(local_indices).to_pandas()
        for i, local_idx in enumerate(local_indices):
            gidx = local_to_global[local_idx]
            global_row_cache[gidx] = taken.iloc[i].to_dict()
        # 显式释放 Arrow Table，降低峰值内存
        del table
        del taken

    logger.info(
        "rank=%s 批量读取完成: temp_files=%s needed=%s cached=%s 耗时=%.1fs",
        rank, len(file_to_needed), total_needed, len(global_row_cache),
        time.time() - t_batch,
    )
    return global_row_cache


def write_shard(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path)


# =============================================================================
# 数据生成结束：统计每个 source 的样本条数 + 横向字符柱状图
# =============================================================================
def _count_sources_from_shards(shard_paths: List[str]) -> Dict[str, int]:
    """读取所有输出分片 parquet，统计每个 source 的样本条数。

    只读 `source` 列（不读 messages 等大列），速度很快。
    """
    source_counts: Dict[str, int] = {}
    for shard_path in tqdm(shard_paths, desc="统计 source 分布", disable=False):
        try:
            df = pd.read_parquet(shard_path, columns=["source"])
        except Exception as e:
            logger.warning("读取分片失败 %s: %s", shard_path, e)
            continue
        counts = df["source"].value_counts()
        for src, cnt in counts.items():
            source_counts[src] = source_counts.get(src, 0) + int(cnt)
    return source_counts


def _print_source_bar_chart(source_counts: Dict[str, int]) -> None:
    """打印横向字符柱状图展示各 source 的样本条数与占比。

    示例:
      RecIF_RecCOT       |████████████████████████████| 1,200,000 (52.3%)
      RecIF_VideoRec      |██████████████              |   640,000 (27.9%)
      ...
    """
    if not source_counts:
        print("  (无 source 数据)")
        return

    total = sum(source_counts.values())
    if total == 0:
        print("  (总样本数为 0)")
        return

    # 按 count 降序排列
    sorted_items = sorted(source_counts.items(), key=lambda x: -x[1])

    # 格式化：source 名左对齐，count 右对齐带千位分隔符，百分比右对齐
    max_name_len = max(len(src) for src, _ in sorted_items)
    max_count_str = max(len(f"{cnt:,}") for _, cnt in sorted_items)

    # 柱状图最大宽度（字符数）
    max_bar_width = 32
    max_count = sorted_items[0][1]

    print("\n================ Source Distribution ================")
    print(f"  {'source':<{max_name_len}}  |{'bar':<{max_bar_width}}| {'count':>{max_count_str}} (ratio)")
    print(f"  {'-' * max_name_len}  |{'-' * max_bar_width}| {'-' * max_count_str} -------")

    for src, cnt in sorted_items:
        ratio = cnt / total
        bar_len = int(round(max_bar_width * cnt / max_count)) if max_count > 0 else 0
        bar = "█" * bar_len
        print(
            f"  {src:<{max_name_len}}  |{bar:<{max_bar_width}}| "
            f"{cnt:>{max_count_str},} ({ratio:5.1%})"
        )

    print(f"  {'-' * max_name_len}  |{'-' * max_bar_width}| {'-' * max_count_str} -------")
    print(f"  {'Total':<{max_name_len}}  |{' ' * max_bar_width}| {total:>{max_count_str},} (100.0%)")
    print("======================================================")


# =============================================================================
# 主流程
# =============================================================================
def run(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)

    # MPI 初始化
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        comm = None
        rank = 0
        world_size = 1
        logger.warning("mpi4py 不可用，退化为单进程模式。")

    random.seed(args.seed + rank)  # 各 rank 不同，保证 prompt 采样的多样性
    np.random.seed(args.seed)

    # 覆盖模块级 TARGET_MAX_LEN，使后续所有引用（build 函数内部的截断、
    # compute_target_subsets 的 max_len、build_summary、终端 print）一致使用
    # CLI 指定值。模块级常量仅作为 argparse default。
    global TARGET_MAX_LEN
    TARGET_MAX_LEN = args.target_max_len
    logger.info(
        "rank=%s target_max_len=%s split_target=%s",
        rank, TARGET_MAX_LEN, args.split_target,
    )

    t_total = time.time()

    # ------------------------------------------------------------------
    # 0. 校验输入
    # ------------------------------------------------------------------
    cot_dir = Path(args.cot_dir)
    bench_parquet = args.bench_parquet
    caption_dir = Path(args.caption_dir) if args.caption_dir else None
    pid2sid_path = args.pid2sid
    product_pid2sid_path = args.product_pid2sid
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir) if args.temp_dir else output_dir / "_temp"

    if not cot_dir.exists():
        raise FileNotFoundError(f"cot_dir 不存在: {cot_dir}")
    if not Path(bench_parquet).exists():
        raise FileNotFoundError(f"bench_parquet 不存在: {bench_parquet}")
    if args.enable_caption2sid or args.enable_sid2caption:
        if caption_dir is None or not caption_dir.exists():
            raise FileNotFoundError(f"caption_dir 不存在: {caption_dir}")
    if args.enable_product_sid2caption or args.enable_product_caption2sid:
        if not args.product_caption_dir:
            raise ValueError("product_caption_dir 必填当启用 product caption 任务时")
        if not Path(args.product_caption_dir).exists():
            raise FileNotFoundError(f"product_caption_dir 不存在: {args.product_caption_dir}")
    if not Path(pid2sid_path).exists():
        raise FileNotFoundError(f"pid2sid 不存在: {pid2sid_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载 pid2sid 映射（每 rank 全量加载）
    # ------------------------------------------------------------------
    pid2sid = load_pid2sid(pid2sid_path, rank, "pid2sid(video/ad)")
    product_pid2sid: Dict[int, Any] = {}
    # ★ product_pid2sid 只要提供了路径就加载（不门控于 enable_product_rec）。
    #   原因：CoT 样本可能含 product 域（三域 2_cot_merged），build_cot_record 对
    #   product CoT 样本的 target + goods history 段**必须**用 product_pid2sid
    #   （电商 sid），且不再有 fallback（旧 bug：fallback 到 video/ad pid2sid，
    #   因 pid 数值重叠得到错误 sid，样本数/token 数不变但 sid 标签 100% 错）。
    #   未提供 --product_pid2sid 路径时留空 dict，build_cot_record 遇到 product
    #   样本会立刻抛 ValueError 暴露问题（而非静默用错 sid）。
    if product_pid2sid_path:
        product_pid2sid = load_pid2sid(product_pid2sid_path, rank, "product_pid2sid")

    # ------------------------------------------------------------------
    # 2. 加载 bench（完整），构建 uid→row_idx 与 uid→split 映射
    # ------------------------------------------------------------------
    # bench_df 同时用于：
    #   a) CoT 样本按 uid 查找原始 history / target（修复 CoT 与非 CoT 历史不一致）
    #   b) CoT 按 split 过滤（uid_to_split）
    #   c) 传给 build_non_cot_rec_records 构造非 CoT 推荐样本（复用同一份 df）
    # 只在需要时加载：启用非 CoT 推荐或有 CoT 样本时。CoT 样本数在加载前未知，
    # 但只要 enable_non_cot_rec 或 cot_dir 非空就需要 bench，故按这两个条件判断。
    need_bench = args.enable_non_cot_rec or cot_dir.exists()
    bench_df: Optional[pd.DataFrame] = None
    uid_to_idx: Dict[int, int] = {}
    uid_to_split: Optional[Dict[int, int]] = None
    if need_bench:
        bench_df, uid_to_idx, uid_to_split_from_bench = load_bench_df(bench_parquet, rank)
        if args.enable_cot_split_filter:
            uid_to_split = uid_to_split_from_bench
        else:
            uid_to_split = None
    elif args.enable_cot_split_filter:
        # 没有 bench 但需要 split 过滤的退化路径（理论上不会走到，cot_dir 存在时 need_bench=True）
        uid_to_split = load_uid_split_map(bench_parquet, rank)

    # ------------------------------------------------------------------
    # 3. 加载 CoT 样本（step2 / step4 输出），按 uid 的 split 过滤
    # ------------------------------------------------------------------
    cot_samples, local_cot_user_keys = load_cot_samples(
        cot_dir, rank, world_size, uid_to_split=uid_to_split,
    )

    # ★ Fail-fast：若 CoT 样本含 product 域但 product_pid2sid 为空，立刻报错。
    #   避免 build_cot_record 在 product 样本上因缺失 product_pid2sid 而抛错
    #   （旧逻辑会更糟：静默 fallback 到 video/ad pid2sid → 错误 sid 不报错）。
    #   扫描本地 cot_samples 即可判断（product 域的存在性跨 rank 一致）。
    local_has_product = any(s.get("target_domain") == "product" for s in cot_samples)
    if local_has_product and not product_pid2sid:
        raise ValueError(
            "CoT 样本含 target_domain=product（三域 2_cot_merged?），"
            "但 product_pid2sid 为空（未传 --product_pid2sid 或路径不存在）。"
            "product 域 sid 映射缺失，无法构造正确训练数据。"
            "请传 --product_pid2sid 指向 product_pid2sid.parquet。"
        )

    # 汇总 CoT user_keys 到所有 rank（用于非 cot 重叠判断）
    if comm is not None:
        global_cot_user_keys_lists = comm.allgather(local_cot_user_keys)
        cot_user_keys: Set[Tuple[int, str]] = set()
        for s in global_cot_user_keys_lists:
            cot_user_keys.update(s)
        logger.info("rank=%s 全局 CoT user_keys=%s", rank, len(cot_user_keys))
    else:
        cot_user_keys = local_cot_user_keys

    # ------------------------------------------------------------------
    # 4. 构造 CoT 训练样本
    # ------------------------------------------------------------------
    # 修复 bug：CoT 与非 CoT 历史不一致。
    #   - history 优先用 bench row 的**原始** pid（经 pid2sid 过滤），
    #     与 video_rec.py / ad_rec.py 完全对齐；caption 过滤丢弃的 pid 也会出现。
    #   - bench row 缺失（uid 不在 bench）时退化为 caption-filtered history_pids。
    #
    # ★ cot_target_source 选项（0717 新增）：
    #   - "bench"（默认）：target_pids 用 bench row 的原始 target（原行为）
    #   - "cot"：target_pids 用 CoT sample 自带的 target_pids
    #     （0717 新链路：4a_filter_ppl 已做 per-target 筛选，输出的 target_pids
    #      是 helpful bench targets，5_data_format 直接用，不再做额外筛选）
    #
    # ★ 0717 改动：筛选在 step 4a 完成（per-target filter），5_data_format 不再做
    #   per-target PPL 筛选，只根据 4a 输出的 target_pids 构造训练数据。
    #
    # 按 split_target 拆分 target：
    #   split_target=0 → 每条样本保留前 TARGET_MAX_LEN 个 target（单条多 target）
    #   split_target!=0 → 每条原始样本按 target 拆成最多 TARGET_MAX_LEN 条，
    #                     每条 1 个 target（CoT 文本复用同一段，因为 CoT 不引用具体 id）
    # CoT 侧的 siddoc：仅在 siddoc_mode==1（纯替换）时把 CoT 样本也变成 siddoc
    # 格式（assistant = 历史 + CoT answer）。siddoc_mode==2（mix）只作用于非 CoT
    # rec 样本，CoT 保持原格式（CoT 本身已有 think + target，无需再补历史 loss）。
    cot_siddoc = (args.siddoc_mode == 1)
    cot_target_source = getattr(args, "cot_target_source", "bench")
    logger.info(
        "rank=%s CoT target 配置: cot_target_source=%s (筛选在 step 4a 完成)",
        rank, cot_target_source,
    )
    cot_records: List[Dict[str, Any]] = []
    cot_missing_bench = 0
    for sample in cot_samples:
        uid = int(sample["uid"])
        target_domain = sample["target_domain"]

        # 查找 bench row（用于 history 对齐）
        bench_row: Optional[pd.Series] = None
        if bench_df is not None and uid in uid_to_idx:
            bench_row = bench_df.iloc[uid_to_idx[uid]]
        else:
            cot_missing_bench += 1

        # ---- target_pids 来源选择 ----
        if cot_target_source == "cot":
            # 用 CoT sample 自带的 target_pids（4a 已做 per-target 筛选，= helpful bench targets）
            orig_target_pids = list(sample["target_pids"])
        else:
            # 默认：用 bench row 的原始 target（与非 CoT 对齐）
            orig_target_pids = get_orig_target_pids(
                bench_row, target_domain, sample["target_pids"],
            )

        rng = random.Random(_stable_seed(
            sample.get("sample_key") or "",
            uid,
            target_domain,
            args.seed,
        ))
        subsets = compute_target_subsets(
            orig_target_pids, args.split_target, TARGET_MAX_LEN, rng,
        )
        for subset in subsets:
            # 浅拷贝 + 覆盖 target_pids，避免修改原始 sample
            sub_sample = dict(sample, target_pids=subset)
            rec = build_cot_record(
                sub_sample, pid2sid,
                product_pid2sid=product_pid2sid,
                bench_row=bench_row,
                siddoc_mode=cot_siddoc,
            )
            if rec:
                cot_records.append(rec)
    if cot_missing_bench:
        logger.warning(
            "rank=%s CoT 样本中 %s 条找不到对应 bench row，退化为 caption-filtered history",
            rank, cot_missing_bench,
        )
    logger.info("rank=%s CoT 训练样本构造完成: %s", rank, len(cot_records))

    # ------------------------------------------------------------------
    # 5. 构造非 CoT 推荐样本（从 bench）
    # ------------------------------------------------------------------
    non_cot_records: List[Dict[str, Any]] = []
    if args.enable_non_cot_rec:
        if bench_df is None:
            # enable_non_cot_rec=1 但 need_bench 为 False 的情况不会发生
            # （enable_non_cot_rec=1 时 need_bench=True），防御性兜底
            raise RuntimeError("bench_df 未加载，无法构造非 CoT 样本")
        non_cot_records = build_non_cot_rec_records(
            bench_df=bench_df,
            pid2sid=pid2sid,
            product_pid2sid=product_pid2sid,
            cot_user_keys=cot_user_keys,
            allow_overlap=args.allow_cot_non_cot_user_overlap,
            enable_video=args.enable_video_rec,
            enable_ad=args.enable_ad_rec,
            enable_product=args.enable_product_rec,
            split_target=args.split_target,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            siddoc_mode=args.siddoc_mode,
        )

    # ------------------------------------------------------------------
    # 5. 构造 caption 任务样本（sid2caption / caption2sid）
    # ------------------------------------------------------------------
    s2c_records: List[Dict[str, Any]] = []
    c2s_records: List[Dict[str, Any]] = []
    if args.enable_sid2caption or args.enable_caption2sid:
        if caption_dir is None:
            raise ValueError("caption_dir 必填当启用 caption 任务时")

        file_rng = random.Random(args.seed + 10)
        caption_files = resolve_caption_files(caption_dir, args.shuffle_caption_files, file_rng)

        # 采样 disjoint pid allowlist
        total_target = args.sid2caption_size + args.caption2sid_size
        sid2caption_pids: Set[int] = set()
        caption2sid_pids: Set[int] = set()
        if total_target > 0:
            # 只在 rank 0 采样一次，然后广播
            if comm is not None:
                if rank == 0:
                    sampled, seen_unique = reservoir_sample_disjoint_pids(
                        caption_files,
                        valid_pids=set(pid2sid.keys()),
                        total_target=total_target,
                        sample_rng=random.Random(args.seed + 1),
                        rank=rank,
                    )
                    if len(sampled) < total_target:
                        raise RuntimeError(
                            f"valid unique pids 不足: need {total_target}, got {len(sampled)}"
                        )
                    split_rng = random.Random(args.seed + 20)
                    split_rng.shuffle(sampled)
                    sid2cap_list = sampled[:args.sid2caption_size]
                    cap2sid_list = sampled[args.sid2caption_size:total_target]
                else:
                    sid2cap_list = None
                    cap2sid_list = None
                sid2cap_list = comm.bcast(sid2cap_list, root=0)
                cap2sid_list = comm.bcast(cap2sid_list, root=0)
            else:
                sampled, seen_unique = reservoir_sample_disjoint_pids(
                    caption_files,
                    valid_pids=set(pid2sid.keys()),
                    total_target=total_target,
                    sample_rng=random.Random(args.seed + 1),
                    rank=rank,
                )
                if len(sampled) < total_target:
                    raise RuntimeError(
                        f"valid unique pids 不足: need {total_target}, got {len(sampled)}"
                    )
                split_rng = random.Random(args.seed + 20)
                split_rng.shuffle(sampled)
                sid2cap_list = sampled[:args.sid2caption_size]
                cap2sid_list = sampled[args.sid2caption_size:total_target]

            sid2caption_pids = set(sid2cap_list)
            caption2sid_pids = set(cap2sid_list)
            logger.info(
                "rank=%s caption allowlist: sid2caption=%s caption2sid=%s",
                rank, len(sid2caption_pids), len(caption2sid_pids),
            )

        if not args.enable_sid2caption:
            sid2caption_pids = set()
        if not args.enable_caption2sid:
            caption2sid_pids = set()

        s2c_records, c2s_records = build_caption_records(
            caption_files=caption_files,
            pid2sid=pid2sid,
            sid2caption_pids=sid2caption_pids,
            caption2sid_pids=caption2sid_pids,
            rank=rank,
            world_size=world_size,
            domain="video",  # video/ad 共享"视频"prompt + video_ad_pid2sid
        )

    # ------------------------------------------------------------------
    # 5b. 构造 product 域 caption 任务样本（0723 新增，独立于 video/ad）
    # ------------------------------------------------------------------
    # 与 video/ad caption 结构一致，但：
    #   - caption 来源：goods_pid2caption（列 new_pid + caption）
    #   - sid 映射：product_pid2sid（key = new_pid）
    #   - 采样 seed 偏移 +101/+201 与 video/ad 的 +1/+20 错开
    # 实现下放到 product_caption.py（依赖注入避免循环导入）
    product_s2c_records: List[Dict[str, Any]] = []
    product_c2s_records: List[Dict[str, Any]] = []
    if args.enable_product_sid2caption or args.enable_product_caption2sid:
        if not args.product_caption_dir:
            raise ValueError("product_caption_dir 必填当启用 product caption 任务时")
        if not product_pid2sid:
            raise ValueError("product_pid2sid 为空，无法构造 product caption 样本（需 --product_pid2sid）")

        from product_caption import (
            sample_product_caption_pids,
            build_product_caption_records,
        )

        product_caption_dir = Path(args.product_caption_dir)
        if not product_caption_dir.exists():
            raise FileNotFoundError(f"product_caption_dir 不存在: {product_caption_dir}")

        product_file_rng = random.Random(args.seed + 110)
        product_caption_files = resolve_caption_files(
            product_caption_dir, args.shuffle_caption_files, product_file_rng,
        )

        prod_sid2cap_size = args.product_sid2caption_size if args.enable_product_sid2caption else 0
        prod_cap2sid_size = args.product_caption2sid_size if args.enable_product_caption2sid else 0

        prod_sid2cap_pids, prod_cap2sid_pids = sample_product_caption_pids(
            product_caption_files=product_caption_files,
            product_pid2sid=product_pid2sid,
            sid2caption_size=prod_sid2cap_size,
            caption2sid_size=prod_cap2sid_size,
            seed=args.seed,
            rank=rank,
            comm=comm,
            reservoir_sample_fn=reservoir_sample_disjoint_pids,
        )
        logger.info(
            "rank=%s product caption allowlist: sid2caption=%s caption2sid=%s",
            rank, len(prod_sid2cap_pids), len(prod_cap2sid_pids),
        )

        product_s2c_records, product_c2s_records = build_product_caption_records(
            product_caption_files=product_caption_files,
            product_pid2sid=product_pid2sid,
            sid2caption_pids=prod_sid2cap_pids,
            caption2sid_pids=prod_cap2sid_pids,
            rank=rank,
            world_size=world_size,
            build_caption_fn=build_caption_records,
        )
        logger.info(
            "rank=%s product caption 构造完成: s2c=%s c2s=%s",
            rank, len(product_s2c_records), len(product_c2s_records),
        )

    # ------------------------------------------------------------------
    # 6. 阶段1：合并本地所有 records，写 per-rank 临时文件
    # ------------------------------------------------------------------
    all_local_records = (
        cot_records + non_cot_records
        + s2c_records + c2s_records
        + product_s2c_records + product_c2s_records
    )
    logger.info(
        "rank=%s 本地 records 汇总: cot=%s non_cot=%s s2c=%s c2s=%s prod_s2c=%s prod_c2s=%s total=%s",
        rank, len(cot_records), len(non_cot_records),
        len(s2c_records), len(c2s_records),
        len(product_s2c_records), len(product_c2s_records),
        len(all_local_records),
    )

    temp_path = temp_dir / f"temp_rank{rank:05d}.parquet"
    local_rows = write_temp_file(all_local_records, temp_path)
    logger.info("rank=%s 写出临时文件: %s rows=%s", rank, temp_path, local_rows)

    if comm is not None:
        comm.Barrier()

    # ------------------------------------------------------------------
    # 7. 阶段2：全局 shuffle + 分片写出
    # ------------------------------------------------------------------
    # 7.1 Allgather 临时文件布局
    if comm is not None:
        temp_paths, temp_counts, total_rows = gather_temp_layout(comm, str(temp_path), local_rows)
    else:
        temp_paths = [str(temp_path)]
        temp_counts = [local_rows]
        total_rows = local_rows
    logger.info("rank=%s 全局布局: total_rows=%s temp_files=%s", rank, total_rows, len(temp_paths))

    # 7.2 计算全局置换（所有 rank 用相同 seed → 相同 perm）
    keep_ratio = args.sample_ratio if 0 < args.sample_ratio < 1.0 else None
    if args.global_shuffle or keep_ratio is not None:
        perm = compute_global_permutation(total_rows, args.seed, keep_ratio if keep_ratio else 1.0)
        if keep_ratio is not None:
            total_rows = len(perm)
        logger.info("rank=%s 全局置换: keep=%s (ratio=%s)", rank, total_rows, args.sample_ratio)
    else:
        perm = np.arange(total_rows)
    n_keep = len(perm)

    # 7.3 计算分片数与各 rank 负责的分片
    rows_per_shard = args.rows_per_shard
    n_shards = max(1, math.ceil(n_keep / rows_per_shard))
    logger.info("rank=%s 分片数=%s rows_per_shard=%s", rank, n_shards, rows_per_shard)

    # 清理旧分片
    if rank == 0:
        for old in output_dir.glob("part-*.parquet"):
            old.unlink()
    if comm is not None:
        comm.Barrier()

    # 7.4 各 rank 拥有的分片索引
    my_shard_indices = [i for i in range(n_shards) if i % world_size == rank]
    logger.info("rank=%s 负责分片数=%s: %s", rank, len(my_shard_indices), my_shard_indices[:5])

    # ------------------------------------------------------------------
    # 7.5 批量读取：收集本 rank 所有分片需要的 global_indices，按 temp file
    #     一次性读取，修复"每个 shard 全量重读 temp file"的性能瓶颈。
    #     原实现：每个 shard 调 read_rows_from_temp_files → pq.read_table(全文件)
    #     → O(n_shards × n_temp_files) 次全文件 I/O（2.3M行/2330shards/16ranks
    #     ≈ 每 rank 2336 次全文件读 = ~1TB I/O/rank = 数小时）。
    #     修复后：每 rank 对每个 temp file 只读一次，I/O 降低 ~n_shards 倍。
    # ------------------------------------------------------------------
    shard_ranges: List[Tuple[int, np.ndarray]] = []  # [(shard_idx, global_indices)]
    for shard_idx in my_shard_indices:
        lo = shard_idx * rows_per_shard
        hi = min((shard_idx + 1) * rows_per_shard, n_keep)
        global_indices = perm[lo:hi]
        if len(global_indices) == 0:
            continue
        shard_ranges.append((shard_idx, global_indices))

    # 批量读取所有所需行：{global_idx: row_dict}（每个 temp file 只读一次）
    global_row_cache = batch_read_rows_from_temp_files(
        temp_paths, temp_counts, shard_ranges, rank,
    )

    shard_output_paths: List[str] = []
    for shard_idx, global_indices in shard_ranges:
        # 从 cache 按 perm 顺序取行，组装 DataFrame
        rows = [global_row_cache[int(g)] for g in global_indices]
        df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        out_path = output_dir / f"part-{shard_idx:05d}.parquet"
        write_shard(df, out_path)
        shard_output_paths.append(str(out_path))
        logger.info("rank=%s 写出 %s rows=%s", rank, out_path, len(df))
        # 释放已写完 shard 的行，降低峰值内存
        for g in global_indices:
            global_row_cache.pop(int(g), None)

    # ------------------------------------------------------------------
    # 8. 汇总所有分片路径，rank0 写 file_list.json
    # ------------------------------------------------------------------
    if comm is not None:
        all_shard_paths_lists = comm.gather(shard_output_paths, root=0)
        if rank == 0:
            all_shard_paths: List[str] = []
            for paths in all_shard_paths_lists:
                all_shard_paths.extend(paths)
            all_shard_paths.sort()
        else:
            all_shard_paths = []
    else:
        all_shard_paths = sorted(shard_output_paths)

    if rank == 0:
        # 写 file_list.json 到 blended 目录
        blended_json_path = Path(args.blended_json_path)
        blended_json_path.parent.mkdir(parents=True, exist_ok=True)
        with blended_json_path.open("w", encoding="utf-8") as f:
            json.dump(all_shard_paths, f, indent=2, ensure_ascii=False)
        logger.info("file_list.json 写出: %s (%s files)", blended_json_path, len(all_shard_paths))

        # 写 build_summary.json
        summary = {
            "cot_dir": str(cot_dir),
            "bench_parquet": str(bench_parquet),
            "caption_dir": str(caption_dir) if caption_dir else None,
            "pid2sid": str(pid2sid_path),
            "product_pid2sid": str(product_pid2sid_path),
            "output_dir": str(output_dir),
            "blended_json_path": str(blended_json_path),
            "rows_per_shard": rows_per_shard,
            "n_shards": n_shards,
            "total_rows": total_rows,
            "global_shuffle": args.global_shuffle,
            "sample_ratio": args.sample_ratio,
            "seed": args.seed,
            "split_target": args.split_target,
            "target_max_len": TARGET_MAX_LEN,
            "allow_cot_non_cot_user_overlap": args.allow_cot_non_cot_user_overlap,
            "enable_cot_split_filter": args.enable_cot_split_filter,
            "enable_non_cot_rec": args.enable_non_cot_rec,
            "enable_video_rec": args.enable_video_rec,
            "enable_ad_rec": args.enable_ad_rec,
            "enable_product_rec": args.enable_product_rec,
            "enable_sid2caption": args.enable_sid2caption,
            "enable_caption2sid": args.enable_caption2sid,
            "sid2caption_size": args.sid2caption_size,
            "caption2sid_size": args.caption2sid_size,
            # product 域 caption（0723 新增）
            "product_caption_dir": args.product_caption_dir or None,
            "enable_product_sid2caption": args.enable_product_sid2caption,
            "enable_product_caption2sid": args.enable_product_caption2sid,
            "product_sid2caption_size": args.product_sid2caption_size,
            "product_caption2sid_size": args.product_caption2sid_size,
            # CoT history 对齐策略：CoT 样本的 history / target 优先从 bench row
            # 读取原始 pid（经 pid2sid 过滤），与非 CoT 样本完全一致；caption
            # 过滤丢弃的 pid 也会出现在最终序列中。metadata 中 video_hist_sid_count
            # / ad_hist_sid_count 标明 ad 域两段式 history 的分段位置。
            "cot_history_aligned_with_bench": True,
            "cot_missing_bench_count": cot_missing_bench,
            "sources": {
                "rec_cot": SOURCE_REC_COT,
                "video_rec": SOURCE_VIDEO_REC,
                "ad_rec": SOURCE_AD_REC,
                "product_rec": SOURCE_PRODUCT_REC,
                "caption_to_sid": SOURCE_CAPTION_TO_SID,
                "item_understand": SOURCE_ITEM_UNDERSTAND,
            },
        }
        summary_path = output_dir / "build_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("build_summary.json 写出: %s", summary_path)

        print("\n================ Step 5 (data_format) Summary ================")
        print(f"  total_rows      : {total_rows}")
        print(f"  n_shards       : {n_shards}")
        print(f"  rows_per_shard : {rows_per_shard}")
        print(f"  global_shuffle : {args.global_shuffle}")
        print(f"  sample_ratio   : {args.sample_ratio}")
        print(f"  split_target   : {args.split_target}")
        print(f"  target_max_len : {TARGET_MAX_LEN}")
        print(f"  output_dir     : {output_dir}")
        print(f"  blended_json   : {blended_json_path}")
        print(f"  elapsed        : {time.time() - t_total:.1f}s")
        print("================================================================")

        # ------------------------------------------------------------------
        # 8.1 统计每个 source 的样本条数 + 横向字符柱状图
        # ------------------------------------------------------------------
        # 读取所有输出分片的 `source` 列，统计各 source 条数并打印柱状图。
        # 只读 source 列（不读 messages 等大列），速度快。
        if all_shard_paths:
            logger.info("rank=0 开始统计 source 分布（读取 %s 个分片）...", len(all_shard_paths))
            source_counts = _count_sources_from_shards(all_shard_paths)
            _print_source_bar_chart(source_counts)
            # 写 source_counts.json 到输出目录
            source_counts_path = output_dir / "source_counts.json"
            with source_counts_path.open("w", encoding="utf-8") as f:
                json.dump(source_counts, f, indent=2, ensure_ascii=False)
            logger.info("source_counts.json 写出: %s", source_counts_path)

    # ------------------------------------------------------------------
    # 9. 清理临时文件（rank0 负责）
    # ------------------------------------------------------------------
    if comm is not None:
        comm.Barrier()
    if rank == 0 and not args.keep_temp:
        for tp in temp_dir.glob("temp_rank*.parquet"):
            try:
                tp.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass
        logger.info("rank=0 清理临时文件完成")


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 5: 合成 RecIF CoT SFT 数据集（MPI 并行 + 全局 shuffle）"
    )
    # ---- 输入 ----
    parser.add_argument("--cot_dir", type=str, required=True,
                        help="CoT 输入目录（step2 2_cot 或 step4 4_filter 的输出），含 part-*.parquet")
    parser.add_argument("--bench_parquet", type=str, required=True,
                        help="onerec_bench_release.parquet 路径，用于构造非 CoT 推荐样本 + split 过滤")
    parser.add_argument("--caption_dir", type=str, default="",
                        help="caption parquet 目录（pid + dense_caption），用于 caption2sid / sid2caption")
    parser.add_argument("--pid2sid", type=str, required=True,
                        help="video_ad_pid2sid.parquet 路径")
    parser.add_argument("--product_pid2sid", type=str, default="",
                        help="product_pid2sid.parquet 路径；CoT 含 product 域"
                             "（2_cot_merged）或启用 product_rec 时**必填**。"
                             "缺失时若有 product CoT 样本会立刻报错（无 fallback）。")

    # ---- 输出 ----
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，存放 part-*.parquet 分片")
    parser.add_argument("--blended_json_path", type=str, required=True,
                        help="file_list.json 输出路径（绝对路径，通常在 OpenOneRec/pretrain/examples/blended/ 下）")
    parser.add_argument("--temp_dir", type=str, default="",
                        help="临时文件目录，默认 <output_dir>/_temp")
    parser.add_argument("--keep_temp", action="store_true",
                        help="保留临时文件（调试用）")

    # ---- 任务开关 ----
    parser.add_argument("--enable_non_cot_rec", type=int, default=1,
                        help="是否构造非 CoT 推荐样本（video/ad/product），0=关闭")
    parser.add_argument("--enable_video_rec", type=int, default=1,
                        help="是否构造 video 非 CoT 样本")
    parser.add_argument("--enable_ad_rec", type=int, default=1,
                        help="是否构造 ad 非 CoT 样本")
    parser.add_argument("--enable_product_rec", type=int, default=1,
                        help="是否构造 product 非 CoT 样本")
    parser.add_argument("--enable_sid2caption", type=int, default=1,
                        help="是否构造 sid2caption 样本")
    parser.add_argument("--enable_caption2sid", type=int, default=1,
                        help="是否构造 caption2sid 样本")

    # ---- 用户重叠 ----
    parser.add_argument("--allow_cot_non_cot_user_overlap", type=int, default=0,
                        help="1=允许 CoT 与非 CoT 共享用户（非 CoT 使用全部用户）；0=非 CoT 仅使用 CoT 以外用户")

    # ---- CoT split 过滤 ----
    parser.add_argument("--enable_cot_split_filter", type=int, default=1,
                        help="1=通过 uid join bench 的 split 列过滤 CoT（跳过 split!=0 的 uid）；0=不过滤")

    # ---- CoT target 来源（0717 新增）----
    parser.add_argument("--cot_target_source", type=str, default="bench",
                        choices=["bench", "cot"],
                        help="CoT 样本的 target_pids 来源。"
                             "'bench'（默认）= 用 bench row 的原始 target（target_video_pid / target_ad_pid），"
                             "与非 CoT 样本对齐。"
                             "'cot' = 用 CoT sample 自带的 target_pids（来自 4a_filter_ppl 输出，"
                             "已做 per-target 筛选，5_data_format 直接用，不再做额外筛选）。"
                             "★ 0717 新链路：4a 做 per-target filter（输出 helpful targets），"
                             "5_data_format 用 cot_target_source=cot 直接消费")

    # ---- 全局 shuffle + 采样 ----
    parser.add_argument("--global_shuffle", type=int, default=1,
                        help="1=全局 shuffle；0=保持原顺序")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="全局采样比例，(0, 1]")
    parser.add_argument("--rows_per_shard", type=int, default=ROWS_PER_SHARD,
                        help="每个输出分片的行数，默认 1000")

    # ---- target 上限 + 拆分 ----
    parser.add_argument("--target_max_len", type=int, default=TARGET_MAX_LEN,
                        help="每条推荐样本保留的 target 数量上限（作用于 CoT 与非 CoT 推荐样本）。"
                             "split_target=0 时：每条样本的 target 截断到前 target_max_len 个（单条多 target）；"
                             "split_target!=0 时：每条原始样本最多拆成 target_max_len 条单 target 样本，"
                             "原始 target 数 > target_max_len 时用 per-sample 确定性 RNG 随机采样。"
                             "默认 3。caption 任务本身就是单 pid，不受此参数影响")
    parser.add_argument("--split_target", type=int, default=0,
                        help="target 拆分开关（作用于 CoT 与非 CoT 推荐样本）。"
                             "0=不拆分，每条样本保留前 target_max_len 个 target（单条多 target）；"
                             "非0=拆分，每条原始样本按 target 拆成最多 target_max_len 条，每条 1 个 target，"
                             "原始 target 数 > target_max_len 时用 per-sample 确定性 RNG 随机采样。"
                             "caption 任务本身就是单 pid，不受此参数影响")

    # ---- caption 任务采样 ----
    parser.add_argument("--sid2caption_size", type=int, default=260000,
                        help="sid2caption 目标样本数，0=不构造")
    parser.add_argument("--caption2sid_size", type=int, default=260000,
                        help="caption2sid 目标样本数，0=不构造")
    parser.add_argument("--shuffle_caption_files", type=int, default=1,
                        help="1=打乱 caption 分片顺序")

    # ---- product caption 任务采样（0723 新增，独立于 video/ad caption）----
    parser.add_argument("--product_caption_dir", type=str, default="",
                        help="product 域 caption 目录（goods_pid2caption，列 new_pid + caption），"
                             "启用 product caption 任务时必填")
    parser.add_argument("--product_caption_column", type=str, default="caption",
                        help="product caption 列名（goods_pid2caption 默认 'caption'）")
    parser.add_argument("--product_pid_column", type=str, default="new_pid",
                        help="product pid 列名（goods_pid2caption 用 new_pid，★ 不要用 old_pid）")
    parser.add_argument("--enable_product_sid2caption", type=int, default=0,
                        help="是否构造 product 域 sid2caption 样本（0723 新增，默认 0=关闭）")
    parser.add_argument("--enable_product_caption2sid", type=int, default=0,
                        help="是否构造 product 域 caption2sid 样本（0723 新增，默认 0=关闭）")
    parser.add_argument("--product_sid2caption_size", type=int, default=260000,
                        help="product sid2caption 目标样本数，0=不构造")
    parser.add_argument("--product_caption2sid_size", type=int, default=260000,
                        help="product caption2sid 目标样本数，0=不构造")

    # ---- 其他 ----
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--siddoc_mode", type=int, default=0,
                        help="siddoc 模式（仅作用于非 CoT rec 样本，caption 任务不受影响）："
                             "0=原格式（rec 样本，assistant=target）；"
                             "1=纯 siddoc 替换（rec 样本全部变 siddoc，assistant=带标签的历史序列，"
                             "无 target/prediction prompt，对历史序列做自回归补 input CE loss）；"
                             "2=mix：每个用户每域同时造一条原格式 rec + 一条 siddoc，混在一起训练，"
                             "rec 学 target 预测、siddoc 补历史自回归。"
                             "CoT 样本仅在 mode==1 时变 siddoc 格式，mode==2 时 CoT 保持原格式")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
