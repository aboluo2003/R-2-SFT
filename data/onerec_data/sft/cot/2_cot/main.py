#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (cot): 为 RecIF CoT 消融数据合成推理文本（单机多线程版）。

参考：OpenOneRec/data/onerec_data/sft/deactived/reason/1_cot/main.py（Amazon CoT 生成）

输入：Step 1 (1_history_text) 的输出 parquet
  schema: uid, target_domain, history_pids, history_captions,
          target_pids, target_captions, history_meta

输出：合成的 cot 文本，保留原始字段
  schema: sample_key, uid, target_domain, cot,
          history_pids, history_captions, target_pids, target_captions, history_meta

关键设计：
  - 单机运行，不使用 MPI（API 并发受限，多机无收益）。
  - 多线程并发调用 LLM（ThreadPoolExecutor）。
  - 历史以 caption 文本形式（每行一条）喂给 LLM，不提供任何 pid / sid。
  - 不截断历史序列和 caption（完整喂给 API）。
  - 中文 prompt（prompt.txt），告知目标域（短视频 / 广告）。
  - 断点续跑：按 sample_key=f"{uid}_{target_domain}" 跳过已处理样本。
  - token 用量统计：每次 flush / 结束时输出。
  - --max_samples 限制本次运行处理的新样本数（0 = 不限制）。

用法：
  python3 main.py --input_dir <step1 输出> --output_dir <step2 输出> \
      --api_config_path /root/gemini3.json --num_workers 64

本地干跑（不调用 API）：
  python3 main.py --dry_run --input_dir ... --output_dir ... --num_workers 1 --max_samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 复用同目录上层 cot/api.py 的 LLM 客户端
#   2_cot/main.py -> parent (cot/) 下的 api.py
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent          # .../cot/2_cot
COT_DIR = CURRENT_DIR.parent                             # .../cot
if str(COT_DIR) not in sys.path:
    sys.path.insert(0, str(COT_DIR))

from api import TokenStats, init_llm_client  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 默认路径 / 常量
# -----------------------------------------------------------------------------
PREFIX = COT_DIR.parents[3]  # OpenOneRec/ root (cot/2_cot → OpenOneRec is 4 levels up)
DEFAULT_INPUT_DIR = PREFIX / "data" / "RecIF-cot" / "steps" / "1_history_text"
DEFAULT_OUTPUT_DIR = PREFIX / "data" / "RecIF-cot" / "steps" / "2_cot"
DEFAULT_PROMPT_PATH = CURRENT_DIR / "prompt.txt"
DEFAULT_PROMPT_AD_PATH = CURRENT_DIR / "prompt_ad.txt"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_PROMPT = (
    "你是一位严谨的推荐系统分析专家，擅长从用户历史交互中推断偏好"
    "并预测下一个可能感兴趣的内容，避免臆造证据。"
)

DOMAIN_DESC = {
    "video": "短视频",
    "ad": "广告",
}

# ad 域两段式 history 的分段标签（与 prompt_ad.txt 中的说明对齐）
AD_VIDEO_SEGMENT_LABEL = "【用户观看过的视频】"
AD_AD_SEGMENT_LABEL = "【用户点击过的广告视频】"

OUTPUT_COLUMNS = (
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

PART_FILE_PATTERN = re.compile(r"^part-(\d{6})\.parquet$")


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_part_files(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    files = [
        p for p in output_dir.iterdir()
        if p.is_file() and PART_FILE_PATTERN.match(p.name)
    ]
    return sorted(files)


def next_part_index(output_dir: Path) -> int:
    max_index = -1
    for path in list_part_files(output_dir):
        m = PART_FILE_PATTERN.match(path.name)
        if m:
            max_index = max(max_index, int(m.group(1)))
    return max_index + 1


def load_processed_keys(output_dir: Path) -> set:
    """读取已写出 part 文件的 sample_key 列，用于断点续跑。"""
    processed: set = set()
    for path in list_part_files(output_dir):
        try:
            table = pq.read_table(path, columns=["sample_key"])
            for v in table.column("sample_key").to_pylist():
                if v is not None and str(v).strip():
                    processed.add(str(v))
        except Exception as exc:
            logger.warning("读取已处理 sample_key 失败: %s, error=%s", path, exc)
    return processed


def write_records(
    output_dir: Path,
    records: List[Dict[str, Any]],
    *,
    part_index: int,
) -> int:
    if not records:
        return part_index
    ensure_dir(output_dir)
    frame = pd.DataFrame(records)
    frame = frame.reindex(columns=list(OUTPUT_COLUMNS))
    tmp_path = output_dir / f"part-{part_index:06d}.parquet.tmp"
    final_path = output_dir / f"part-{part_index:06d}.parquet"
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(final_path)
    logger.info("写出 %s rows=%s", final_path, len(frame))
    return part_index + 1


def _to_list(value: Any) -> list:
    """把 numpy array / pandas list / None 安全转成 list。"""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_input_samples(input_dir: Path) -> pd.DataFrame:
    """读取 step1 输出的所有 part 文件，合并成一个 DataFrame。"""
    files = sorted(input_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"未在 {input_dir} 找到 step1 输出 parquet")
    logger.info("读取 step1 输入: %s 个文件", len(files))
    frames = []
    for f in tqdm(files, desc="loading step1"):
        frames.append(pq.read_table(f).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    logger.info("合并后样本数: %s", len(df))
    return df


def to_samples(df: pd.DataFrame) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        uid = int(row["uid"])
        target_domain = str(row["target_domain"])
        sample_key = f"{uid}_{target_domain}"
        samples.append({
            "sample_key": sample_key,
            "uid": uid,
            "target_domain": target_domain,
            "history_pids": _to_list(row.get("history_pids")),
            "history_captions": _to_list(row.get("history_captions")),
            "target_pids": _to_list(row.get("target_pids")),
            "target_captions": _to_list(row.get("target_captions")),
            "history_meta": row.get("history_meta"),
        })
    return samples


# -----------------------------------------------------------------------------
# prompt 构造（不截断）
# -----------------------------------------------------------------------------
def load_prompt_template(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def _captions_to_bullets(captions: Sequence[str]) -> str:
    """把 captions 拼成「每行一条」的 bullet 文本，空 caption 跳过。"""
    lines: List[str] = []
    for cap in captions:
        if not cap or not str(cap).strip():
            continue
        text = re.sub(r"\s+", " ", str(cap)).strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines).strip()


def build_history_text(captions: Sequence[str]) -> str:
    """video 域：把 history_captions 拼成「每行一条」的文本，不截断。"""
    if not captions:
        return ""
    return _captions_to_bullets(captions)


def _parse_history_meta(meta_str: Optional[str]) -> Dict[str, Any]:
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


def build_history_text_ad(
    captions: Sequence[str],
    history_meta: Optional[str],
) -> str:
    """ad 域：根据 history_meta 把 captions 拆成「视频观看」+「广告点击」两段。

    step1 (1_history_text) 的输出把 ad 域 history 拼成
    longview_video_captions + ad_click_captions，分段长度存在 history_meta 的
    kept_video_hist_len / kept_ad_hist_len 里。这里按长度还原两段并加标签，
    让外部模型能区分视频观看历史和广告点击历史。

    若 history_meta 缺失或长度为 0，退化为把全部 captions 当广告点击历史。
    """
    if not captions:
        return ""
    meta = _parse_history_meta(history_meta)
    video_len = int(meta.get("kept_video_hist_len", 0) or 0)
    ad_len = int(meta.get("kept_ad_hist_len", 0) or 0)

    if video_len or ad_len:
        video_caps = list(captions[:video_len])
        ad_caps = list(captions[video_len:video_len + ad_len])
        # 若 meta 长度不够覆盖全部 captions，剩余的归到 ad 段（防御性）
        if video_len + ad_len < len(captions):
            ad_caps.extend(captions[video_len + ad_len:])
    else:
        # meta 缺失：全部当广告点击历史（保留旧行为，但加标签）
        video_caps = []
        ad_caps = list(captions)

    parts: List[str] = []
    video_text = _captions_to_bullets(video_caps)
    if video_text:
        parts.append(f"{AD_VIDEO_SEGMENT_LABEL}\n{video_text}")
    ad_text = _captions_to_bullets(ad_caps)
    if ad_text:
        parts.append(f"{AD_AD_SEGMENT_LABEL}\n{ad_text}")
    return "\n\n".join(parts).strip()


def make_dry_run_cot(target_domain: str, n_history: int) -> str:
    domain_desc = DOMAIN_DESC.get(target_domain, target_domain)
    return (
        f"（dry run 占位）根据用户历史中 {n_history} 条交互记录，"
        f"可以看出该用户具有相对稳定的偏好模式。"
        f"综合这些观察到的行为，推断该用户接下来在{domain_desc}域"
        f"可能继续关注与其历史兴趣相符的内容类型。"
    )


def build_record(
    sample: Dict[str, Any],
    prompt_templates: Dict[str, str],
    client: Optional[Any],
    token_stats: TokenStats,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    target_domain = sample["target_domain"]
    domain_desc = DOMAIN_DESC.get(target_domain, target_domain)
    history_captions = sample.get("history_captions") or []

    # 按目标域选择 history 文本构造方式：
    #   video    → 单段 bullet（build_history_text）
    #   ad       → 两段标签（build_history_text_ad，区分视频观看 / 广告点击）
    if target_domain == "ad":
        history_text = build_history_text_ad(
            history_captions, sample.get("history_meta")
        )
    else:
        history_text = build_history_text(history_captions)

    # 按目标域选择 prompt 模板（ad 用 prompt_ad.txt，其余用 prompt.txt）
    prompt_template = prompt_templates.get(
        target_domain, prompt_templates.get("video", "")
    )

    if not history_text:
        # 没有可用历史文本，直接返回空 cot
        cot = ""
    elif args.dry_run:
        cot = make_dry_run_cot(target_domain, len(history_captions))
    else:
        assert client is not None
        prompt = prompt_template.format(
            target_domain_desc=domain_desc,
            history_text=history_text,
        )
        cot = client.call(
            prompt,
            verbose=False,
            token_stats=token_stats,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.api_max_retries,
            retry_initial_delay=args.retry_initial_delay,
            retry_backoff_factor=args.retry_backoff_factor,
            retry_max_delay=args.retry_max_delay,
        ).strip()

    return {
        "sample_key": sample["sample_key"],
        "uid": sample["uid"],
        "target_domain": target_domain,
        "cot": cot,
        "history_pids": sample.get("history_pids"),
        "history_captions": sample.get("history_captions"),
        "target_pids": sample.get("target_pids"),
        "target_captions": sample.get("target_captions"),
        "history_meta": sample.get("history_meta"),
    }


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 1) 读取 step1 输出
    df = load_input_samples(input_dir)
    all_samples = to_samples(df)

    # 2) 断点续跑：过滤已处理样本
    processed_keys = load_processed_keys(output_dir)
    logger.info("已处理 sample_key 数=%s", len(processed_keys))
    pending = [s for s in all_samples if s["sample_key"] not in processed_keys]
    logger.info("待处理样本=%s / 总样本=%s", len(pending), len(all_samples))

    # 3) --max_samples 限制本次运行处理的新样本数（0 = 不限制）
    if args.max_samples > 0 and len(pending) > args.max_samples:
        pending = pending[: args.max_samples]
        logger.info("截断到 max_samples=%s", len(pending))

    # 3b) --target_domain_filter：仅处理指定域（用于 ad 域单独重造）
    if args.target_domain_filter:
        before = len(pending)
        pending = [s for s in pending if s["target_domain"] == args.target_domain_filter]
        logger.info(
            "target_domain_filter=%s 过滤后: %s → %s",
            args.target_domain_filter, before, len(pending),
        )

    if not pending:
        logger.info("没有待处理样本，退出。")
        print("\n================ Step 2 (cot) Summary ================")
        print(f"  total samples   : 0 (全部已处理)")
        print(f"  output_dir      : {output_dir}")
        print("================================================================")
        return

    # 4) 加载 prompt（按域分别加载）
    prompt_templates: Dict[str, str] = {}
    video_prompt_path = Path(args.prompt_path)
    prompt_templates["video"] = load_prompt_template(video_prompt_path)
    logger.info("加载 video prompt: %s", video_prompt_path)

    ad_prompt_path = Path(args.prompt_path_ad)
    if ad_prompt_path.exists():
        prompt_templates["ad"] = load_prompt_template(ad_prompt_path)
        logger.info("加载 ad prompt: %s", ad_prompt_path)
    else:
        # ad prompt 缺失时退化为 video prompt（保持兼容）
        prompt_templates["ad"] = prompt_templates["video"]
        logger.warning(
            "ad prompt 不存在 %s，ad 域退化为 video prompt", ad_prompt_path,
        )

    # 5) 初始化 LLM client
    need_llm = any(len(s.get("history_captions") or []) > 0 for s in pending)
    client = None
    token_stats = TokenStats()
    if need_llm and not args.dry_run:
        if not args.api_config_path:
            raise ValueError("--api_config_path 是必填项（除非 --dry_run）")
        client = init_llm_client(
            model=args.model,
            api_config_path=args.api_config_path,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.api_max_retries,
            retry_initial_delay=args.retry_initial_delay,
            retry_backoff_factor=args.retry_backoff_factor,
            retry_max_delay=args.retry_max_delay,
            timeout=args.timeout,
        )

    # 6) 并发处理
    buffer: List[Dict[str, Any]] = []
    part_index = next_part_index(output_dir)
    failed = 0
    checkpoint_interval = args.checkpoint_interval
    t_start = time.time()

    try:
        progress = tqdm(total=len(pending), desc="cot", unit="row")
        if args.num_workers <= 1:
            for idx, sample in enumerate(pending, start=1):
                try:
                    rec = build_record(sample, prompt_templates, client, token_stats, args)
                    if rec is not None:
                        buffer.append(rec)
                except Exception as exc:
                    failed += 1
                    logger.exception("处理 sample_key=%s 失败: %s", sample["sample_key"], exc)
                finally:
                    progress.update(1)
                if len(buffer) >= checkpoint_interval:
                    part_index = write_records(output_dir, buffer, part_index=part_index)
                    buffer = []
                    _log_progress(idx, len(pending), failed, token_stats, t_start, args.dry_run)
                if idx % 100 == 0 or idx == len(pending):
                    _log_progress(idx, len(pending), failed, token_stats, t_start, args.dry_run)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_key = {
                    executor.submit(build_record, s, prompt_templates, client, token_stats, args): s["sample_key"]
                    for s in pending
                }
                for idx, future in enumerate(as_completed(future_to_key), start=1):
                    key = future_to_key[future]
                    try:
                        rec = future.result()
                        if rec is not None:
                            buffer.append(rec)
                    except Exception as exc:
                        failed += 1
                        logger.exception("处理 sample_key=%s 失败: %s", key, exc)
                    finally:
                        progress.update(1)
                    if len(buffer) >= checkpoint_interval:
                        part_index = write_records(output_dir, buffer, part_index=part_index)
                        buffer = []
                        _log_progress(idx, len(pending), failed, token_stats, t_start, args.dry_run)
                    if idx % 100 == 0 or idx == len(pending):
                        _log_progress(idx, len(pending), failed, token_stats, t_start, args.dry_run)
    finally:
        progress.close()
        if client is not None:
            client.close()

    if buffer:
        part_index = write_records(output_dir, buffer, part_index=part_index)

    # 7) 汇总
    elapsed = time.time() - t_start
    written = len(pending) - failed
    print("\n================ Step 2 (cot) Summary ================")
    print(f"  total processed : {len(pending)}")
    print(f"  written         : {written}")
    print(f"  failed          : {failed}")
    print(f"  elapsed         : {elapsed:.1f}s")
    if args.target_domain_filter:
        print(f"  domain filter   : {args.target_domain_filter}")
    print(f"  prompt (video)  : {args.prompt_path}")
    print(f"  prompt (ad)     : {args.prompt_path_ad}")
    if not args.dry_run:
        snap = token_stats.snapshot()
        print(f"  token usage     :")
        print(f"    request_count : {snap.get('request_count', 0)}")
        print(f"    prompt_tokens  : {snap.get('prompt_tokens', 0)}")
        print(f"    response_tokens: {snap.get('response_tokens', 0)}")
        print(f"    thoughts_tokens: {snap.get('thoughts_tokens', 0)}")
        print(f"    cached_tokens  : {snap.get('cached_tokens', 0)}")
        print(f"    total_tokens   : {snap.get('total_tokens', 0)}")
    print(f"  output_dir      : {output_dir}")
    print("================================================================")


def _log_progress(
    idx: int,
    total: int,
    failed: int,
    token_stats: TokenStats,
    t_start: float,
    dry_run: bool,
) -> None:
    elapsed = time.time() - t_start
    rate = idx / elapsed if elapsed > 0 else 0.0
    snap = token_stats.snapshot() if not dry_run else {}
    logger.info(
        "进度 %s/%s failed=%s rate=%.1f rows/s elapsed=%.1fs%s",
        idx, total, failed, rate, elapsed,
        "" if dry_run else f" tokens={snap.get('total_tokens', 0)}",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 2: RecIF CoT 合成（单机多线程）")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Step 1 (1_history_text) 输出目录")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Step 2 输出目录")
    parser.add_argument("--prompt_path", type=str, default=str(DEFAULT_PROMPT_PATH),
                        help="video 域 prompt 模板文件路径")
    parser.add_argument("--prompt_path_ad", type=str, default=str(DEFAULT_PROMPT_AD_PATH),
                        help="ad 域 prompt 模板文件路径（区分视频观看 / 广告点击两段历史）。"
                             "文件不存在时退化为 --prompt_path")
    parser.add_argument("--target_domain_filter", type=str, default="",
                        choices=["", "video", "ad"],
                        help="仅处理指定目标域的样本。"
                             "空字符串=不过滤（处理 video+ad），"
                             "'video'=仅 video，'ad'=仅 ad")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="LLM 模型名")
    parser.add_argument("--api_config_path", type=str, default=None,
                        help="API 配置 JSON 路径（dry_run 时可省略）")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="系统提示词")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="本次运行最多处理的新样本数，0 表示不限制")
    parser.add_argument("--checkpoint_interval", type=int, default=64,
                        help="每 N 条 flush 一次 parquet")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--reasoning_effort", type=str, default="none",
                        help="支持的 reasoning effort 级别")
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--retry_initial_delay", type=float, default=1.0)
    parser.add_argument("--retry_backoff_factor", type=float, default=2.0)
    parser.add_argument("--retry_max_delay", type=float, default=30.0)
    parser.add_argument("--timeout", type=int, default=None,
                        help="API 超时（秒）")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="并发线程数")
    parser.add_argument("--dry_run", action="store_true",
                        help="跳过 API 调用，写占位 cot")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
