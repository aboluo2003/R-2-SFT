#!/usr/bin/env python3
"""Step 4a (filter_ppl) — 基于 PPL 增益的 RecIF CoT 全局筛选 (MPI, CPU)。

输入: 3a join 输出 (per-target parquet, 含 logp_y_given_x / logp_y_given_x_cot /
      num_target_tokens + step2 列 cot / history_pids / history_captions /
      target_pids / target_captions / history_meta)
输出: per-sample parquet (合并 target), 可直接作为 step5 的 --cot_dir

指标 (per-target):
  log_ppl_x    = -logp_y_given_x     / num_target_tokens
  log_ppl_cot  = -logp_y_given_x_cot / num_target_tokens
  diff         = log_ppl_cot - log_ppl_x   (越负 = CoT 越有益)

筛选门:
  卫生门 (hygiene):  log_ppl_x     < --max_log_ppl_x     (丢掉连裸预测都给不出概率的噪声 target)
  CoT 有效门 (help): diff          < --max_diff           (越负越保留)
  CoT target ppl门:  log_ppl_cot   < --max_log_ppl_cot    (0723 新增, 越低越好; 不传=不启用)
  按域 CoT ppl门:     log_ppl_cot   < --max_log_ppl_cot_by_domain 中对应 target_domain 的阈值
                         (例如 ad:3.3932,video:4.6238,product:3.8894；未配置的域回退到 --max_log_ppl_cot)

模式:
  annotate_all  不指定任何门 / limit        -> 全保留, 仅追加 filter_score / filter_meta
  threshold      --max_diff / --max_log_ppl_x / --max_log_ppl_cot -> 按门 + merge_rule 保留
  top_k          --limit / --limit_ratio       -> 卫生门后按聚合 diff 升序取 top-N sample
  (top_k + threshold 组合: 先门后 top-N)

合并规则 (--merge_rule, 默认 any):
  any       任一 target 通过两门即保留该 sample (宽松, 最大化样本)
  all       所有 valid target 都通过 CoT 门才保留 (严格)
  majority  多数 valid target 通过 CoT 门
  mean      聚合 filter_score < --max_diff 才保留

MPI 策略: 按 hash(sample_key) % world_size 分片, 每个 rank 拥有完整 sample 的所有 target,
本地完成打分 / 聚合 / 生存判定。top-k 模式下 allgather (filter_score, sample_key) 候选,
rank0 全局排序取 top-N 后 bcast kept 集合。最终各 rank 只写自己分片内命中的 sample。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

logger = logging.getLogger(__name__)

# 输入要读的列 (3a join 输出的子集; 跳过 ppl_y_given_* / eval_meta 节省内存)
INPUT_COLUMNS: Tuple[str, ...] = (
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
    "num_target_tokens",
)

# 输出列 (per-sample, 合并后; step5 直接消费 step2 列, 忽略 filter_* 列)
OUTPUT_COLUMNS: Tuple[str, ...] = (
    "sample_key",
    "uid",
    "target_domain",
    "cot",
    "history_pids",
    "history_captions",
    "target_pids",
    "target_captions",
    "history_meta",
    "filter_score",
    "filter_rank",
    "filter_meta",
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def hash_shard(sample_key: Any, world_size: int) -> int:
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


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isinf(v) or math.isnan(v):
        return float("nan")
    return v


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# per-target 打分
# ---------------------------------------------------------------------------
@dataclass
class TargetInfo:
    target_idx: int
    target_pid: int
    logp_x: float
    logp_cot: float
    num_tokens: int
    log_ppl_x: float
    log_ppl_cot: float
    diff: float
    valid: bool       # scorable 且通过卫生门
    helpful: bool     # valid 且通过 CoT 有效门
    max_log_ppl_cot_threshold: Optional[float] = None  # 实际使用的 log_ppl_cot 阈值（支持按域）


def parse_domain_thresholds(spec: Optional[str]) -> Dict[str, float]:
    """解析按域阈值配置：'ad:3.39,video=4.62,product:3.89'。"""
    out: Dict[str, float] = {}
    if spec is None or str(spec).strip() == "":
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, val = part.split(":", 1)
        elif "=" in part:
            key, val = part.split("=", 1)
        else:
            raise ValueError(
                f"非法 --max_log_ppl_cot_by_domain 片段: {part!r}，期望 domain:value"
            )
        key = key.strip()
        if not key:
            raise ValueError(f"非法 --max_log_ppl_cot_by_domain 片段: {part!r}，domain 为空")
        try:
            out[key] = float(val.strip())
        except ValueError as exc:
            raise ValueError(
                f"非法 --max_log_ppl_cot_by_domain 阈值: {part!r}"
            ) from exc
    return out


def get_max_log_ppl_cot_for_domain(args: argparse.Namespace, target_domain: Any) -> Optional[float]:
    domain_map = getattr(args, "max_log_ppl_cot_by_domain_map", {}) or {}
    domain = str(target_domain) if target_domain is not None else ""
    if domain in domain_map:
        return domain_map[domain]
    return args.max_log_ppl_cot


def make_target_info(row: Dict[str, Any], args: argparse.Namespace) -> TargetInfo:
    logp_x = to_float(row.get("logp_y_given_x"))
    logp_cot = to_float(row.get("logp_y_given_x_cot"))
    nt = int(row.get("num_target_tokens") or 0)

    if nt > 0 and not math.isnan(logp_x):
        log_ppl_x = -logp_x / nt
    else:
        log_ppl_x = float("nan")
    if nt > 0 and not math.isnan(logp_cot):
        log_ppl_cot = -logp_cot / nt
    else:
        log_ppl_cot = float("nan")
    if not (math.isnan(log_ppl_x) or math.isnan(log_ppl_cot)):
        diff = log_ppl_cot - log_ppl_x
    else:
        diff = float("nan")

    scorable = not (math.isnan(log_ppl_x) or math.isnan(log_ppl_cot))
    if args.max_log_ppl_x is not None:
        valid = scorable and (not math.isnan(log_ppl_x)) and (log_ppl_x < args.max_log_ppl_x)
    else:
        valid = scorable
    if args.max_diff is not None:
        helpful = valid and (not math.isnan(diff)) and (diff < args.max_diff)
    else:
        helpful = valid
    max_log_ppl_cot_threshold = get_max_log_ppl_cot_for_domain(args, row.get("target_domain"))
    # 0723 新增：log_ppl_cot 门（PPL(y|x,cot) 越低越好，CoT 让 target ppl 低 = 好）
    # 0729 新增：支持按 target_domain 配不同阈值；未配置域回退到全局 --max_log_ppl_cot。
    if max_log_ppl_cot_threshold is not None:
        helpful = helpful and (not math.isnan(log_ppl_cot)) and (log_ppl_cot < max_log_ppl_cot_threshold)

    return TargetInfo(
        target_idx=int(row.get("target_idx") or 0),
        target_pid=int(row.get("target_pid") or 0),
        logp_x=logp_x,
        logp_cot=logp_cot,
        num_tokens=nt,
        log_ppl_x=log_ppl_x,
        log_ppl_cot=log_ppl_cot,
        diff=diff,
        valid=valid,
        helpful=helpful,
        max_log_ppl_cot_threshold=max_log_ppl_cot_threshold,
    )


# ---------------------------------------------------------------------------
# per-sample 聚合 + 生存判定
# ---------------------------------------------------------------------------
@dataclass
class SampleInfo:
    sample_key: str
    uid: Any
    target_domain: Any
    cot: Any
    history_pids: Any
    history_captions: Any
    target_pids: Any
    target_captions: Any
    history_meta: Any
    targets: List[TargetInfo] = field(default_factory=list)
    filter_score: float = float("nan")
    survived: bool = False
    filter_meta: str = ""


def compute_filter_score(sample: SampleInfo) -> float:
    valid_diffs = [t.diff for t in sample.targets if t.valid and not math.isnan(t.diff)]
    if not valid_diffs:
        return float("nan")
    return float(np.mean(valid_diffs))


def compute_survival(sample: SampleInfo, args: argparse.Namespace, mode: str) -> bool:
    """sample 是否存活（至少 1 个 helpful target）。

    0717 改动：统一 per-target filter 语义，survived = any helpful。
    - annotate_all 模式：helpful = valid（不门控 diff），survived = any valid
    - threshold / topk 模式：helpful = valid 且 diff < max_diff，survived = any helpful
    - 全部 target 都不 helpful → 丢 sample
    （旧 merge_rule any/all/majority/mean 已清理，per-target filter 下输出只保留 helpful target）
    """
    if mode == "annotate_all":
        return any(t.helpful for t in sample.targets)
    helpful_targets = [t for t in sample.targets if t.helpful]
    return len(helpful_targets) >= 1


def build_filter_meta(sample: SampleInfo, args: argparse.Namespace, mode: str) -> str:
    valid_targets = [t for t in sample.targets if t.valid]
    helpful_targets = [t for t in valid_targets if t.helpful]
    valid_diffs = [t.diff for t in valid_targets if not math.isnan(t.diff)]
    meta = {
        "mode": mode,
        "merge_rule": args.merge_rule,
        "max_diff": args.max_diff,
        "max_log_ppl_x": args.max_log_ppl_x,
        "max_log_ppl_cot": args.max_log_ppl_cot,
        "max_log_ppl_cot_by_domain": getattr(args, "max_log_ppl_cot_by_domain_map", {}) or {},
        "num_targets": len(sample.targets),
        "num_valid": len(valid_targets),
        "num_helpful": len(helpful_targets),
        "diff_mean": sample.filter_score,
        "diff_min": float(min(valid_diffs)) if valid_diffs else None,
        "diff_max": float(max(valid_diffs)) if valid_diffs else None,
        "targets": [
            {
                "target_idx": t.target_idx,
                "target_pid": t.target_pid,
                "log_ppl_x": t.log_ppl_x,
                "log_ppl_cot": t.log_ppl_cot,
                "diff": t.diff,
                "max_log_ppl_cot_threshold": t.max_log_ppl_cot_threshold,
                "num_tokens": t.num_tokens,
                "valid": t.valid,
                "helpful": t.helpful,
            }
            for t in sorted(sample.targets, key=lambda x: x.target_idx)
        ],
    }
    return json.dumps(_clean_for_json(meta), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase 1: per-rank 扫描 + 打分
# ---------------------------------------------------------------------------
def scan_and_score(
    files: Sequence[str], rank: int, world_size: int, args: argparse.Namespace, mode: str
) -> Dict[str, SampleInfo]:
    samples: Dict[str, SampleInfo] = {}
    files_read_ok = 0
    files_read_failed = 0
    rows_scanned = 0

    for f in files:
        try:
            df = pd.read_parquet(f, columns=list(INPUT_COLUMNS))
            files_read_ok += 1
        except Exception as exc:
            files_read_failed += 1
            logger.warning("[rank %d] 读文件失败 %s: %s", rank, f, exc)
            continue
        if len(df) == 0:
            continue
        if world_size > 1:
            df = df[_shard_mask(df["sample_key"].astype(str), world_size, rank)]
        rows_scanned += len(df)
        for row in df.to_dict(orient="records"):
            sk = str(row.get("sample_key"))
            if sk not in samples:
                samples[sk] = SampleInfo(
                    sample_key=sk,
                    uid=row.get("uid"),
                    target_domain=row.get("target_domain"),
                    cot=row.get("cot"),
                    history_pids=row.get("history_pids"),
                    history_captions=row.get("history_captions"),
                    target_pids=row.get("target_pids"),
                    target_captions=row.get("target_captions"),
                    history_meta=row.get("history_meta"),
                )
            samples[sk].targets.append(make_target_info(row, args))

    # 打分 + 生存
    for sample in samples.values():
        sample.filter_score = compute_filter_score(sample)
        sample.survived = compute_survival(sample, args, mode)
        sample.filter_meta = build_filter_meta(sample, args, mode)

    logger.info(
        "[rank %d] 扫描 %d 文件 (ok=%d fail=%d), %d 行, %d samples (survived=%d)",
        rank, len(files), files_read_ok, files_read_failed, rows_scanned,
        len(samples), sum(1 for s in samples.values() if s.survived),
    )
    return samples


# ---------------------------------------------------------------------------
# Phase 2: 全局 top-k
# ---------------------------------------------------------------------------
def _sort_key(item: Tuple[float, str]) -> Tuple[Any, ...]:
    score, sk = item
    if score != score:  # NaN
        return (1, 0.0, sk)
    return (0, score, sk)


def global_topk(
    samples: Dict[str, SampleInfo],
    args: argparse.Namespace,
    comm: Any,
    rank: int,
    world_size: int,
) -> Tuple[Optional[Dict[str, int]], int, int]:
    """返回 (rank_map, global_survived_count, effective_limit)。非 top-k 模式返回 (None, count, 0)。"""
    local_survived = [s for s in samples.values() if s.survived]
    local_count = len(local_survived)

    if world_size > 1:
        global_count = comm.allreduce(local_count, op=MPI.SUM)
    else:
        global_count = local_count

    use_topk = (args.limit and args.limit > 0) or (args.limit_ratio and args.limit_ratio > 0)
    if not use_topk:
        return None, global_count, 0

    if args.limit_ratio and args.limit_ratio > 0:
        effective_limit = int(global_count * args.limit_ratio)
    else:
        effective_limit = int(args.limit or 0)
    effective_limit = max(0, min(effective_limit, global_count))

    local_cands = [(s.filter_score, s.sample_key) for s in local_survived]
    if world_size > 1:
        all_cands = comm.allgather(local_cands)
    else:
        all_cands = [local_cands]

    if rank == 0:
        flat = [c for sub in all_cands for c in sub]
        flat.sort(key=_sort_key)
        kept = flat[:effective_limit]
        rank_map = {sk: i + 1 for i, (_, sk) in enumerate(kept)}
    else:
        rank_map = {}

    if world_size > 1:
        rank_map = comm.bcast(rank_map, root=0)
    return rank_map, global_count, effective_limit


# ---------------------------------------------------------------------------
# Phase 3: per-rank 写出 (合并 target -> per-sample)
# ---------------------------------------------------------------------------
def write_outputs(
    samples: Dict[str, SampleInfo],
    rank_map: Optional[Dict[str, int]],
    output: str,
    rank: int,
    rows_per_file: int,
) -> Dict[str, Any]:
    survived = [s for s in samples.values() if s.survived]
    if rank_map is not None:
        survived = [s for s in survived if s.sample_key in rank_map]
    # 按 filter_rank 排序 (top-k) 或按 filter_score 升序 (threshold/annotate)
    if rank_map is not None:
        survived.sort(key=lambda s: (rank_map[s.sample_key], s.sample_key))
    else:
        survived.sort(key=lambda s: (_sort_key((s.filter_score, s.sample_key))))

    rows: List[Dict[str, Any]] = []
    for s in survived:
        fr = rank_map.get(s.sample_key, -1) if rank_map is not None else -1
        # 0717: target_pids 只保留 helpful target（per-target filter），
        # target_captions 不输出（5_data_format SID 路径不用 captions）。
        # target_pid 来自 s.targets（per-target，bench target，来自新 1_preprocess bench 模式）
        helpful_target_pids = [t.target_pid for t in s.targets if t.helpful]
        rows.append({
            "sample_key": s.sample_key,
            "uid": s.uid,
            "target_domain": s.target_domain,
            "cot": s.cot,
            "history_pids": s.history_pids,
            "history_captions": s.history_captions,
            "target_pids": helpful_target_pids,
            "target_captions": [],
            "history_meta": s.history_meta,
            "filter_score": float(s.filter_score) if not math.isnan(s.filter_score) else float("nan"),
            "filter_rank": int(fr),
            "filter_meta": s.filter_meta,
        })

    os.makedirs(output, exist_ok=True)
    parts_written = 0
    for chunk_idx, start in enumerate(range(0, len(rows), rows_per_file), start=1):
        chunk = rows[start: start + rows_per_file]
        df = pd.DataFrame(chunk, columns=list(OUTPUT_COLUMNS))
        path = os.path.join(output, f"part-{rank:05d}_{chunk_idx:05d}.parquet")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        parts_written += 1

    return {
        "rank": rank,
        "survived_samples": len(survived),
        "parts_written": parts_written,
    }


# ---------------------------------------------------------------------------
# 统计汇总
# ---------------------------------------------------------------------------
def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "p25": None, "p50": None, "p75": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
    }


def collect_local_stats(samples: Dict[str, SampleInfo], rank_map: Optional[Dict[str, int]]) -> Dict[str, Any]:
    survived = [s for s in samples.values() if s.survived]
    if rank_map is not None:
        survived = [s for s in survived if s.sample_key in rank_map]
    diffs: List[float] = []
    log_ppl_xs: List[float] = []
    log_ppl_cots: List[float] = []
    for s in survived:
        for t in s.targets:
            if t.valid and not math.isnan(t.diff):
                diffs.append(t.diff)
                if not math.isnan(t.log_ppl_x):
                    log_ppl_xs.append(t.log_ppl_x)
                if not math.isnan(t.log_ppl_cot):
                    log_ppl_cots.append(t.log_ppl_cot)
    return {
        "survived_samples": len(survived),
        "survived_valid_targets": len(diffs),
        "diffs": diffs,
        "log_ppl_xs": log_ppl_xs,
        "log_ppl_cots": log_ppl_cots,
    }


def aggregate_stats(
    local_stats: Dict[str, Any],
    comm: Any,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    if comm is None or world_size <= 1:
        diffs = np.array(local_stats["diffs"], dtype=np.float64)
        lpx = np.array(local_stats["log_ppl_xs"], dtype=np.float64)
        lpc = np.array(local_stats["log_ppl_cots"], dtype=np.float64)
        return {
            "survived_samples": local_stats["survived_samples"],
            "survived_valid_targets": local_stats["survived_valid_targets"],
            "diff_stats": _percentiles(diffs),
            "log_ppl_x_stats": _percentiles(lpx),
            "log_ppl_cot_stats": _percentiles(lpc),
        }
    all_stats = comm.allgather(local_stats)
    total_samples = sum(s["survived_samples"] for s in all_stats)
    total_targets = sum(s["survived_valid_targets"] for s in all_stats)
    diffs = np.array([d for s in all_stats for d in s["diffs"]], dtype=np.float64)
    lpx = np.array([d for s in all_stats for d in s["log_ppl_xs"]], dtype=np.float64)
    lpc = np.array([d for s in all_stats for d in s["log_ppl_cots"]], dtype=np.float64)
    return {
        "survived_samples": total_samples,
        "survived_valid_targets": total_targets,
        "diff_stats": _percentiles(diffs),
        "log_ppl_x_stats": _percentiles(lpx),
        "log_ppl_cot_stats": _percentiles(lpc),
    }


# ---------------------------------------------------------------------------
# 模式判定 / 参数
# ---------------------------------------------------------------------------
def resolve_mode(args: argparse.Namespace) -> str:
    use_topk = (args.limit and args.limit > 0) or (args.limit_ratio and args.limit_ratio > 0)
    use_threshold = (
        (args.max_diff is not None)
        or (args.max_log_ppl_x is not None)
        or (args.max_log_ppl_cot is not None)
        or bool(getattr(args, "max_log_ppl_cot_by_domain_map", {}) or {})
    )
    if use_topk:
        return "top_k"
    if use_threshold:
        return "threshold"
    return "annotate_all"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Step 4a: 基于 PPL 增益的 RecIF CoT 全局筛选 (MPI 并行)"
    )
    p.add_argument("--input", required=True, help="3a join 输出目录 (per-target parquet)")
    p.add_argument("--output", required=True, help="输出目录 (per-sample parquet)")
    p.add_argument("--max_diff", type=float, default=None,
                   help="CoT 有效门: diff < max_diff 视为 helpful (不传=不启用该门)")
    p.add_argument("--max_log_ppl_x", type=float, default=None,
                   help="卫生门: log_ppl_x < max_log_ppl_x 视为 valid (不传=不启用该门)")
    p.add_argument("--max_log_ppl_cot", type=float, default=None,
                   help="CoT target ppl 门 (0723 新增): log_ppl_cot < max_log_ppl_cot 视为 helpful "
                        "(越低越好, 不传=不启用该门；按域阈值未覆盖的域会回退到该全局阈值)")
    p.add_argument("--max_log_ppl_cot_by_domain", type=str, default="",
                   help="按 target_domain 设置 log_ppl_cot 阈值，格式如 "
                        "'ad:3.3932,video:4.6238,product:3.8894'；"
                        "命中的域优先于 --max_log_ppl_cot，未命中域回退到全局阈值")
    p.add_argument("--merge_rule", choices=["any", "all", "majority", "mean"], default="any",
                   help="per-target -> per-sample 合并规则 (默认 any)")
    p.add_argument("--limit", type=int, default=0,
                   help="保留全局 top-k 的 k 值; 0=不启用 top-k")
    p.add_argument("--limit_ratio", type=float, default=0.0,
                   help="保留全局 top-k 的比例 (0,1]; 优先于 --limit")
    p.add_argument("--rows_per_output_file", type=int, default=5000,
                   help="每个输出 parquet 行数 (默认 5000)")
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()
    args.max_log_ppl_cot_by_domain_map = parse_domain_thresholds(args.max_log_ppl_cot_by_domain)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    else:
        comm = None
        rank = 0
        world_size = 1
        logger.warning("mpi4py 不可用, 退化为单进程模式。")

    mode = resolve_mode(args)
    files = list_parquet_files(args.input)

    logger.info(
        "[rank %d/%d] mode=%s input=%s output=%s files=%d max_diff=%s max_log_ppl_x=%s max_log_ppl_cot=%s max_log_ppl_cot_by_domain=%s merge=%s limit=%s ratio=%s",
        rank, world_size, mode, args.input, args.output, len(files),
        args.max_diff, args.max_log_ppl_x, args.max_log_ppl_cot,
        args.max_log_ppl_cot_by_domain_map, args.merge_rule, args.limit, args.limit_ratio,
    )

    # Phase 1
    samples = scan_and_score(files, rank, world_size, args, mode)

    # Phase 2
    rank_map, global_survived, effective_limit = global_topk(samples, args, comm, rank, world_size)

    # Phase 3
    write_stats = write_outputs(samples, rank_map, args.output, rank, args.rows_per_output_file)

    # 统计
    local_stats = collect_local_stats(samples, rank_map)
    agg_stats = aggregate_stats(local_stats, comm, rank, world_size)

    if rank == 0:
        summary = {
            "mode": mode,
            "max_diff": args.max_diff,
            "max_log_ppl_x": args.max_log_ppl_x,
            "max_log_ppl_cot": args.max_log_ppl_cot,
            "max_log_ppl_cot_by_domain": args.max_log_ppl_cot_by_domain_map,
            "merge_rule": args.merge_rule,
            "limit": args.limit,
            "limit_ratio": args.limit_ratio,
            "effective_limit": effective_limit,
            "global_survived_before_topk": global_survived,
            "global_kept_samples": agg_stats["survived_samples"],
            "global_kept_valid_targets": agg_stats["survived_valid_targets"],
            "diff_stats": agg_stats["diff_stats"],
            "log_ppl_x_stats": agg_stats["log_ppl_x_stats"],
            "log_ppl_cot_stats": agg_stats["log_ppl_cot_stats"],
        }
        print("=" * 60)
        print("Step 4a (filter_ppl) 完成")
        print("=" * 60)
        print(f"mode                       : {mode}")
        print(f"max_diff / max_log_ppl_x   : {args.max_diff} / {args.max_log_ppl_x}")
        print(f"max_log_ppl_cot            : {args.max_log_ppl_cot}")
        print(f"max_log_ppl_cot_by_domain  : {args.max_log_ppl_cot_by_domain_map}")
        print(f"merge_rule                 : {args.merge_rule}")
        print(f"limit / limit_ratio        : {args.limit} / {args.limit_ratio}")
        print(f"effective_limit (top-k)    : {effective_limit}")
        print(f"global_survived (pre top-k): {global_survived}")
        print(f"global_kept_samples        : {agg_stats['survived_samples']}")
        print(f"global_kept_valid_targets   : {agg_stats['survived_valid_targets']}")
        ds = agg_stats["diff_stats"]
        print(f"diff (kept valid targets)  : count={ds['count']} "
              f"min={ds['min']} max={ds['max']} mean={ds['mean']} "
              f"p25={ds['p25']} p50={ds['p50']} p75={ds['p75']}")
        ls = agg_stats["log_ppl_x_stats"]
        print(f"log_ppl_x (kept valid)      : count={ls['count']} "
              f"min={ls['min']} max={ls['max']} mean={ls['mean']} "
              f"p50={ls['p50']}")
        lcs = agg_stats["log_ppl_cot_stats"]
        print(f"log_ppl_cot (kept valid)    : count={lcs['count']} "
              f"min={lcs['min']} max={lcs['max']} mean={lcs['mean']} "
              f"p50={lcs['p50']}")
        print(f"output_dir                 : {args.output}")
        print(f"summary                    : {os.path.join(args.output, 'filter_summary.json')}")
        print("=" * 60)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "filter_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_clean_for_json(summary), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
