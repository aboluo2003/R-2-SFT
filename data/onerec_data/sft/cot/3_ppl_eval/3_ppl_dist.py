#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a (ppl_dist): 读取 2_eval_ppl.py 的输出 parquet，
在控制台以纯文本形式可视化 PPL 分布，用于 CoT 筛选决策。

背景：
  - 2_eval_ppl.py 对每个样本算了两个 PPL：
      ppl_y_given_x      = exp(mean_j -log P(target_j | input,         target_<j))
      ppl_y_given_x_cot = exp(mean_j -log P(target_j | input+cot,     target_<j))
    同时给出原始 log 概率和：
      logp_y_given_x      = sum_j log P(target_j | input)
      logp_y_given_x_cot  = sum_j log P(target_j | input+cot)
    以及打分 token 数 num_target_tokens。
  - 本脚本计算并可视化三项核心指标（均为 per-token NLL，即 log(PPL)）：
      log_ppl_x    = log(ppl_y_given_x)      = -logp_y_given_x      / num_target_tokens
      log_ppl_cot  = log(ppl_y_given_x_cot)  = -logp_y_given_x_cot  / num_target_tokens
      diff         = log_ppl_cot - log_ppl_x
                     = (logp_y_given_x - logp_y_given_x_cot) / num_target_tokens
    diff < 0  ⟹  加上 CoT 后 per-token NLL 下降，CoT 有帮助；
    diff > 0  ⟹  CoT 反而让答案更难预测，CoT 有害。
  - 这三项分布是下游 CoT 筛选（4_filter）的主要依据。

可视化内容（全部纯文本，适配服务器无图形界面）：
  1. log_ppl_x    分布直方图 + 分位数表
  2. log_ppl_cot  分布直方图 + 分位数表
  3. diff (log_ppl_cot - log_ppl_x) 分布直方图 + 分位数表
  4. CoT 帮助比例 / 不同阈值下的保留率
  5. base 难度 × CoT 效果 联合表（判断 CoT 在哪些难度区间有用 + 联合保留率）
  6. 按 target_idx 分组（CoT 对第 1/2/3 个 target 的帮助差异）
  7. CoT 长度 vs CoT 效果（越长越有用还是越有害 + Spearman）
  8. 按 sample_key (uid+domain) 的 CoT 一致性（user-level vs target-level 筛选决策）
  9. num_target_tokens 分布
  10. log_ppl_x vs log_ppl_cot 2D 热力图（文本）
  11. 按 target_domain (video / ad) 分组对比
  12. 按 representation_type (sid / caption) 分组对比（若混合）
  13. diff 极端样本示例（最有帮助 / 最有害的几条）
  14. 相关性：Pearson + Spearman（base 难度 vs CoT 效果、CoT 长度 vs diff 等）

用法（示例）：
  # sid 表示（默认输出在 ppl_eval_sid/）
  python3 3_ppl_dist.py \
      --input_uri .../data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sid \
      --max_samples 30000

  python3 3_ppl_dist.py \
      --input_uri .../data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_sid \
      --max_samples 30000

  python3 data/onerec_data/sft/cot/3_ppl_eval/3_ppl_dist.py \
      --input_uri .../data/RecIF-cot/steps/3_ppl_eval/ppl_eval_sft_cot \
      --max_samples 30000

  # caption 表示
  python3 3_ppl_dist.py \
      --input_uri .../ppl_eval_caption \
      --max_samples 30000

  # 本地调试
  python3 3_ppl_dist.py --input_uri ./ppl_eval_sid --max_samples 1000 --bins 30

可选：
  --save_csv PATH     把读取到的样本（含三项指标）落盘成 csv，方便后续二次分析
  --save_filtered PATH  把 diff < --diff_threshold 的样本落盘成 parquet（CoT 筛选候选）
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pa_fs


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 读取时只拉需要的列，减少 IO（缺失列会自动被忽略）
NEEDED_COLUMNS = (
    "ppl_sample_key",
    "sample_key",
    "uid",
    "target_domain",
    "target_idx",
    "target_pid",
    "representation_type",
    "logp_y_given_x",
    "logp_y_given_x_cot",
    "ppl_y_given_x",
    "ppl_y_given_x_cot",
    "num_target_tokens",
    "cot",
    "target_text",
)

# 文本直方图用的 Unicode block 字符（从空到满）
_BLOCK_FRACTIONS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


# ---------------------------------------------------------------------------
# 文件系统（复用 2_eval_ppl.py 的逻辑，兼容本地 + viewfs/hdfs）
# ---------------------------------------------------------------------------

def _is_parquet_data_file(path: str) -> bool:
    name = os.path.basename(path)
    return name.endswith(".parquet") or name.startswith("part-")


def get_filesystem_and_path(uri: str) -> Tuple[Any, str]:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        filesystem, path = pa_fs.FileSystem.from_uri(uri)
        return filesystem, path
    local_path = parsed.path if parsed.scheme == "file" else uri
    return pa_fs.LocalFileSystem(), os.path.abspath(local_path)


def list_parquet_files(uri: str) -> Tuple[Any, List[str]]:
    filesystem, base_path = get_filesystem_and_path(uri)
    info = filesystem.get_file_info(base_path)
    if isinstance(info, list):
        info = info[0] if info else None
    if info is None or info.type == pa_fs.FileType.NotFound:
        raise FileNotFoundError(f"输入路径不存在: {uri}")
    if info.is_file:
        if _is_parquet_data_file(info.path):
            return filesystem, [info.path]
        raise FileNotFoundError(f"输入文件不是 parquet: {uri}")
    selector = pa_fs.FileSelector(info.path, recursive=True)
    infos = filesystem.get_file_info(selector)
    files = [i.path for i in infos if i.is_file and _is_parquet_data_file(i.path)]
    files.sort()
    if not files:
        raise FileNotFoundError(f"未在 {uri} 找到 parquet 文件")
    return filesystem, files


def read_parquet_samples(
    filesystem: Any,
    files: List[str],
    max_samples: int,
    columns: Optional[Sequence[str]],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """逐文件读取 parquet，累计到 max_samples 行就停。返回 (df, 实际读到的文件列表)。"""
    frames: List[pd.DataFrame] = []
    total = 0
    read_files: List[str] = []
    for fp in files:
        with filesystem.open_input_file(fp) as f:
            table = pq.read_table(f, columns=columns)
        df = table.to_pandas()
        frames.append(df)
        read_files.append(fp)
        total += len(df)
        if verbose:
            print(f"  [read] {os.path.basename(fp)} rows={len(df)} cum={total}", flush=True)
        if total >= max_samples:
            break
    if not frames:
        return pd.DataFrame(), read_files
    big = pd.concat(frames, ignore_index=True)
    if len(big) > max_samples:
        big = big.iloc[:max_samples].reset_index(drop=True)
    return big, read_files


# ---------------------------------------------------------------------------
# 纯文本可视化工具
# ---------------------------------------------------------------------------

def _format_num(x: float, width: int, decimals: int) -> str:
    """格式化数字到定宽，处理 nan/inf。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan".rjust(width)
    if math.isinf(x):
        return ("+inf" if x > 0 else "-inf").rjust(width)
    return f"{x:.{decimals}f}".rjust(width)


def _bar_str(frac: float, max_width: int) -> str:
    """根据比例画一段长度为 max_width 的 block 字符串。"""
    if max_width <= 0:
        return ""
    if frac <= 0:
        return ""
    if frac >= 1:
        return "█" * max_width
    total = frac * max_width
    full = int(total)
    remainder = total - full
    # remainder in [0,1) 映射到 8 个分数 block
    idx = int(round(remainder * 8))
    if idx >= 8:
        full += 1
        idx = 0
    bar = "█" * full
    if full < max_width:
        bar += _BLOCK_FRACTIONS[idx] if idx > 0 else ""
    # 截断到 max_width
    return bar[:max_width]


def text_histogram(
    values: np.ndarray,
    bins: int = 40,
    title: str = "",
    width: int = 50,
    decimals: int = 3,
    percentiles: Sequence[float] = (1, 5, 25, 50, 75, 90, 95, 99),
) -> None:
    """打印一个水平直方图 + 分位数摘要。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    print(title or "histogram")
    if n == 0:
        print("  (no finite values)")
        print()
        return

    mean = float(values.mean())
    std = float(values.std())
    vmin = float(values.min())
    vmax = float(values.max())
    pcts = np.percentile(values, list(percentiles))

    stats_line = (
        f"  n={n}  mean={mean:.4f}  std={std:.4f}  "
        f"min={vmin:.4f}  max={vmax:.4f}"
    )
    print(stats_line)
    pct_str = "  ".join(f"p{int(p)}={v:.4f}" for p, v in zip(percentiles, pcts))
    print(f"  {pct_str}")
    print()

    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max()
    label_w = max(8, decimals * 2 + 6)  # 容纳 "[xx.xxx, xx.xxx)"

    for i in range(len(counts)):
        lo = edges[i]
        hi = edges[i + 1]
        c = int(counts[i])
        pct = c / n * 100
        bar_w = int(c / max_count * width) if max_count > 0 else 0
        # 分数 block 精度
        frac = c / max_count if max_count > 0 else 0
        bar = _bar_str(frac, width)
        lo_s = _format_num(lo, decimals + 5, decimals)
        hi_s = _format_num(hi, decimals + 5, decimals)
        print(
            f"  [{lo_s}, {hi_s}) "
            f"{c:>8d}  {pct:>6.2f}%  {bar}"
        )
    print()


def text_percentile_table(
    series_map: Sequence[Tuple[str, np.ndarray]],
    percentiles: Sequence[float] = (0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9),
    decimals: int = 4,
) -> None:
    """打印多列分位数对比表。series_map = [(name, values), ...]"""
    names = [name for name, _ in series_map]
    name_w = max(len(n) for n in names)
    header = "percentile".ljust(name_w) + "  " + "  ".join(f"{n:>12s}" for n in names)
    print(header)
    print("-" * len(header))
    for p in percentiles:
        row = f"p{p:<{name_w - 1}g}" if False else f"p{str(p):<{name_w - 1}s}"
        # 上面那行写法有点绕，直接拼
        label = f"p{p}"
        row = label.ljust(name_w)
        for _, vals in series_map:
            v = np.asarray(vals, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                row += "  " + "nan".rjust(12)
            else:
                row += "  " + f"{np.percentile(v, p):>12.{decimals}f}"
        print(row)
    print()


def text_heatmap_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    bins: int = 30,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    width: int = 60,
) -> None:
    """打印 xs vs ys 的 2D 文本热力图。每行一个 y bin，列方向用 block 密度表示。

    布局：
        y_label │ <density grid>
        y_label │ ...
                └──── x axis with ticks ────
                  xlabel
    密度字符: 空格=0, ▏▎▍▌▋▊▉█ = 从低到高（按 cell 占 max 的比例分 8 档）
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    print(title or f"{ylabel} vs {xlabel} 2D heatmap")
    if xs.size == 0:
        print("  (no finite values)")
        print()
        return

    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    max_h = H.max()
    # y 轴标签宽度：容纳 "[yy.yy, yy.yy)" 这种区间更直观
    y_label_w = 22

    print(f"  n={xs.size}  bins={bins}x{bins}  max_cell={int(max_h)}")
    # 密度图例
    legend = "  密度: " + " ".join(
        f"{ch}={i}" for i, ch in enumerate(_BLOCK_FRACTIONS) if i > 0
    )
    print(legend)
    print()

    grid_prefix_w = y_label_w + 1  # 标签 + 分隔符

    # 从上到下打印 y bin（高 y 在上）
    for j in range(bins - 1, -1, -1):
        lo = yedges[j]
        hi = yedges[j + 1]
        label = f"[{lo:.3f},{hi:.3f})"
        row = f"{label:>{y_label_w}s} │"
        for i in range(bins):
            c = H[i, j]
            if max_h <= 0 or c == 0:
                row += " "
            else:
                frac = c / max_h
                idx = min(8, max(1, int(round(frac * 8))))
                row += _BLOCK_FRACTIONS[idx]
        print(row)

    # 底部 x 轴线
    print(" " * grid_prefix_w + "└" + "─" * bins)

    # 底部 x 轴刻度：均匀取若干刻度，自适应数量避免标签重叠
    # 目标：相邻 tick 间距 >= 5 字符，标签 ~4 字符居中后不重叠
    n_ticks = max(2, min(6, bins // 5 + 1))
    tick_positions = np.linspace(0, bins - 1, n_ticks)
    tick_chars = [" "] * bins
    tick_labels = []
    for tp in tick_positions:
        ti = int(round(tp))
        ti = max(0, min(bins - 1, ti))
        center = (xedges[ti] + xedges[ti + 1]) / 2
        tick_chars[ti] = "+"
        tick_labels.append((ti, f"{center:.2f}"))
    print(" " * grid_prefix_w + " " + "".join(tick_chars))

    # 刻度数字行：每个 label 左对齐贴到 tick 位置
    label_line = list(" " * (grid_prefix_w + 1 + bins))
    for ti, lab in tick_labels:
        start = grid_prefix_w + 1 + ti
        # 把 label 居中放在 tick 上方（往左偏一点避免溢出）
        offset = max(0, start - len(lab) // 2)
        for k, ch in enumerate(lab):
            pos = offset + k
            if 0 <= pos < len(label_line):
                label_line[pos] = ch
    print("".join(label_line).rstrip())

    print(f"{'':>{grid_prefix_w}} {xlabel}")
    print(f"  {xlabel} range: [{xedges[0]:.3f}, {xedges[-1]:.3f}]"
          f"   {ylabel} range: [{yedges[0]:.3f}, {yedges[-1]:.3f}]")
    print()


def _section(title: str) -> None:
    bar = "═" * 78
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank 实现（处理 ties），纯 numpy，避免 scipy 依赖。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Spearman 秩相关；返回 None 表示无法计算（样本<3 或方差为 0）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """从 step2 输出列计算 log_ppl_x / log_ppl_cot / diff。"""
    out = df.copy()

    logp_x = pd.to_numeric(out["logp_y_given_x"], errors="coerce").to_numpy(dtype=np.float64)
    logp_cot = pd.to_numeric(out["logp_y_given_x_cot"], errors="coerce").to_numpy(dtype=np.float64)
    n_tok = pd.to_numeric(out["num_target_tokens"], errors="coerce").to_numpy(dtype=np.float64)

    # per-token NLL = -logp / num_tokens = log(ppl)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ppl_x = np.where(
            (n_tok > 0) & np.isfinite(logp_x),
            -logp_x / np.where(n_tok > 0, n_tok, 1),
            np.nan,
        )
        log_ppl_cot = np.where(
            (n_tok > 0) & np.isfinite(logp_cot),
            -logp_cot / np.where(n_tok > 0, n_tok, 1),
            np.nan,
        )
        diff = log_ppl_cot - log_ppl_x

    out["log_ppl_x"] = log_ppl_x
    out["log_ppl_cot"] = log_ppl_cot
    out["diff"] = diff  # < 0 表示 CoT 有帮助
    # 保留 num_target_tokens 的数值版本（float，含 nan）便于画图
    out["num_target_tokens_num"] = n_tok
    return out


def _summarize_validity(df: pd.DataFrame) -> None:
    n = len(df)
    valid_x = np.isfinite(df["log_ppl_x"].to_numpy())
    valid_cot = np.isfinite(df["log_ppl_cot"].to_numpy())
    valid_both = valid_x & valid_cot
    n_tok = df["num_target_tokens_num"].to_numpy() if "num_target_tokens_num" in df else None
    n_zero_tok = int((n_tok == 0).sum()) if n_tok is not None else 0
    print(f"  总样本数           : {n}")
    print(f"  log_ppl_x  有效    : {int(valid_x.sum())}  ({valid_x.mean()*100:.2f}%)")
    print(f"  log_ppl_cot 有效   : {int(valid_cot.sum())}  ({valid_cot.mean()*100:.2f}%)")
    print(f"  两者均有效         : {int(valid_both.sum())}  ({valid_both.mean()*100:.2f}%)")
    print(f"  num_target_tokens=0: {n_zero_tok}")
    if "target_domain" in df.columns:
        vc = df["target_domain"].value_counts(dropna=False)
        parts = [f"{k}={int(v)}" for k, v in vc.items()]
        print(f"  target_domain 分布 : {', '.join(parts)}  (total={int(vc.sum())})")
    if "representation_type" in df.columns:
        vc = df["representation_type"].value_counts(dropna=False)
        parts = [f"{k}={int(v)}" for k, v in vc.items()]
        print(f"  representation 分布: {', '.join(parts)}  (total={int(vc.sum())})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 3a PPL 分布可视化（纯文本控制台，适配服务器无 GUI 环境）",
    )
    parser.add_argument("--input_uri", type=str, required=True,
                        help="2_eval_ppl.py 的输出目录或单文件（本地或 viewfs://hdfs://）")
    parser.add_argument("--max_samples", type=int, default=30000,
                        help="最多读取多少条样本（默认 30000）")
    parser.add_argument("--bins", type=int, default=40,
                        help="直方图 bin 数（默认 40）")
    parser.add_argument("--hist_width", type=int, default=50,
                        help="直方图条形最大宽度（字符，默认 50）")
    parser.add_argument("--heatmap_bins", type=int, default=30,
                        help="2D 热力图 bin 数（默认 30）")
    parser.add_argument("--show_examples", type=int, default=3,
                        help="每个极端展示多少条样本（默认 3）")
    parser.add_argument("--save_csv", type=str, default="",
                        help="可选：把读取到的样本（含三项指标）落盘成 csv")
    parser.add_argument("--save_filtered", type=str, default="",
                        help="可选：把 diff < --diff_threshold 的样本落盘成 parquet")
    parser.add_argument("--diff_threshold", type=float, default=0.0,
                        help="CoT 筛选阈值：diff < threshold 视为 CoT 有帮助（默认 0.0）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印读取进度等更详细日志")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    print("=" * 78)
    print("  Step 3a (ppl_dist) — PPL 分布可视化")
    print("=" * 78)
    print(f"  input_uri      : {args.input_uri}")
    print(f"  max_samples    : {args.max_samples}")
    print(f"  bins           : {args.bins}")
    print(f"  diff_threshold : {args.diff_threshold}")
    print()

    # 1. 列出 parquet 文件
    filesystem, files = list_parquet_files(args.input_uri)
    print(f"  发现 parquet 文件: {len(files)} 个")
    if args.verbose:
        for fp in files[:5]:
            print(f"    - {fp}")
        if len(files) > 5:
            print(f"    ... (共 {len(files)} 个)")
    if not files:
        return

    # 2. 探测可用列（只读第一个文件做 schema 探测）
    with filesystem.open_input_file(files[0]) as f:
        schema = pq.read_schema(f)
    available = set(schema.names)
    columns = [c for c in NEEDED_COLUMNS if c in available]
    missing = [c for c in NEEDED_COLUMNS if c not in available]
    if missing:
        print(f"  [warn] 以下列在输入中不存在，将被忽略: {missing}")
    # 强制需要的列
    required = {"logp_y_given_x", "logp_y_given_x_cot", "num_target_tokens"}
    if not required.issubset(available):
        raise SystemExit(
            f"输入缺少必需列: {required - available}。请确认 --input_uri 指向 2_eval_ppl.py 的输出。"
        )

    # 3. 读取样本
    print(f"  读取样本中 (最多 {args.max_samples} 条) ...")
    df, read_files = read_parquet_samples(
        filesystem, files, args.max_samples, columns, verbose=args.verbose,
    )
    print(f"  实际读取: {len(df)} 条，来自 {len(read_files)} 个文件")
    print()

    if df.empty:
        print("  [error] 读到的样本为空，退出。")
        return

    # 4. 计算指标
    df = compute_metrics(df)

    _section("样本有效性 & 元信息")
    _summarize_validity(df)

    # 取有效子集做分布
    valid = df[np.isfinite(df["log_ppl_x"].to_numpy()) & np.isfinite(df["log_ppl_cot"].to_numpy())].copy()
    if valid.empty:
        print("  [error] 没有任何 (log_ppl_x, log_ppl_cot) 同时有效的样本，无法画分布。")
        return

    log_ppl_x = valid["log_ppl_x"].to_numpy(dtype=np.float64)
    log_ppl_cot = valid["log_ppl_cot"].to_numpy(dtype=np.float64)
    diff = valid["diff"].to_numpy(dtype=np.float64)

    # -------------------------------------------------------------------
    # (1)(2)(3) 三项核心分布
    # -------------------------------------------------------------------
    _section("1) log(ppl_y_given_x) 分布  (无 CoT 的 per-token NLL，越低越好)")
    text_histogram(
        log_ppl_x, bins=args.bins,
        title=f"  log_ppl_x = log(ppl_y_given_x) = -logp_y_given_x / num_target_tokens",
        width=args.hist_width,
    )

    _section("2) log(ppl_y_given_x_cot) 分布  (有 CoT 的 per-token NLL，越低越好)")
    text_histogram(
        log_ppl_cot, bins=args.bins,
        title=f"  log_ppl_cot = log(ppl_y_given_x_cot) = -logp_y_given_x_cot / num_target_tokens",
        width=args.hist_width,
    )

    _section("3) diff = log_ppl_cot - log_ppl_x 分布  (<0 表示 CoT 有帮助)")
    text_histogram(
        diff, bins=args.bins,
        title="  diff = log_ppl_cot - log_ppl_x = (logp_x - logp_cot) / num_target_tokens",
        width=args.hist_width,
    )

    # -------------------------------------------------------------------
    # 分位数对比表
    # -------------------------------------------------------------------
    _section("分位数对比表")
    text_percentile_table([
        ("log_ppl_x", log_ppl_x),
        ("log_ppl_cot", log_ppl_cot),
        ("diff", diff),
    ])

    # -------------------------------------------------------------------
    # CoT 帮助统计
    # -------------------------------------------------------------------
    _section("CoT 帮助统计 (diff < 0 ⟹ CoT 降低 per-token NLL ⟹ 有帮助)")
    n = diff.size
    helps = diff < 0
    hurts = diff > 0
    same = diff == 0
    print(f"  CoT 有帮助 (diff<0): {int(helps.sum())}  ({helps.mean()*100:.2f}%)")
    print(f"  CoT 有害   (diff>0): {int(hurts.sum())}  ({hurts.mean()*100:.2f}%)")
    print(f"  无变化     (diff=0): {int(same.sum())}  ({same.mean()*100:.2f}%)")
    print(f"  diff 均值         : {diff.mean():+.6f}  (负=整体上 CoT 有帮助)")
    print(f"  diff 中位数       : {np.median(diff):+.6f}")
    print()

    thresholds = [-0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.05, 0.1]
    print("  不同 diff 阈值下的保留率 (diff < threshold 视为 CoT 有效，予以保留):")
    print(f"  {'threshold':>12s}  {'retained':>10s}  {'retained%':>10s}")
    for t in thresholds:
        keep = diff < t
        print(f"  {t:>+12.3f}  {int(keep.sum()):>10d}  {keep.mean()*100:>9.2f}%")
    print()

    # -------------------------------------------------------------------
    # base 难度 × CoT 效果 联合表：判断 CoT 在哪些难度区间有用
    # -------------------------------------------------------------------
    _section("base 难度 × CoT 效果 联合表  (按 log_ppl_x 分桶看 CoT 帮助率)")
    # 自适应分桶：用分位数切 5 桶，避免空桶
    try:
        bucket_edges = np.percentile(log_ppl_x, [0, 20, 40, 60, 80, 100])
    except Exception:
        bucket_edges = np.linspace(log_ppl_x.min(), log_ppl_x.max(), 6)
    bucket_edges = np.unique(bucket_edges)  # 防 min==max
    # 联合保留阈值（推荐档：diff < -0.05；保守档：diff < -0.1）
    keep_loose = diff < -0.05
    keep_strict = diff < -0.10
    print(
        f"  {'log_ppl_x 桶':>22s}  {'n':>8s}  {'占比':>7s}  "
        f"{'help%':>7s}  {'mean_diff':>10s}  {'med_diff':>10s}  "
        f"{'keep<-0.05':>11s}  {'keep<-0.10':>11s}"
    )
    for i in range(len(bucket_edges) - 1):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        if i == len(bucket_edges) - 2:
            mask = (log_ppl_x >= lo) & (log_ppl_x <= hi)
        else:
            mask = (log_ppl_x >= lo) & (log_ppl_x < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"  [{lo:>7.3f}, {hi:>7.3f}]  {0:>8d}  {0:>6.2f}%  "
                  f"{'-':>7s}  {'-':>10s}  {'-':>10s}  {'-':>11s}  {'-':>11s}")
            continue
        gd = diff[mask]
        gx = log_ppl_x[mask]
        help_rate = (gd < 0).mean() * 100
        keep_l = int(keep_loose[mask].sum())
        keep_s = int(keep_strict[mask].sum())
        print(
            f"  [{lo:>7.3f}, {hi:>7.3f}]  {cnt:>8d}  {cnt/diff.size*100:>6.2f}%  "
            f"{help_rate:>6.2f}%  {gd.mean():>+10.5f}  {np.median(gd):>+10.5f}  "
            f"{keep_l:>5d}({keep_l/cnt*100:>4.1f}%)  {keep_s:>5d}({keep_s/cnt*100:>4.1f}%)"
        )
    # 整体联合保留率（绝对质量门 + CoT 有效门）
    abs_gate = log_ppl_x < 8.0
    print()
    print("  联合保留率 (log_ppl_x<8.0 且 diff<threshold):")
    for t in [-0.10, -0.05, 0.0]:
        m = abs_gate & (diff < t)
        print(f"    log_ppl_x<8 且 diff<{t:>+5.2f}  ->  {int(m.sum()):>8d}  "
              f"({m.mean()*100:>5.2f}% of all)")
    print()

    # -------------------------------------------------------------------
    # 按 target_idx 分组：CoT 对第几个 target 帮助更大？
    # -------------------------------------------------------------------
    if "target_idx" in valid.columns:
        _section("按 target_idx 分组  (CoT 对第 1/2/3 个 target 的帮助差异)")
        print(
            f"  {'target_idx':>12s}  {'n':>8s}  {'占比':>7s}  "
            f"{'help%':>7s}  {'mean_diff':>10s}  {'med_diff':>10s}  "
            f"{'mean_log_ppl_x':>14s}"
        )
        # 按出现频次排序
        idx_vals = valid["target_idx"].dropna()
        if idx_vals.size:
            for tidx in sorted(idx_vals.unique()):
                mask = (valid["target_idx"] == tidx).to_numpy()
                gd = diff[mask]
                gx = log_ppl_x[mask]
                if gd.size == 0:
                    continue
                help_rate = (gd < 0).mean() * 100
                print(
                    f"  {'target%02d' % int(tidx):>12s}  {gd.size:>8d}  "
                    f"{gd.size/diff.size*100:>6.2f}%  {help_rate:>6.2f}%  "
                    f"{gd.mean():>+10.5f}  {np.median(gd):>+10.5f}  "
                    f"{gx.mean():>14.4f}"
                )
        print()

    # -------------------------------------------------------------------
    # CoT 长度 vs CoT 效果：CoT 越长越有用还是越有害？
    # -------------------------------------------------------------------
    if "cot" in valid.columns:
        cot_series = valid["cot"]
        cot_len = cot_series.fillna("").astype(str).str.len()
        cot_len_arr = cot_len.to_numpy(dtype=np.float64)
        # 只有当长度有变化时才画
        if cot_len_arr.size and cot_len_arr.std() > 1e-6:
            _section("CoT 长度 vs CoT 效果  (CoT 越长越有用还是越有害?)")
            print(f"  CoT 字符长度: mean={cot_len_arr.mean():.1f}  "
                  f"median={np.median(cot_len_arr):.1f}  "
                  f"min={int(cot_len_arr.min())}  max={int(cot_len_arr.max())}")
            # 按长度分位数分 5 桶
            len_edges = np.percentile(cot_len_arr, [0, 20, 40, 60, 80, 100])
            len_edges = np.unique(len_edges)
            print(
                f"  {'CoT 长度桶':>22s}  {'n':>8s}  {'占比':>7s}  "
                f"{'help%':>7s}  {'mean_diff':>10s}  {'med_diff':>10s}  "
                f"{'mean_len':>9s}"
            )
            for i in range(len(len_edges) - 1):
                lo = len_edges[i]
                hi = len_edges[i + 1]
                if i == len(len_edges) - 2:
                    mask = (cot_len_arr >= lo) & (cot_len_arr <= hi)
                else:
                    mask = (cot_len_arr >= lo) & (cot_len_arr < hi)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                gd = diff[mask]
                help_rate = (gd < 0).mean() * 100
                print(
                    f"  [{lo:>7.0f}, {hi:>7.0f}]  {cnt:>8d}  "
                    f"{cnt/diff.size*100:>6.2f}%  {help_rate:>6.2f}%  "
                    f"{gd.mean():>+10.5f}  {np.median(gd):>+10.5f}  "
                    f"{cot_len_arr[mask].mean():>9.1f}"
                )
            # 长度与 diff 的 Spearman 秩相关（对偏态/重尾更稳健）
            rho_len = _spearman(cot_len_arr, diff)
            if rho_len is not None:
                print(f"  Spearman(cot_len, diff) = {rho_len:+.4f}  "
                      "(负=CoT 越长越有帮助；正=越长越有害)")
            print()

    # -------------------------------------------------------------------
    # 按 sample_key (uid+domain) 的 CoT 一致性：
    # 同一个 user 的多个 target，CoT 是同时帮上忙 / 同时有害 / 还是有拆有合？
    # 决定筛选用 user-level 还是 target-level 阈值。
    # -------------------------------------------------------------------
    if "sample_key" in valid.columns:
        _section("按 sample_key (uid+domain) 的 CoT 一致性  (user-level vs target-level 筛选)")
        grp = valid.groupby("sample_key", dropna=False)
        per_user = grp.agg(
            n_targets=("diff", "size"),
            n_helped=("diff", lambda s: int((s < 0).sum())),
            mean_diff=("diff", "mean"),
            min_diff=("diff", "min"),
            max_diff=("diff", "max"),
        )
        total_users = len(per_user)
        # 按「组内帮助比例」分类
        per_user["help_frac"] = per_user["n_helped"] / per_user["n_targets"]
        # 只对有多个 target 的 user 才讨论一致性
        multi = per_user[per_user["n_targets"] >= 2]
        single = per_user[per_user["n_targets"] == 1]
        print(f"  user (sample_key) 总数        : {total_users}")
        print(f"  只有 1 个 target 的 user     : {len(single)}  "
              f"({len(single)/total_users*100:.1f}%)  -> 这些只能 target-level 判断")
        print(f"  有 >=2 个 target 的 user     : {len(multi)}  "
              f"({len(multi)/total_users*100:.1f}%)")
        print()
        if len(multi) > 0:
            all_help = (multi["help_frac"] == 1.0).sum()
            all_hurt = (multi["help_frac"] == 0.0).sum()
            mixed = len(multi) - all_help - all_hurt
            print(f"  在 >=2 target 的 user 中:")
            print(f"    CoT 对所有 target 都有帮助 : {all_help:>6d}  "
                  f"({all_help/len(multi)*100:>5.1f}%)  -> user-level 筛选最省事")
            print(f"    CoT 对所有 target 都有害    : {all_hurt:>6d}  "
                  f"({all_hurt/len(multi)*100:>5.1f}%)  -> user-level 丢弃最省事")
            print(f"    有帮助也有害 (mixed)        : {mixed:>6d}  "
                  f"({mixed/len(multi)*100:>5.1f}%)  -> 必须 target-level 筛选")
            # 组内 diff 的极差，衡量 CoT 对同一 user 不同 target 的不一致程度
            spread = (multi["max_diff"] - multi["min_diff"])
            print(f"  组内 diff 极差 (max-min): mean={spread.mean():.4f}  "
                  f"median={spread.median():.4f}  "
                  f"p90={spread.quantile(0.9):.4f}")
            print(f"    极差小说明 CoT 对同一 user 作用一致 → user-level 筛选可行；")
            print(f"    极差大说明 CoT 是 target-specific → 必须 target-level 筛选。")
        print()

    # -------------------------------------------------------------------
    # num_target_tokens 分布
    # -------------------------------------------------------------------
    _section("num_target_tokens 分布  (打分 token 数；影响 ppl 的尺度)")
    n_tok = valid["num_target_tokens_num"].dropna().to_numpy(dtype=np.float64)
    if n_tok.size:
        text_histogram(n_tok, bins=min(args.bins, 30),
                       title="  num_target_tokens", width=args.hist_width)
        print(f"  均值={n_tok.mean():.2f}  中位数={np.median(n_tok):.1f}  "
              f"min={int(n_tok.min())}  max={int(n_tok.max())}")
        print()

    # -------------------------------------------------------------------
    # 2D 热力图: log_ppl_x vs log_ppl_cot
    # -------------------------------------------------------------------
    _section("log_ppl_x vs log_ppl_cot 2D 热力图  (点落在对角线下方=CoT 有帮助)")
    text_heatmap_2d(
        log_ppl_x, log_ppl_cot, bins=args.heatmap_bins,
        xlabel="log_ppl_x (无 CoT)", ylabel="log_ppl_cot (有 CoT)",
        width=args.heatmap_bins * 2,
    )

    # 对角线参考：在同一个 x 范围打印 y=x 线
    print("  参考: y=x 对角线。点越靠对角线下方，CoT 带来的 NLL 下降越多。")
    print()

    # -------------------------------------------------------------------
    # 按 target_domain 分组
    # -------------------------------------------------------------------
    if "target_domain" in valid.columns:
        _section("按 target_domain 分组对比")
        for domain, grp in valid.groupby("target_domain", dropna=False):
            gx = grp["log_ppl_x"].to_numpy(dtype=np.float64)
            gc = grp["log_ppl_cot"].to_numpy(dtype=np.float64)
            gd = grp["diff"].to_numpy(dtype=np.float64)
            n_d = gd.size
            help_rate = (gd < 0).mean() * 100 if n_d else float("nan")
            print(f"  [{domain}]  n={n_d}  help_rate(diff<0)={help_rate:.2f}%  "
                  f"mean_diff={gd.mean():+.5f}  median_diff={np.median(gd):+.5f}")
            print(f"           log_ppl_x mean={gx.mean():.4f} median={np.median(gx):.4f}  "
                  f"| log_ppl_cot mean={gc.mean():.4f} median={np.median(gc):.4f}")
        print()

    # -------------------------------------------------------------------
    # 按 representation_type 分组（混合时才有意义）
    # -------------------------------------------------------------------
    if "representation_type" in valid.columns:
        reps = valid["representation_type"].value_counts(dropna=False)
        if len(reps) > 1:
            _section("按 representation_type 分组对比")
            for rep, grp in valid.groupby("representation_type", dropna=False):
                gd = grp["diff"].to_numpy(dtype=np.float64)
                help_rate = (gd < 0).mean() * 100 if gd.size else float("nan")
                print(f"  [{rep}]  n={gd.size}  help_rate={help_rate:.2f}%  "
                      f"mean_diff={gd.mean():+.5f}  median_diff={np.median(gd):+.5f}")
            print()

    # -------------------------------------------------------------------
    # 极端样本示例
    # -------------------------------------------------------------------
    if args.show_examples > 0:
        _section(f"CoT 最有帮助 / 最有害的 {args.show_examples} 条样本")
        # 过滤掉异常长的 cot/target，避免刷屏
        def _short(s: Any, n: int = 200) -> str:
            s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
            s = s.replace("\n", " ").replace("\r", " ")
            return s if len(s) <= n else s[: n - 3] + "..."

        valid_idx = valid.index
        # 重新在 valid 上排序
        sorted_help = valid.reindex(valid["diff"].sort_values().index)
        print("  --- CoT 最有帮助 (diff 最小) ---")
        for i, (_, row) in enumerate(sorted_help.head(args.show_examples).iterrows()):
            print(
                f"  [{i+1}] ppl_sample_key={row.get('ppl_sample_key','')} "
                f"domain={row.get('target_domain','')} "
                f"target_pid={row.get('target_pid','')} "
                f"n_tok={row.get('num_target_tokens_num','')}"
            )
            print(f"      diff={row['diff']:+.5f}  log_ppl_x={row['log_ppl_x']:.4f}  "
                  f"log_ppl_cot={row['log_ppl_cot']:.4f}")
            print(f"      target: {_short(row.get('target_text',''))}")
            print(f"      cot   : {_short(row.get('cot',''))}")
        print()
        print("  --- CoT 最有害 (diff 最大) ---")
        for i, (_, row) in enumerate(sorted_help.tail(args.show_examples).iterrows()):
            print(
                f"  [{i+1}] ppl_sample_key={row.get('ppl_sample_key','')} "
                f"domain={row.get('target_domain','')} "
                f"target_pid={row.get('target_pid','')} "
                f"n_tok={row.get('num_target_tokens_num','')}"
            )
            print(f"      diff={row['diff']:+.5f}  log_ppl_x={row['log_ppl_x']:.4f}  "
                  f"log_ppl_cot={row['log_ppl_cot']:.4f}")
            print(f"      target: {_short(row.get('target_text',''))}")
            print(f"      cot   : {_short(row.get('cot',''))}")
        print()

    # -------------------------------------------------------------------
    # 相关性
    # -------------------------------------------------------------------
    _section("相关性分析")
    if log_ppl_x.size > 1:
        corr = float(np.corrcoef(log_ppl_x, log_ppl_cot)[0, 1])
        print(f"  Pearson (log_ppl_x, log_ppl_cot) = {corr:.4f}")
    # base 难度 vs CoT 效果（Spearman 更稳）
    rho_xd = _spearman(log_ppl_x, diff)
    if rho_xd is not None:
        print(f"  Spearman(log_ppl_x, diff)       = {rho_xd:+.4f}  "
              "(负=base 越难 CoT 越有帮助；正=base 越难 CoT 越有害)")
    if "num_target_tokens_num" in valid.columns:
        n_tok_arr = valid["num_target_tokens_num"].dropna().to_numpy(dtype=np.float64)
        if n_tok_arr.size > 1 and n_tok_arr.size == diff.size:
            mask = np.isfinite(n_tok_arr) & np.isfinite(diff)
            if mask.sum() > 1:
                a = n_tok_arr[mask]
                b = diff[mask]
                # 当 num_target_tokens 恒定时（如 sid 表示恒为 3），stddev=0，
                # corrcoef 会产生 nan + RuntimeWarning，这里提前拦截。
                if a.std() < 1e-12 or b.std() < 1e-12:
                    uniq = float(np.unique(a)[0]) if a.size else float("nan")
                    print(f"  Pearson (num_target_tokens, diff) = N/A  "
                          f"(num_target_tokens 恒定 = {uniq:g}，方差为 0，相关性无定义)")
                else:
                    corr_td = float(np.corrcoef(a, b)[0, 1])
                    print(f"  Pearson (num_target_tokens, diff) = {corr_td:+.4f}  "
                          "(若为负，说明目标越长 CoT 越容易帮上忙)")
    print()

    # -------------------------------------------------------------------
    # 可选落盘
    # -------------------------------------------------------------------
    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)) or ".", exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"  [save] 已落盘 csv: {args.save_csv}  ({len(df)} 行)")

    if args.save_filtered:
        keep_mask = (df["diff"].to_numpy() < args.diff_threshold)
        keep_df = df[keep_mask].copy()
        # 删掉中间列 num_target_tokens_num
        if "num_target_tokens_num" in keep_df.columns:
            keep_df = keep_df.drop(columns=["num_target_tokens_num"])
        # 写 parquet
        table = pa.Table.from_pandas(keep_df, preserve_index=False)
        parsed = urlparse(args.save_filtered)
        if parsed.scheme and parsed.scheme != "file":
            fs2, fs_path = get_filesystem_and_path(args.save_filtered)
            parent = os.path.dirname(fs_path.rstrip("/"))
            try:
                fs2.create_dir(parent, recursive=True)
            except Exception:
                pass
            pq.write_table(table, fs_path, filesystem=fs2)
        else:
            local = parsed.path if parsed.scheme == "file" else args.save_filtered
            local = os.path.abspath(local)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            pq.write_table(table, local)
        print(f"  [save] diff<{args.diff_threshold} 的样本落盘 parquet: "
              f"{args.save_filtered}  ({len(keep_df)} 行)")

    print()
    print("=" * 78)
    print("  完成。")
    print("=" * 78)


if __name__ == "__main__":
    main()
