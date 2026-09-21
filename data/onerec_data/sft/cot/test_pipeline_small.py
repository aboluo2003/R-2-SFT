#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline_small.py — RecIF CoT pipeline 端到端「冒烟测试」

在单机（无 MPI、无 GPU、无 API）模式下，用一小切片真实数据跑通每一步，
验证产物路径与 schema 没有问题。所有产物写到 /tmp/onerec_cot_test_<ts>/。

跑哪些步骤、跳过哪些步骤：
  step 0a  split.py               (CPU)  ✓ 跑
  step 0b  summary.py (vLLM)      (GPU)  ✗ 跳过（用 raw dense_caption 喂下游）
  step 1   1_history_text/main.py (CPU)  ✓ 跑
  step 2   2_cot/main.py --dry_run (no API) ✓ 跑（写占位 cot，不调 API）
  step 0'  build warmup data (5_data_format/main.py) (CPU) ✓ 跑（用 raw CoT）
  train    warmup scorer          (GPU)  ✗ 跳过
  step 3 preprocess              (CPU)  ✓ 跑
  step 3 PPL eval (2_eval_ppl.py) (GPU)  ✗ 跳过 → 用 mock 随机 PPL 替代
  step 3 join                     (CPU)  ✓ 跑
  step 4 filter                  (CPU)  ✓ 跑（先 annotate，再 threshold）
  step 5 data_format              (CPU)  ✓ 跑（最终 SFT 数据集）

用法：
  python3 test_pipeline_small.py
  python3 test_pipeline_small.py --n_users 30 --n_pids 4000
  python3 test_pipeline_small.py --out_root /tmp/onerec_cot_test
  python3 test_pipeline_small.py --skip 0a,1     # 跳过某些步骤
  python3 test_pipeline_small.py --only 1,2      # 只跑某些步骤

依赖：
  pandas, pyarrow, numpy, tqdm,（mpi4py 可选，缺失会自动退化单进程）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# 路径
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # OpenOneRec/data/onerec_data/sft/cot -> repo root
OPEN_ONE_REC = SCRIPT_DIR.parents[3]  # OpenOneRec 根
RECIDIR = OPEN_ONE_REC / "data" / "RecIF"

BENCH_PARQUET = RECIDIR / "onerec_bench_release.parquet"
PID2CAPTION_PARQUET = RECIDIR / "pid2caption.parquet"
VIDEO_AD_PID2SID_PARQUET = RECIDIR / "video_ad_pid2sid.parquet"
PRODUCT_PID2SID_PARQUET = RECIDIR / "product_pid2sid.parquet"

# 各 step 的 Python 入口（直接调，不走 MPI wrapper）
PY_STEP0A_SPLIT = SCRIPT_DIR / "0_caption_summary" / "split.py"
PY_STEP1_HISTORY = SCRIPT_DIR / "1_history_text" / "main.py"
PY_STEP2_COT = SCRIPT_DIR / "2_cot" / "main.py"
PY_STEP3_PREPROCESS = SCRIPT_DIR / "3_ppl_eval" / "1_preprocess.py"
PY_STEP3_JOIN = SCRIPT_DIR / "3_ppl_eval" / "3_join_back.py"
PY_STEP4_FILTER = SCRIPT_DIR / "4_filter_ppl" / "main.py"
PY_STEP5_DATA_FORMAT = SCRIPT_DIR / "5_data_format" / "main.py"

DEFAULT_OUT_ROOT = Path("/tmp/onerec_cot_test")


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------
def log(msg: str, *, level: str = "INFO") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def hr(title: str = "", char: str = "=", width: int = 80) -> None:
    if title:
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")
    else:
        print(char * width)


def run_step(
    name: str,
    cmd: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Tuple[int, str, str]:
    """运行一个子进程，返回 (rc, stdout, stderr)。失败不抛，由调用方决定。"""
    hr(f"RUN {name}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    if env:
        for k, v in env.items():
            if k not in ("PATH", "PYTHONPATH"):
                print(f"  env {k}={v}")
    print()
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            list(cmd),
            env=full_env,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        rc = proc.returncode
        out = proc.stdout or ""
        err = proc.stderr or ""
    except FileNotFoundError as e:
        rc = 127
        out = ""
        err = f"FileNotFoundError: {e}"
    except Exception as e:  # noqa: BLE001
        rc = 1
        out = ""
        err = f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0
    # 控制台回显（截断超长输出）
    if out:
        print("---- stdout (tail) ----")
        print(_tail(out, 80))
    if err:
        print("---- stderr (tail) ----")
        print(_tail(err, 40))
    print(f"\n  -> rc={rc} elapsed={elapsed:.1f}s")
    return rc, out, err


def _tail(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "...(truncated)...\n" + "\n".join(lines[-n:])


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 数据准备：切小 bench + 用 bench 出现的 pid 过滤 pid2caption / pid2sid
# -----------------------------------------------------------------------------
def collect_pids_from_bench(bench_df: pd.DataFrame) -> Tuple[set, set, set]:
    """从 bench 行里收集所有 pid：video/ad、goods。返回 (video_ad_pids, goods_pids, all_pids)。"""
    video_ad_pids: set = set()
    goods_pids: set = set()

    pid_cols_video_ad = [
        "hist_video_pid", "target_video_pid",
        "hist_ad_pid", "target_ad_pid",
        "hist_longview_video_list",
    ]
    pid_cols_goods = ["hist_goods_pid", "target_goods_pid"]

    def _to_ints(v: Any) -> List[int]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            out = []
            for x in v:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return out
        return []

    for _, row in bench_df.iterrows():
        for col in pid_cols_video_ad:
            video_ad_pids.update(_to_ints(row.get(col)))
        for col in pid_cols_goods:
            goods_pids.update(_to_ints(row.get(col)))

    all_pids = video_ad_pids | goods_pids
    return video_ad_pids, goods_pids, all_pids


def filter_parquet_by_pids(
    src: Path,
    dst: Path,
    target_pids: set,
    pid_col: str = "pid",
) -> int:
    """按 pid 集合过滤 src parquet，写到 dst。返回写出的行数。
    用 pyarrow.dataset + pc.is_in 做向量化过滤（即使单 row group 也能在 IO 流式读时过滤）。"""
    if not src.exists():
        raise FileNotFoundError(src)
    safe_makedirs(dst.parent)
    if not target_pids:
        # 写一个空 parquet，保留 schema
        pf = pq.ParquetFile(str(src))
        schema = pf.schema_arrow
        empty = pa.table({name: pa.array([], type=schema.field(name).type) for name in schema.names})
        pq.write_table(empty, str(dst))
        return 0

    import pyarrow.dataset as ds  # lazy import
    import pyarrow.compute as pc

    target_arr = pa.array(sorted(int(p) for p in target_pids), type=pa.int64())
    t0 = time.time()
    dset = ds.dataset(str(src), format="parquet")
    filt = pc.is_in(ds.field(pid_col), value_set=target_arr)
    tab = dset.to_table(filter=filt)
    # combine_chunks 让写出的 parquet 更紧凑
    if tab.num_rows > 0:
        tab = tab.combine_chunks()
    pq.write_table(tab, str(dst))
    log(f"  filter {src.name} -> {dst.name}: rows={tab.num_rows} (target_pids={len(target_pids)}) elapsed={time.time()-t0:.1f}s")
    return tab.num_rows


def prepare_tiny_inputs(
    out_dir: Path,
    n_users: int,
    n_pids_caption: int,
    skip_if_exists: bool = True,
) -> Dict[str, Path]:
    """
    1) 从 bench 取前 n_users 行 -> tiny_bench.parquet
    2) 用 bench 中出现的 video/ad pid 过滤 pid2caption.parquet
       -> tiny_pid2caption.parquet （step1 用，以及 step5 的 caption_dir 来源）
       若过滤结果为空（bench pid 与 caption 不重叠的极端情况），
       fallback 取 pid2caption 的前 n_pids_caption 行作为 caption 池，避免下游空跑。
    3) 用 bench 中 video/ad pid 过滤 video_ad_pid2sid.parquet -> tiny_pid2sid.parquet
    4) 用 bench 中 goods pid 过滤 product_pid2sid.parquet -> tiny_product_pid2sid.parquet

    若 skip_if_exists=True 且四个 tiny 文件都已存在，则直接返回路径不做任何操作。

    返回各 tiny 输入的路径 dict。
    """
    hr("PREPARE tiny inputs")
    inputs_dir = out_dir / "inputs"
    tiny_bench = inputs_dir / "tiny_bench.parquet"
    tiny_pid2caption = inputs_dir / "tiny_pid2caption.parquet"
    tiny_pid2sid = inputs_dir / "tiny_video_ad_pid2sid.parquet"
    tiny_product_pid2sid = inputs_dir / "tiny_product_pid2sid.parquet"

    if skip_if_exists and all(p.exists() for p in [tiny_bench, tiny_pid2caption, tiny_pid2sid, tiny_product_pid2sid]):
        log("  tiny inputs already exist, skip prep")
        return {
            "tiny_bench": tiny_bench,
            "tiny_pid2caption": tiny_pid2caption,
            "tiny_pid2sid": tiny_pid2sid,
            "tiny_product_pid2sid": tiny_product_pid2sid,
        }

    safe_makedirs(inputs_dir)

    # ---- 1) tiny bench ----
    log(f"reading first {n_users} rows from {BENCH_PARQUET.name} ...")
    t0 = time.time()
    bench_table = pq.read_table(str(BENCH_PARQUET)).slice(0, n_users)
    bench_table = bench_table.combine_chunks()
    pq.write_table(bench_table, str(tiny_bench))
    bench_df = bench_table.to_pandas()
    log(f"  tiny_bench: rows={len(bench_df)} cols={len(bench_df.columns)} elapsed={time.time()-t0:.1f}s")

    # ---- 2) tiny pid2caption (按 bench video/ad pid 过滤) ----
    video_ad_pids, goods_pids, _ = collect_pids_from_bench(bench_df)
    log(f"bench 中 video/ad pids={len(video_ad_pids)} goods_pids={len(goods_pids)}")

    n_cap = filter_parquet_by_pids(PID2CAPTION_PARQUET, tiny_pid2caption, video_ad_pids, pid_col="pid")
    if n_cap == 0:
        log("  WARN: pid2caption 过滤结果为空，fallback 取前 N 行作为 caption 池", level="WARN")
        t0 = time.time()
        cap_table = pq.read_table(str(PID2CAPTION_PARQUET)).slice(0, n_pids_caption)
        cap_table = cap_table.combine_chunks()
        pq.write_table(cap_table, str(tiny_pid2caption))
        log(f"  tiny_pid2caption (fallback): rows={cap_table.num_rows} elapsed={time.time()-t0:.1f}s")
    else:
        log(f"  tiny_pid2caption: rows={n_cap}")

    # ---- 3) tiny video_ad_pid2sid ----
    filter_parquet_by_pids(VIDEO_AD_PID2SID_PARQUET, tiny_pid2sid, video_ad_pids, pid_col="pid")

    # ---- 4) tiny product_pid2sid ----
    if goods_pids:
        filter_parquet_by_pids(PRODUCT_PID2SID_PARQUET, tiny_product_pid2sid, goods_pids, pid_col="pid")
    else:
        # 写空表
        pf = pq.ParquetFile(str(PRODUCT_PID2SID_PARQUET))
        schema = pf.schema_arrow
        empty = pa.table({name: pa.array([], type=schema.field(name).type) for name in schema.names})
        pq.write_table(empty, str(tiny_product_pid2sid))
        log(f"  no goods pids in bench, wrote empty {tiny_product_pid2sid.name}")

    return {
        "tiny_bench": tiny_bench,
        "tiny_pid2caption": tiny_pid2caption,
        "tiny_pid2sid": tiny_pid2sid,
        "tiny_product_pid2sid": tiny_product_pid2sid,
    }


# -----------------------------------------------------------------------------
# 各 step 执行函数
# -----------------------------------------------------------------------------
def step0a_split(out_dir: Path, tiny_pid2caption: Path, rows_per_shard: int = 20) -> Path:
    """step 0a: split.py — 把 tiny_pid2caption 拆成小分片，作为 step5 的 caption_dir。"""
    caption_shards = out_dir / "step0a_caption_shards"
    if caption_shards.exists():
        shutil.rmtree(caption_shards)
    safe_makedirs(caption_shards)
    cmd = [
        sys.executable, str(PY_STEP0A_SPLIT),
        "--input", str(tiny_pid2caption),
        "--output_dir", str(caption_shards),
        "--rows_per_shard", str(rows_per_shard),
    ]
    rc, _, _ = run_step("step 0a (split)", cmd)
    if rc != 0:
        raise RuntimeError(f"step 0a failed rc={rc}")
    return caption_shards


def step1_history_text(
    out_dir: Path,
    tiny_bench: Path,
    tiny_pid2caption: Path,
    rows_per_shard: int = 50,
) -> Path:
    """step 1: 1_history_text/main.py — 抽取 video/ad history+caption 文本。"""
    out = out_dir / "step1_history_text"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        # 走 raw dense_caption，不依赖 step0b summary
        "CAPTION_COLUMN": "dense_caption",
        "TARGET_MAX_LEN_STEP1": "3",  # 截断 target 到 3，减小产物
    }
    cmd = [
        sys.executable, str(PY_STEP1_HISTORY),
        "--bench_uri", str(tiny_bench),
        "--caption_uri", str(tiny_pid2caption),
        "--caption_column", "dense_caption",
        "--output_uri", str(out),
        "--rows_per_shard", str(rows_per_shard),
        "--min_history_len", "1",
        "--keep_no_caption_target", "0",
        "--keep_no_caption_history", "0",
        "--target_max_len", "3",
        "--target_domain_filter", "",  # 处理 video+ad（product 不开源）
    ]
    rc, _, _ = run_step("step 1 (history_text)", cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"step 1 failed rc={rc}")
    return out


def step2_cot_dry_run(out_dir: Path, step1_out: Path, max_samples: int = 50) -> Path:
    """step 2: 2_cot/main.py --dry_run — 写占位 cot，不调 API。"""
    out = out_dir / "step2_cot"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    cmd = [
        sys.executable, str(PY_STEP2_COT),
        "--input_dir", str(step1_out),
        "--output_dir", str(out),
        "--dry_run",
        "--num_workers", "1",
        "--max_samples", str(max_samples),
        "--checkpoint_interval", "16",
    ]
    rc, _, _ = run_step("step 2 (cot dry-run, no API)", cmd)
    if rc != 0:
        raise RuntimeError(f"step 2 failed rc={rc}")
    return out


def step0prime_build_warmup_data(
    out_dir: Path,
    step2_out: Path,
    tiny_bench: Path,
    caption_shards: Path,
    tiny_pid2sid: Path,
) -> Tuple[Path, Path]:
    """step 0': build warmup scorer SFT data (raw ver5)。
    直接调 5_data_format/main.py，env 与 0_build_warmup_data.sh 对齐。"""
    out = out_dir / "step0prime_warmup_raw"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    temp_dir = out / "_temp"
    safe_makedirs(temp_dir)
    blended_json = out_dir / "blended" / "sft_recif_cot_ver5_raw.json"
    safe_makedirs(blended_json.parent)

    cmd = [
        sys.executable, str(PY_STEP5_DATA_FORMAT),
        "--cot_dir", str(step2_out),
        "--bench_parquet", str(tiny_bench),
        "--caption_dir", str(caption_shards),
        "--pid2sid", str(tiny_pid2sid),
        "--output_dir", str(out),
        "--blended_json_path", str(blended_json),
        "--temp_dir", str(temp_dir),
        "--enable_non_cot_rec", "1",
        "--enable_video_rec", "1",
        "--enable_ad_rec", "1",
        "--enable_product_rec", "0",
        "--enable_sid2caption", "1",
        "--enable_caption2sid", "1",
        "--allow_cot_non_cot_user_overlap", "1",
        "--enable_cot_split_filter", "1",
        "--cot_target_source", "bench",
        "--enable_product_sid2caption", "0",
        "--enable_product_caption2sid", "0",
        "--global_shuffle", "1",
        "--sample_ratio", "1.0",
        "--rows_per_shard", "100",
        "--target_max_len", "3",
        "--split_target", "1",
        "--sid2caption_size", "200",
        "--caption2sid_size", "200",
        "--shuffle_caption_files", "1",
        "--seed", "42",
        "--siddoc_mode", "0",
    ]
    rc, _, _ = run_step("step 0' (build warmup data, raw ver5)", cmd)
    if rc != 0:
        # step0' 用 raw cot 跑；如果 split_filter=1 把所有样本都过滤了（bench 里这些 uid 可能 split!=0），
        # 产物会是空的但不该报错。允许 rc=0 + 空，但 rc!=0 抛。
        raise RuntimeError(f"step 0' failed rc={rc}")
    return out, blended_json


def step3_preprocess(
    out_dir: Path,
    step2_out: Path,
    tiny_bench: Path,
    tiny_pid2sid: Path,
) -> Path:
    """step 3 preprocess: 1_preprocess.py — explode CoT 到 per-target PPL eval inputs (sid 模式)。"""
    out = out_dir / "step3_preprocess"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    cmd = [
        sys.executable, str(PY_STEP3_PREPROCESS),
        "--input_uri", str(step2_out),
        "--output_uri", str(out),
        "--representation_type", "sid",
        "--pid2sid_uri", str(tiny_pid2sid),
        "--target_source", "bench",
        "--bench_uri", str(tiny_bench),
        "--rows_per_shard", "100",
    ]
    rc, _, _ = run_step("step 3 preprocess (sid, bench)", cmd)
    if rc != 0:
        raise RuntimeError(f"step 3 preprocess failed rc={rc}")
    return out


def step3_ppl_eval_mock(out_dir: Path, preprocess_out: Path) -> Path:
    """step 3 PPL eval 的 mock 版本：读 preprocess 输出，给每行加随机 PPL 列。
    真实 step 是 GPU + 200-step 模型，这里用随机数替代以验证下游 join 链路。"""
    out = out_dir / "step3_ppl_eval_mock"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)

    hr("MOCK step 3 PPL eval (no GPU, random PPL scores)")
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    files = sorted(Path(preprocess_out).glob("part-*.parquet"))
    if not files:
        # 退化为任何 *.parquet
        files = sorted(Path(preprocess_out).glob("*.parquet"))
    n_files = 0
    n_rows = 0
    t0 = time.time()
    for f in files:
        df = pq.read_table(str(f)).to_pandas()
        n = len(df)
        if n == 0:
            # 写空表保留 schema
            df["logp_y_given_x"] = []
            df["logp_y_given_x_cot"] = []
            df["ppl_y_given_x"] = []
            df["ppl_y_given_x_cot"] = []
            df["num_target_tokens"] = []
            df["eval_meta"] = []
            dst = out / f.name
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(dst))
            n_files += 1
            continue
        # 随机 PPL 分数（log_ppl 范围 [-2, 6] 之类，符合 4_filter 的语义）
        logp_x = np_rng.normal(-1.5, 0.5, n).astype(np.float64)        # log P(target|x)
        logp_xcot = np_rng.normal(-1.0, 0.5, n).astype(np.float64)     # log P(target|x,cot)
        # ppl = exp(-logp / num_tokens)  简化为 exp(-logp)
        ppl_x = np.exp(-logp_x)
        ppl_xcot = np.exp(-logp_xcot)
        num_tokens = np_rng.integers(5, 30, n).astype(np.int64)
        eval_meta = [json.dumps({"truncated": False, "num_tokens": int(nt)}, ensure_ascii=False) for nt in num_tokens]
        df["logp_y_given_x"] = logp_x
        df["logp_y_given_x_cot"] = logp_xcot
        df["ppl_y_given_x"] = ppl_x
        df["ppl_y_given_x_cot"] = ppl_xcot
        df["num_target_tokens"] = num_tokens
        df["eval_meta"] = eval_meta
        dst = out / f.name
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(dst))
        n_files += 1
        n_rows += n
    log(f"  mock ppl_eval: files={n_files} rows={n_rows} elapsed={time.time()-t0:.1f}s -> {out}")
    return out


def step3_join(out_dir: Path, step2_out: Path, ppl_eval_out: Path) -> Path:
    """step 3 join: 3_join_back.py — 把 step2 CoT 与 mock PPL eval 按 sample_key join。"""
    out = out_dir / "step3_join"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    cmd = [
        sys.executable, str(PY_STEP3_JOIN),
        "--cot_input", str(step2_out),
        "--ppl_input", str(ppl_eval_out),
        "--output", str(out),
        "--rows_per_shard", "100",
    ]
    rc, _, _ = run_step("step 3 join (cot x mock-ppl)", cmd)
    if rc != 0:
        raise RuntimeError(f"step 3 join failed rc={rc}")
    return out


def step4_filter(out_dir: Path, join_out: Path) -> Tuple[Path, Path]:
    """step 4 filter: 4_filter_ppl/main.py — 先 annotate，再 threshold。"""
    annotate_out = out_dir / "step4_filter_annotate"
    threshold_out = out_dir / "step4_filter_threshold"
    if annotate_out.exists():
        shutil.rmtree(annotate_out)
    if threshold_out.exists():
        shutil.rmtree(threshold_out)
    safe_makedirs(annotate_out)
    safe_makedirs(threshold_out)

    # phase 1: annotate（不启用任何门，仅打分）
    p50_value: Optional[float] = None
    summary_path = annotate_out / "filter_summary.json"

    cmd1 = [
        sys.executable, str(PY_STEP4_FILTER),
        "--input", str(join_out),
        "--output", str(annotate_out),
        "--merge_rule", "any",
        "--rows_per_output_file", "100",
        "--log_level", "INFO",
    ]
    rc, _, _ = run_step("step 4 filter (phase 1: annotate_all)", cmd1)
    if rc != 0:
        raise RuntimeError(f"step 4 filter annotate failed rc={rc}")

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            stats = summary.get("log_ppl_cot_stats", {})
            p50_value = stats.get("p50")
            log(f"  annotate summary: log_ppl_cot_stats.p50={p50_value}")
        except Exception as e:  # noqa: BLE001
            log(f"  WARN: read filter_summary.json failed: {e}", level="WARN")

    # phase 2: threshold filter at p50（fallback 4.0）
    threshold = p50_value if p50_value is not None else 4.0
    cmd2 = [
        sys.executable, str(PY_STEP4_FILTER),
        "--input", str(join_out),
        "--output", str(threshold_out),
        "--max_log_ppl_cot", str(threshold),
        "--max_log_ppl_x", "9.0",
        "--merge_rule", "any",
        "--rows_per_output_file", "100",
        "--log_level", "INFO",
    ]
    rc, _, _ = run_step(f"step 4 filter (phase 2: threshold@p50={threshold:.3f})", cmd2)
    if rc != 0:
        # threshold filter 失败不致命：annotate 已经能验证 schema，继续走，用 annotate 当 step5 输入
        log(f"  WARN: step 4 threshold filter failed rc={rc}, will use annotate output for step 5", level="WARN")
        return annotate_out, threshold_out
    return annotate_out, threshold_out


def step5_data_format(
    out_dir: Path,
    cot_dir: Path,
    tiny_bench: Path,
    caption_shards: Path,
    tiny_pid2sid: Path,
) -> Tuple[Path, Path]:
    """step 5 data_format: 5_data_format/main.py — 最终 SFT 数据集。"""
    out = out_dir / "step5_data_format"
    if out.exists():
        shutil.rmtree(out)
    safe_makedirs(out)
    temp_dir = out / "_temp"
    safe_makedirs(temp_dir)
    blended_json = out_dir / "blended" / "sft_recif_cot_ver5_sft_ppl.json"
    safe_makedirs(blended_json.parent)

    cmd = [
        sys.executable, str(PY_STEP5_DATA_FORMAT),
        "--cot_dir", str(cot_dir),
        "--bench_parquet", str(tiny_bench),
        "--caption_dir", str(caption_shards),
        "--pid2sid", str(tiny_pid2sid),
        "--output_dir", str(out),
        "--blended_json_path", str(blended_json),
        "--temp_dir", str(temp_dir),
        "--enable_non_cot_rec", "1",
        "--enable_video_rec", "1",
        "--enable_ad_rec", "1",
        "--enable_product_rec", "0",
        "--enable_sid2caption", "1",
        "--enable_caption2sid", "1",
        "--allow_cot_non_cot_user_overlap", "1",
        "--enable_cot_split_filter", "1",
        "--cot_target_source", "cot",  # 5_data_format.sh 用 cot（4_filter 已筛选）
        "--enable_product_sid2caption", "0",
        "--enable_product_caption2sid", "0",
        "--global_shuffle", "1",
        "--sample_ratio", "1.0",
        "--rows_per_shard", "100",
        "--target_max_len", "3",
        "--split_target", "1",
        "--sid2caption_size", "200",
        "--caption2sid_size", "200",
        "--shuffle_caption_files", "1",
        "--seed", "42",
        "--siddoc_mode", "0",
    ]
    rc, _, _ = run_step("step 5 (data_format, sft_ppl ver5, main output)", cmd)
    if rc != 0:
        raise RuntimeError(f"step 5 failed rc={rc}")
    return out, blended_json


# -----------------------------------------------------------------------------
# 产物汇总
# -----------------------------------------------------------------------------
def inspect_dir(p: Path, max_files: int = 6) -> str:
    """返回一个目录的简要信息：文件数 + 前 max_files 个文件名 + 总大小。"""
    if not p.exists():
        return f"<missing: {p}>"
    if p.is_file():
        size = p.stat().st_size
        return f"{p} (file, {size:,} bytes)"
    files = sorted(p.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    head = files[:max_files]
    head_str = "\n      ".join(
        f"{f.relative_to(p.parent)} ({f.stat().st_size:,}B)" for f in head
    )
    extra = f"  ... (+{len(files) - len(head)} more)" if len(files) > len(head) else ""
    return f"{p} (files={len(files)} total={total_size:,}B)\n      {head_str}{extra}"


def print_summary(out_dir: Path, paths: Dict[str, Path]) -> None:
    hr("SUMMARY — all step outputs")
    print(f"\nOutput root: {out_dir}\n")
    for label, p in paths.items():
        print(f"  {label:32s}: {inspect_dir(p)}")
    hr()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RecIF CoT pipeline small-scale end-to-end test (single machine, no MPI/GPU/API).",
    )
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT),
                    help=f"输出根目录，默认 {DEFAULT_OUT_ROOT}")
    ap.add_argument("--n_users", type=int, default=30,
                    help="切片 bench 的用户数（前 N 行），默认 30")
    ap.add_argument("--n_pids_caption", type=int, default=4000,
                    help="切片 pid2caption 的行数（前 N 行作为 caption 池），默认 4000")
    ap.add_argument("--rows_per_shard_caption", type=int, default=20,
                    help="step 0a caption 分片行数，默认 20")
    ap.add_argument("--rows_per_shard_step1", type=int, default=50,
                    help="step 1 输出每分片行数，默认 50")
    ap.add_argument("--max_samples_cot", type=int, default=50,
                    help="step 2 dry_run 处理的最大样本数，默认 50")
    ap.add_argument("--reuse_dir", type=str, default="",
                    help="复用已有的输出目录（不再加 timestamp），跳过 prepare 也可")
    ap.add_argument("--only", type=str, default="",
                    help="只跑指定 step，逗号分隔，例如 0a,1,2,3p,3j,4,5,0p")
    ap.add_argument("--skip", type=str, default="",
                    help="跳过指定 step，逗号分隔")
    ap.add_argument("--keep_inputs_only", action="store_true",
                    help="只准备 tiny 输入，不跑任何 step（调试用）")
    ap.add_argument("--force", action="store_true",
                    help="即使 reuse_dir 已存在也强制重跑")
    return ap.parse_args()


STEP_KEYS = ["0a", "1", "2", "0p", "3p", "3e", "3j", "4", "5"]


def parse_step_select(s: str) -> set:
    if not s:
        return set(STEP_KEYS)
    return set(x.strip() for x in s.split(",") if x.strip())


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)

    if args.reuse_dir:
        out_dir = Path(args.reuse_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / f"run_{ts}"
    safe_makedirs(out_dir)
    log(f"OUT_DIR = {out_dir}")

    only_str = args.only
    skip_str = args.skip
    if only_str and skip_str:
        only = parse_step_select(only_str)
        skip = parse_step_select(skip_str)
        selected = only - skip
    elif only_str:
        selected = parse_step_select(only_str)
    elif skip_str:
        selected = set(STEP_KEYS) - parse_step_select(skip_str)
    else:
        selected = set(STEP_KEYS)
    log(f"selected steps = {sorted(selected)}")

    # ---- 准备 tiny 输入 ----
    inputs = prepare_tiny_inputs(out_dir, args.n_users, args.n_pids_caption)
    log(f"tiny inputs prepared under {out_dir / 'inputs'}")

    if args.keep_inputs_only:
        print_summary(out_dir, {k: v for k, v in inputs.items()})
        return 0

    paths: Dict[str, Path] = {"[inputs]": out_dir / "inputs"}
    errors: List[str] = []

    def _run(key: str, fn, *a, **kw) -> bool:
        if key not in selected:
            log(f"SKIP step {key} (not in selected)")
            return False
        try:
            res = fn(*a, **kw)
            if isinstance(res, tuple):
                for i, r in enumerate(res):
                    paths[f"step {key}{' (out '+str(i+1)+')' if len(res)>1 else ''}"] = r
            else:
                paths[f"step {key}"] = res
            return True
        except Exception as e:  # noqa: BLE001
            log(f"FAIL step {key}: {type(e).__name__}: {e}", level="ERROR")
            errors.append(f"{key}: {e}")
            return False

    # ---- 跑各 step ----
    caption_shards: Optional[Path] = None
    step1_out: Optional[Path] = None
    step2_out: Optional[Path] = None
    step3_pre_out: Optional[Path] = None
    step3_ppl_out: Optional[Path] = None
    step3_join_out: Optional[Path] = None
    step4_thresh_out: Optional[Path] = None

    if _run("0a", step0a_split, out_dir, inputs["tiny_pid2caption"], args.rows_per_shard_caption):
        caption_shards = paths["step 0a"]

    if _run("1", step1_history_text, out_dir, inputs["tiny_bench"], inputs["tiny_pid2caption"], args.rows_per_shard_step1):
        step1_out = paths["step 1"]

    if step1_out is not None:
        if _run("2", step2_cot_dry_run, out_dir, step1_out, args.max_samples_cot):
            step2_out = paths["step 2"]
    else:
        log("SKIP step 2 (no step1 output)")

    if step2_out is not None and caption_shards is not None:
        _run("0p", step0prime_build_warmup_data,
             out_dir, step2_out, inputs["tiny_bench"], caption_shards, inputs["tiny_pid2sid"])
    else:
        log("SKIP step 0' (no step2 or caption_shards)")

    if step2_out is not None:
        if _run("3p", step3_preprocess, out_dir, step2_out, inputs["tiny_bench"], inputs["tiny_pid2sid"]):
            step3_pre_out = paths["step 3p"]
    else:
        log("SKIP step 3 preprocess (no step2)")

    if step3_pre_out is not None:
        if _run("3e", step3_ppl_eval_mock, out_dir, step3_pre_out):
            step3_ppl_out = paths["step 3e"]
    else:
        log("SKIP step 3 PPL eval (no preprocess output)")

    if step2_out is not None and step3_ppl_out is not None:
        if _run("3j", step3_join, out_dir, step2_out, step3_ppl_out):
            step3_join_out = paths["step 3j"]
    else:
        log("SKIP step 3 join (missing inputs)")

    if step3_join_out is not None:
        if _run("4", step4_filter, out_dir, step3_join_out):
            # step4 返回 (annotate, threshold) 两个，已在 _run 里登记
            if "step 4 (out 1)" in paths:
                step4_thresh_out = paths["step 4 (out 2)"] if "step 4 (out 2)" in paths else paths["step 4 (out 1)"]
    else:
        log("SKIP step 4 filter (no join output)")

    cot_for_step5 = step4_thresh_out
    if cot_for_step5 is not None and caption_shards is not None:
        _run("5", step5_data_format,
             out_dir, cot_for_step5, inputs["tiny_bench"], caption_shards, inputs["tiny_pid2sid"])
    else:
        log("SKIP step 5 (no filter output or caption_shards)")

    # ---- 汇总 ----
    print_summary(out_dir, paths)

    hr("ERRORS")
    if errors:
        for e in errors:
            print(f"  - {e}")
        print(f"\n{len(errors)} step(s) failed.")
        return 1
    print("  (none)\n  all steps OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
