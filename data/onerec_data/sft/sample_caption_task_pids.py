"""
Sample disjoint pid allowlists for caption-based SFT tasks.

Input: caption parquet / shard dir + pid2sid parquet
Output: newline-delimited pid lists for sid2caption / caption2sid
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from caption_task_utils import (
    dump_json,
    dump_pid_list,
    load_pid2sid_map,
    normalize_caption,
    resolve_input_files,
)


def reservoir_sample_disjoint_pids(
    input_files: List[Path],
    valid_pid_set: set[int],
    sample_size: int,
    sample_rng: random.Random,
) -> tuple[List[int], int, int]:
    """Reservoir-sample unique valid pids across caption shards."""
    if sample_size <= 0:
        return [], 0, 0

    sampled: List[int] = []
    seen_unique = 0
    raw_rows = 0
    seen_valid_pid = set()

    file_pbar = tqdm(input_files, desc='Sampling caption shards', unit='file')
    for file_path in file_pbar:
        df = pd.read_parquet(file_path, columns=['pid', 'dense_caption'])
        raw_rows += len(df)
        file_pbar.set_postfix(rows=f'{raw_rows:,}', unique=seen_unique)

        for row in df.itertuples(index=False):
            pid = int(row.pid)
            if pid not in valid_pid_set:
                continue
            if not normalize_caption(row.dense_caption):
                continue
            if pid in seen_valid_pid:
                continue
            seen_valid_pid.add(pid)

            seen_unique += 1
            if len(sampled) < sample_size:
                sampled.append(pid)
                continue

            replace_idx = sample_rng.randint(0, seen_unique - 1)
            if replace_idx < sample_size:
                sampled[replace_idx] = pid
        file_pbar.set_postfix(rows=f'{raw_rows:,}', unique=seen_unique)
    file_pbar.close()

    return sampled, seen_unique, raw_rows


def main():
    parser = argparse.ArgumentParser(description="Sample disjoint pid allowlists for caption tasks")
    parser.add_argument('--input', type=str, required=True, help='Caption parquet path or shard directory')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for pid lists')
    parser.add_argument('--sid2caption_size', type=int, default=0, help='Target sid2caption sample count')
    parser.add_argument('--caption2sid_size', type=int, default=0, help='Target caption2sid sample count')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shuffle_files', action='store_true', help='Shuffle input file order before scanning')
    args = parser.parse_args()

    total_target = args.sid2caption_size + args.caption2sid_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_rng = random.Random(args.seed + 10)
    split_rng = random.Random(args.seed + 20)
    input_files = resolve_input_files(
        Path(args.input),
        shuffle_files=args.shuffle_files,
        file_rng=file_rng,
    )

    print(f"Loading pid2sid from {args.pid2sid}...")
    pid2sid = load_pid2sid_map(args.pid2sid)
    valid_pid_set = {int(pid) for pid in pid2sid.keys()}
    print(f"  Loaded {len(valid_pid_set):,} valid pids")
    print(f"Resolved {len(input_files):,} input parquet file(s) from {args.input}")
    print(f"Sampling target: sid2caption={args.sid2caption_size:,}, caption2sid={args.caption2sid_size:,}")
    if args.shuffle_files:
        print(f"  shuffle_files=True (seed={args.seed + 10})")

    if total_target <= 0:
        raise ValueError('At least one target size must be greater than 0')

    sampled_pids, valid_unique_rows, raw_rows = reservoir_sample_disjoint_pids(
        input_files=input_files,
        valid_pid_set=valid_pid_set,
        sample_size=total_target,
        sample_rng=random.Random(args.seed + 1),
    )

    if len(sampled_pids) < total_target:
        raise RuntimeError(
            f'Not enough valid unique pids to satisfy disjoint sampling: '
            f'need {total_target:,}, got {len(sampled_pids):,}'
        )

    split_rng.shuffle(sampled_pids)
    sid2caption_pids = sampled_pids[:args.sid2caption_size]
    caption2sid_pids = sampled_pids[args.sid2caption_size:args.sid2caption_size + args.caption2sid_size]

    sid2caption_path = output_dir / 'sid2caption_pids.txt'
    caption2sid_path = output_dir / 'caption2sid_pids.txt'
    dump_pid_list(sid2caption_path, sid2caption_pids)
    dump_pid_list(caption2sid_path, caption2sid_pids)

    stats = {
        'seed': args.seed,
        'shuffle_files': args.shuffle_files,
        'input': args.input,
        'pid2sid': args.pid2sid,
        'raw_rows_scanned': raw_rows,
        'valid_unique_rows': valid_unique_rows,
        'sid2caption_size': len(sid2caption_pids),
        'caption2sid_size': len(caption2sid_pids),
        'sid2caption_pid_file': str(sid2caption_path),
        'caption2sid_pid_file': str(caption2sid_path),
    }
    dump_json(output_dir / 'stats.json', stats)

    print(f"Scanned {raw_rows:,} raw rows, found {valid_unique_rows:,} valid unique items")
    print(f"Saved sid2caption pid list: {sid2caption_path} ({len(sid2caption_pids):,} items)")
    print(f"Saved caption2sid pid list: {caption2sid_path} ({len(caption2sid_pids):,} items)")
    print(f"Saved stats: {output_dir / 'stats.json'}")


if __name__ == '__main__':
    main()
