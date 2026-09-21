"""
Caption-to-SID Task
Input: caption parquet (pid, dense_caption) + pid2sid parquet
Output: LLM SFT training format parquet

Task: Given a dense video caption, generate its video SID.
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from caption_task_utils import (
    load_pid2sid_map,
    load_pid_allowlist,
    normalize_caption,
    pid_to_sid,
    resolve_input_files,
)

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一名视频检索助手，需要根据视频描述找出对应的视频token。",
    "你是一个专业的视频内容定位助手，能够根据文字描述匹配正确的视频token。",
    "你擅长根据视频内容描述识别对应的视频token，请直接输出最匹配的视频token。",
    "作为视频内容理解助手，你需要把给定的视频描述映射为对应的视频token。",
    "你是一个视频标识预测专家，请根据描述返回准确的视频token。",
]

# User prompts (Chinese)
USER_PROMPTS = [
    "请根据下面的视频描述，输出对应的视频token：\n{caption}",
    "这段视频的内容描述如下，请给出它对应的视频token：\n{caption}",
    "请阅读以下视频描述，并返回唯一对应的视频token：\n{caption}",
    "下面是一个视频的完整描述，请输出该视频的视频token：\n{caption}",
    "根据这段视频描述，识别并输出对应的视频token：\n{caption}",
    "请将以下视频内容说明转换为对应的视频token：\n{caption}",
]


def build_messages(caption: str, sid: str, prompt_rng: random.Random) -> str:
    """Build chat messages."""
    system_prompt = prompt_rng.choice(SYSTEM_PROMPTS)
    user_prompt = prompt_rng.choice(USER_PROMPTS).format(caption=caption)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": sid}]},
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_record(
    pid,
    dense_caption,
    pid2sid: dict,
    prompt_rng: random.Random,
    allowed_pids: Optional[set[int]] = None,
) -> Optional[dict]:
    """Process a single caption -> sid sample."""
    if allowed_pids is not None and int(pid) not in allowed_pids:
        return None

    dense_caption = normalize_caption(dense_caption)
    if not dense_caption:
        return None

    sid = pid_to_sid(pid, pid2sid)
    if not sid:
        return None

    return {
        'source': 'RecIF_CaptionToSid',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(dense_caption, sid, prompt_rng),
        'metadata': json.dumps({'pid': int(pid), 'sid': sid}, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser(description="Caption to SID Task Data Processing")
    parser.add_argument('--input', type=str, required=True, help='Input caption parquet path or shard directory')
    parser.add_argument('--pid2sid', type=str, required=True, help='pid2sid mapping parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of output samples (0 = no limit)')
    parser.add_argument('--sample_strategy', choices=['head', 'reservoir'], default='head',
                        help='head: stop early after max_samples; reservoir: uniform streaming sample')
    parser.add_argument('--shuffle_files', action='store_true',
                        help='Shuffle input parquet file order before reading')
    parser.add_argument('--allow_pid_file', type=str, default='',
                        help='Optional newline-delimited pid allowlist')
    args = parser.parse_args()

    prompt_rng = random.Random(args.seed)
    sample_rng = random.Random(args.seed + 1)
    file_rng = random.Random(args.seed + 2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = resolve_input_files(
        Path(args.input),
        shuffle_files=args.shuffle_files,
        file_rng=file_rng,
    )

    print(f"Loading pid2sid from {args.pid2sid}...")
    pid2sid = load_pid2sid_map(args.pid2sid)
    print(f"  Loaded {len(pid2sid):,} mappings")
    allowed_pids = load_pid_allowlist(args.allow_pid_file)
    if allowed_pids is not None:
        print(f"  Loaded pid allowlist: {len(allowed_pids):,} items")

    print(f"Resolved {len(input_files):,} input parquet file(s) from {args.input}")
    if args.max_samples > 0:
        print(f"  max_samples={args.max_samples:,}, sample_strategy={args.sample_strategy}")
    if args.shuffle_files:
        print(f"  shuffle_files=True (seed={args.seed + 2})")

    print("Processing...")
    results = []
    valid_seen = 0
    total_rows = 0
    stop_early = False
    file_pbar = tqdm(input_files, desc='Caption shards', unit='file')
    for file_path in file_pbar:
        df = pd.read_parquet(file_path, columns=['pid', 'dense_caption'])
        total_rows += len(df)
        file_pbar.set_postfix(rows=f'{total_rows:,}', kept=len(results))

        for row in df.itertuples(index=False):
            result = process_record(
                row.pid,
                row.dense_caption,
                pid2sid,
                prompt_rng,
                allowed_pids=allowed_pids,
            )
            if result is None:
                continue

            valid_seen += 1
            if args.max_samples <= 0:
                results.append(result)
                continue

            if args.sample_strategy == 'head':
                results.append(result)
                if len(results) >= args.max_samples:
                    stop_early = True
                    break
            else:
                if len(results) < args.max_samples:
                    results.append(result)
                else:
                    replace_idx = sample_rng.randint(0, valid_seen - 1)
                    if replace_idx < args.max_samples:
                        results[replace_idx] = result

        if stop_early:
            break
    file_pbar.close()

    df_output = pd.DataFrame(results)
    output_path = output_dir / 'train.parquet'
    df_output.to_parquet(output_path, index=False)

    print(f"Scanned {total_rows:,} raw rows, kept {len(df_output):,} samples from {valid_seen:,} valid rows")
    print(f"Saved: {output_path} ({len(df_output):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
