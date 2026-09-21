"""
Item Understand Task
Input: caption parquet (pid, dense_caption) + pid2sid parquet
Output: LLM SFT training format parquet

Task: Given a video SID, generate its description/caption.
"""

import pandas as pd
import argparse
import json
import uuid
import random
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from caption_task_utils import (
    load_pid2sid_map,
    load_pid_allowlist,
    normalize_caption,
    pid_to_sid,
    resolve_input_files,
)

# System prompts (Chinese)
SYSTEM_PROMPTS = [
    "你是一名视频描述生成器，请根据下面的视频token生成视频描述。",
    "你是一个专业的视频内容分析助手，能够理解视频token并生成准确的描述。",
    "你是一位视频理解专家，擅长将视频token转换为详细的文字描述。",
    "作为视频内容解析助手，你需要根据视频token提供精准的内容描述。",
    "你是一个智能视频解说员，可以根据视频token创建生动的描述。",
    "你具备理解视频token并生成高质量描述的能力。",
    "你是视频内容描述专家，能够将视频token转化为易懂的文字说明。",
    "作为AI视频分析助手，你可以根据视频token生成详细准确的描述。",
]

# User prompts (Chinese)
USER_PROMPTS = [
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


def build_messages(sid: str, caption: str, prompt_rng: random.Random) -> str:
    """Build messages format JSON string."""
    system_prompt = prompt_rng.choice(SYSTEM_PROMPTS)
    user_prompt = prompt_rng.choice(USER_PROMPTS).format(sid=sid)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    ]
    return json.dumps(messages, ensure_ascii=False)


def process_record(
    pid,
    dense_caption,
    pid2sid: dict,
    prompt_rng: random.Random,
    allowed_pids: Optional[set[int]] = None,
) -> dict:
    """Process a single record."""
    if allowed_pids is not None and int(pid) not in allowed_pids:
        return None

    dense_caption = normalize_caption(dense_caption)
    if not dense_caption:
        return None

    # Convert pid to SID
    sid = pid_to_sid(pid, pid2sid)
    if not sid:
        return None

    return {
        'source': 'RecIF_ItemUnderstand',
        'uuid': str(uuid.uuid4()),
        'messages': build_messages(sid, dense_caption, prompt_rng),
        'metadata': json.dumps({'pid': int(pid), 'sid': sid}, ensure_ascii=False)
    }


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Item Understand Task Data Processing")
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

    # 1. Load pid2sid mapping
    print(f"Loading pid2sid from {args.pid2sid}...")
    pid2sid = load_pid2sid_map(args.pid2sid)
    print(f"  Loaded {len(pid2sid):,} mappings")
    allowed_pids = load_pid_allowlist(args.allow_pid_file)
    if allowed_pids is not None:
        print(f"  Loaded pid allowlist: {len(allowed_pids):,} items")

    # 2. Load caption data
    print(f"Resolved {len(input_files):,} input parquet file(s) from {args.input}")
    if args.max_samples > 0:
        print(f"  max_samples={args.max_samples:,}, sample_strategy={args.sample_strategy}")
    if args.shuffle_files:
        print(f"  shuffle_files=True (seed={args.seed + 2})")

    # 3. Process data
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

    # 4. Save results
    df_output = pd.DataFrame(results)
    output_path = output_dir / 'train.parquet'
    df_output.to_parquet(output_path, index=False)

    print(f"Scanned {total_rows:,} raw rows, kept {len(df_output):,} samples from {valid_seen:,} valid rows")
    print(f"Saved: {output_path} ({len(df_output):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
