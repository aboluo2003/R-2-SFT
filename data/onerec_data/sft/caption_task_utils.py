"""Shared helpers for caption-based SFT tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'


def pid_to_sid(pid, pid2sid: Dict[int, object]) -> str:
    """Convert a single pid to SID string."""
    if pid not in pid2sid:
        return ""
    code = pid2sid[pid]
    return SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])


def normalize_caption(dense_caption) -> Optional[str]:
    """Normalize caption text and drop empty / null values."""
    if dense_caption is None or pd.isna(dense_caption):
        return None
    dense_caption = str(dense_caption).strip()
    if not dense_caption:
        return None
    return dense_caption


def resolve_input_files(
    input_path: Path,
    shuffle_files: bool = False,
    file_rng: Optional[random.Random] = None,
) -> List[Path]:
    """Resolve parquet inputs from a single file or a shard directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob('part-*.parquet'))
        if not files:
            files = sorted(input_path.glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'No parquet files found under directory: {input_path}')
        if shuffle_files:
            if file_rng is None:
                raise ValueError('file_rng is required when shuffle_files=True')
            file_rng.shuffle(files)
        return files
    raise FileNotFoundError(f'Input path does not exist: {input_path}')


def load_pid2sid_map(pid2sid_path: str) -> Dict[int, object]:
    """Load pid -> sid-code mapping from parquet."""
    df_pid2sid = pd.read_parquet(pid2sid_path)
    return dict(zip(df_pid2sid['pid'], df_pid2sid['sid']))


def load_pid_allowlist(path: Optional[str]) -> Optional[Set[int]]:
    """Load a newline-delimited pid allowlist."""
    if not path:
        return None
    allowlist_path = Path(path)
    if not allowlist_path.exists():
        raise FileNotFoundError(f'Allowlist file not found: {allowlist_path}')

    pids: Set[int] = set()
    with allowlist_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pids.add(int(line))
    return pids


def dump_pid_list(path: Path, pids: List[int]) -> None:
    """Write pid list as newline-delimited text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for pid in pids:
            f.write(f"{pid}\n")


def dump_json(path: Path, payload: dict) -> None:
    """Write json payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
