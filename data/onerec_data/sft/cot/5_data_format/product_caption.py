"""Product domain sid2caption / caption2sid task construction (0723 新增).

与 video/ad 域的 caption 任务（main.py build_caption_records）结构一致，但使用
product 域的 schema：
  - caption 来源：goods_pid2caption（列 new_pid + caption，★ key 是 new_pid 不是 old_pid）
  - sid 映射：product_pid2sid（key = new_pid，product pid 空间）

实现方式：复用 main.py 里已参数化的 reservoir_sample_disjoint_pids /
build_caption_records（两者都接受 pid_column / caption_column），通过依赖注入
把这两个函数传进来，避免循环导入（main.py 是脚本入口，模块名是 __main__）。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple


def sample_product_caption_pids(
    product_caption_files: List[Path],
    product_pid2sid: Dict[int, Any],
    sid2caption_size: int,
    caption2sid_size: int,
    seed: int,
    rank: int,
    comm: Any,
    reservoir_sample_fn: Callable,
) -> Tuple[Set[int], Set[int]]:
    """reservoir-sample disjoint product pids for sid2caption / caption2sid.

    与 main.py run() 里的 video/ad 采样逻辑结构一致，但：
      - valid_pids 用 product_pid2sid.keys()（product pid 空间）
      - pid_column='new_pid', caption_column='caption'（goods_pid2caption schema）
      - seed 偏移 +101/+201 与 video/ad 的 +1/+20 错开，避免两域采样到相同 pid
    """
    total_target = sid2caption_size + caption2sid_size
    sid2caption_pids: Set[int] = set()
    caption2sid_pids: Set[int] = set()
    if total_target <= 0:
        return sid2caption_pids, caption2sid_pids

    valid_pids = set(product_pid2sid.keys())

    if comm is not None:
        if rank == 0:
            sampled, _seen = reservoir_sample_fn(
                product_caption_files,
                valid_pids=valid_pids,
                total_target=total_target,
                sample_rng=random.Random(seed + 101),
                rank=rank,
                pid_column="new_pid",
                caption_column="caption",
            )
            if len(sampled) < total_target:
                raise RuntimeError(
                    f"valid unique product pids 不足: need {total_target}, got {len(sampled)}"
                )
            split_rng = random.Random(seed + 201)
            split_rng.shuffle(sampled)
            s2c_list = sampled[:sid2caption_size]
            c2s_list = sampled[sid2caption_size:total_target]
        else:
            s2c_list = None
            c2s_list = None
        s2c_list = comm.bcast(s2c_list, root=0)
        c2s_list = comm.bcast(c2s_list, root=0)
    else:
        sampled, _seen = reservoir_sample_fn(
            product_caption_files,
            valid_pids=valid_pids,
            total_target=total_target,
            sample_rng=random.Random(seed + 101),
            rank=rank,
            pid_column="new_pid",
            caption_column="caption",
        )
        if len(sampled) < total_target:
            raise RuntimeError(
                f"valid unique product pids 不足: need {total_target}, got {len(sampled)}"
            )
        split_rng = random.Random(seed + 201)
        split_rng.shuffle(sampled)
        s2c_list = sampled[:sid2caption_size]
        c2s_list = sampled[sid2caption_size:total_target]

    return set(s2c_list), set(c2s_list)


def build_product_caption_records(
    product_caption_files: List[Path],
    product_pid2sid: Dict[int, Any],
    sid2caption_pids: Set[int],
    caption2sid_pids: Set[int],
    rank: int,
    world_size: int,
    build_caption_fn: Callable,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build sid2caption / caption2sid records for product domain pids.

    复用 main.py 的 build_caption_records，传 product 列名（new_pid / caption）
    和 product_pid2sid（product pid → sid 映射）。★ 0728 修复：传 domain="product"
    使用商品主题 prompt（PRODUCT_SID2CAPTION_* / PRODUCT_CAPTION2SID_*），
    旧 bug 复用 video "视频"prompt → assistant 输出商品描述但 prompt 问"视频"，
    prompt-content mismatch，污染 product sid 语义。
    """
    return build_caption_fn(
        caption_files=product_caption_files,
        pid2sid=product_pid2sid,
        sid2caption_pids=sid2caption_pids,
        caption2sid_pids=caption2sid_pids,
        rank=rank,
        world_size=world_size,
        pid_column="new_pid",
        caption_column="caption",
        domain="product",  # 0728 修复：用"商品"prompt，旧 bug 复用 video"视频"prompt
    )
