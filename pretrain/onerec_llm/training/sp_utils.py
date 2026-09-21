"""sp_utils.py

Sequence Parallel (Ulysses) batch processing utilities.

核心逻辑：
  SP group 内 sp_size 个 rank 各从 dataloader 取一个 batch，
  通过 all-gather 汇聚成列表 [batch_0, ..., batch_{sp_size-1}]，
  每个 step 所有 rank 处理同一个 batch（比如 step 0 取 batch_0），
  并沿 seq 方向切分取 local_S = S // sp_size 段，用于 SP forward。
  一个 all-gather 产出 sp_size 个 step 的数据，缓存在内部逐步消耗。

  sp_size=1 时所有操作退化为 identity，与原训练逻辑完全一致。
"""

import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ─── 轻量滚动计时器 ──────────────────────────────────────────────────────────

class _RollingTimer:
    """记录最近 window 次操作的耗时（ms），提供 last/avg/max 查询。"""

    def __init__(self, name: str, window: int = 20):
        self.name = name
        self._window = window
        self._buf: deque = deque(maxlen=window)
        self._t0: float = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """记录并返回本次耗时（ms）。"""
        elapsed = (time.perf_counter() - self._t0) * 1000.0
        self._buf.append(elapsed)
        return elapsed

    @property
    def avg_ms(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._buf) if self._buf else 0.0

    @property
    def last_ms(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: last={self.last_ms:.1f} avg={self.avg_ms:.1f} "
            f"max={self.max_ms:.1f} ms (n={len(self._buf)})"
        )


# ─── SP 进程组查询 ──────────────────────────────────────────────────────────

def get_sp_group() -> Optional[dist.ProcessGroup]:
    """获取当前 rank 所属的 SP 进程组（来自 modeling_qwen3_5_sp 全局变量）。"""
    try:
        from onerec_llm.models.qwen3_5.modeling_qwen3_5_sp import get_sequence_parallel_group
        return get_sequence_parallel_group()
    except ImportError:
        return None


def get_sp_rank() -> int:
    """获取当前 rank 在 SP 组内的序号。sp_size=1 时返回 0。"""
    grp = get_sp_group()
    if grp is None:
        return 0
    try:
        return dist.get_rank(group=grp)
    except Exception:
        return 0


def get_sp_size() -> int:
    """获取 SP 组大小。未启用 SP 时返回 1。"""
    grp = get_sp_group()
    if grp is None:
        return 1
    try:
        return dist.get_world_size(group=grp)
    except Exception:
        return 1


# ─── batch all-gather ────────────────────────────────────────────────────────

def allgather_batch_as_list(
    batch: Dict[str, torch.Tensor],
    sp_group: Optional[dist.ProcessGroup] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    在 SP 组内 all-gather 每个 rank 的 batch，返回列表
    [batch_sp_rank_0, ..., batch_sp_rank_{sp_size-1}]。

    使用 dist.all_gather_object 直接传输整个 batch dict，
    无需手动序列化，支持 tensor 和任意 Python 对象（如 data_source: List[str]）。

    sp_size=1 时返回 [batch]（单元素列表）。
    """
    sp_size = get_sp_size()
    if sp_size <= 1:
        return [batch]

    if sp_group is None:
        sp_group = get_sp_group()

    result = [None] * sp_size
    dist.all_gather_object(result, batch, group=sp_group)
    return result


# ─── batch 沿 seq 维度切分 ──────────────────────────────────────────────────

# 这些 key 对应的 tensor 没有 seq 维度（dim=1），切分时保持原样。
# 注意：sample_idx 的 shape 是 (1, S)，需要随 seq 切分（不在此集合中）。
# cu_seqlens 描述完整序列边界，是 shape (N+1,) 的 1D tensor，所有 rank 保持相同。
NON_SEQ_KEYS = {
    "cu_seqlens",
    "data_source",
    "epoch_idx",
}


def split_batch_along_seq(
    batch: Dict[str, torch.Tensor],
    sp_rank: int,
    sp_size: int,
) -> Dict[str, torch.Tensor]:
    """
    对 batch 中所有带 sequence 维度（dim=1，shape=(B, S, ...)）的 tensor，
    沿 dim=1 切取 sp_rank 对应的 local_S = S // sp_size 段。
    没有 sequence 维度的 tensor 保持不变。

    sp_size=1 时返回原 batch（identity）。
    """
    if sp_size <= 1:
        return batch

    local_batch: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if k in NON_SEQ_KEYS:
            local_batch[k] = v
            continue

        if v.ndim >= 2 and v.shape[1] > 1:
            S = v.shape[1]
            assert S % sp_size == 0, (
                f"Cannot split key '{k}': shape={tuple(v.shape)}, "
                f"S={S} not divisible by sp_size={sp_size}"
            )
            local_S = S // sp_size
            start = sp_rank * local_S
            end = start + local_S
            local_batch[k] = v[:, start:end]
        else:
            local_batch[k] = v

    return local_batch


# ─── SPBatchIterator ────────────────────────────────────────────────────────

class SPBatchIterator:
    """
    封装 SP batch 处理逻辑的迭代器。

    工作流程（sp_size=4 示例）：
      1. SP 组内 4 rank 各从 dataloader 取一个 raw_batch
      2. all-gather 得到 [batch_0, batch_1, batch_2, batch_3]
      3. 缓存这 4 个 batch，逐 step 消耗：
         - step 0: 所有 rank 处理 batch_0（沿 seq 切分）
         - step 1: 所有 rank 处理 batch_1（沿 seq 切分）
         - step 2: 所有 rank 处理 batch_2（沿 seq 切分）
         - step 3: 所有 rank 处理 batch_3（沿 seq 切分）
      4. 缓存消耗完后，触发下一轮 all-gather

    sp_size=1 时完全退化为直接 next(data_iter)，与原训练逻辑一致。

    用法：
        data_iter = iter(dataloader)
        sp_iter = SPBatchIterator(data_iter)

        while True:
            try:
                batch = sp_iter.next()
            except StopIteration:
                break
            compute_forward_backward(model, batch, ...)
    """

    def __init__(self, dataloader_iter, debug: bool = False, timing_log_freq: int = 20):
        self._data_iter = dataloader_iter
        self._sp_size = get_sp_size()
        self._sp_rank = get_sp_rank()
        self._sp_group = get_sp_group()
        # 缓存：all-gather 后的 batch 列表，逐 step 消耗
        self._cached_batches: List[Dict[str, torch.Tensor]] = []
        # 当前消耗到缓存中的第几个 batch（0 ~ sp_size-1）
        self._cache_idx: int = 0
        # 调试模式：打印切分前后各 tensor 形状（只打印前几步）
        self._debug = debug
        self._debug_steps = 0
        # ── 时间监控开关 ─────────────────────────────────────────────────
        # 环境变量 SP_BATCH_TIMING_LOG 控制：
        #   "0"/"off"/"false" → 关闭 timing log（生产训练推荐）
        #   "1"/"on"/未设置   → 开启 timing log（默认）
        _env = os.environ.get("SP_BATCH_TIMING_LOG", "1").lower()
        self._timing_enabled: bool = _env not in ("0", "off", "false", "no")
        # 时间监控：三个阶段各自的滚动计时器
        # dataload/allgather 每 sp_size 步触发一次，split 每步触发一次
        # 使用相同 window，窗口内记录次数不同，avg/max 仍有意义
        self._t_dataload  = _RollingTimer("dataload",   window=timing_log_freq)
        self._t_allgather = _RollingTimer("allgather",  window=timing_log_freq)
        self._t_split     = _RollingTimer("seq_split",  window=timing_log_freq)
        # 本次 step 是否触发了 _refill_cache（决定日志中是否打印 dataload/allgather 时间）
        self._did_refill_this_step: bool = False
        # 总调用次数（控制打印频率）
        self._step_cnt: int = 0
        self._timing_log_freq = timing_log_freq

    def _refill_cache(self):
        """从 dataloader 取 batch 并 all-gather，填充缓存。"""
        # ── 阶段 1：从 dataloader 取 raw batch ──────────────────────────────
        if self._timing_enabled:
            self._t_dataload.start()
        try:
            raw_batch = next(self._data_iter)
        except StopIteration:
            raise
        if self._timing_enabled:
            self._t_dataload.stop()

        # ── 阶段 2：SP 组内 all-gather ───────────────────────────────────────
        if self._timing_enabled:
            self._t_allgather.start()
        if self._sp_size <= 1:
            self._cached_batches = [raw_batch]
        else:
            self._cached_batches = allgather_batch_as_list(raw_batch, sp_group=self._sp_group)
        if self._timing_enabled:
            self._t_allgather.stop()

        self._cache_idx = 0
        self._did_refill_this_step = True

    def next(self) -> Dict[str, torch.Tensor]:
        """获取下一个 batch（已沿 seq 切分）。"""
        self._step_cnt += 1
        self._did_refill_this_step = False

        # 缓存已消耗完，需要 refill（含 dataload + allgather 计时）
        if self._cache_idx >= len(self._cached_batches):
            self._refill_cache()

        # 取当前 idx 对应的 batch
        full_batch = self._cached_batches[self._cache_idx]
        self._cache_idx += 1

        # ── 阶段 3：沿 seq 维度切分 ──────────────────────────────────────────
        if self._timing_enabled:
            self._t_split.start()
        local_batch = split_batch_along_seq(full_batch, self._sp_rank, self._sp_size)
        if self._timing_enabled:
            self._t_split.stop()

        # ── 时间日志：前 5 步每步打，之后每 timing_log_freq 步打一次 ────────
        _is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        _should_log = self._timing_enabled and _is_rank0 and (
            self._step_cnt <= 5 or self._step_cnt % self._timing_log_freq == 0
        )
        if _should_log:
            input_shape = tuple(local_batch.get("input_ids", torch.empty(0)).shape)
            # dataload/allgather 只在 refill 步有新值，其余步显示 N/A
            if self._did_refill_this_step:
                dl_str = f"dataload={self._t_dataload.last_ms:.1f}ms (avg {self._t_dataload.avg_ms:.1f}ms)"
                ag_str = f"allgather={self._t_allgather.last_ms:.1f}ms (avg {self._t_allgather.avg_ms:.1f}ms)"
            else:
                dl_str = f"dataload=N/A (avg {self._t_dataload.avg_ms:.1f}ms)"
                ag_str = f"allgather=N/A (avg {self._t_allgather.avg_ms:.1f}ms)"
            logger.warning(
                f"[SPBatchIterator timing step={self._step_cnt} "
                f"sp={self._sp_rank}/{self._sp_size}]  "
                f"{dl_str}  |  {ag_str}  |  "
                f"seq_split={self._t_split.last_ms:.1f}ms  |  "
                f"local input_ids={input_shape}"
            )

        # ── 调试模式：打印 shape 信息（只打印前 3 步） ───────────────────────
        if self._debug and self._debug_steps < 3 and _is_rank0:
            self._debug_steps += 1
            step = self._debug_steps
            lines = [f"[SPBatchIterator debug step={step} sp_rank={self._sp_rank}/{self._sp_size}]"]
            lines.append("  ── full_batch (after all-gather, before split) ──")
            for k, v in sorted(full_batch.items()):
                if isinstance(v, torch.Tensor):
                    lines.append(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    lines.append(f"    {k}: {type(v).__name__} = {v}")
            lines.append("  ── local_batch (after seq split) ──")
            for k, v in sorted(local_batch.items()):
                if isinstance(v, torch.Tensor):
                    lines.append(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    lines.append(f"    {k}: {type(v).__name__} = {v}")
            logger.warning("\n".join(lines))

        return local_batch

    def timing_summary(self) -> str:
        """返回三阶段耗时的滚动统计摘要（可在训练结束时打印）。"""
        return (
            f"SPBatchIterator timing summary (last {self._timing_log_freq} steps):\n"
            f"  {self._t_dataload.summary()}\n"
            f"  {self._t_allgather.summary()}\n"
            f"  {self._t_split.summary()}"
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()