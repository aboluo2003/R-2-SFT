"""Training logging utilities.

Provides TrainingMetrics for metrics accumulation and TensorBoardLogger
for async TensorBoard writing.
"""

import collections
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# DataSourceMetrics — result of cross-rank reduction of per-datasource stats
# ---------------------------------------------------------------------------

@dataclass
class DataSourceMetrics:
    """Holds cross-rank-reduced per-data-source statistics for one logging period."""
    loss:           Dict[str, float] = field(default_factory=dict)
    itemic_loss:    Dict[str, float] = field(default_factory=dict)
    text_loss:      Dict[str, float] = field(default_factory=dict)
    tokens:         Dict[str, int]   = field(default_factory=dict)
    itemic_tokens:  Dict[str, int]   = field(default_factory=dict)
    text_tokens:    Dict[str, int]   = field(default_factory=dict)
    valid_tokens:   Dict[str, int]   = field(default_factory=dict)
    samples:        Dict[str, int]   = field(default_factory=dict)   # local only (no cross-rank)


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------

class TrainingMetrics:
    """Manages training metrics accumulation and statistics.
    
    Tracks metrics in two ways:
    - Period metrics (period_*): accumulated over a logging period
    - Total  metrics (total_*):  accumulated over the entire training run
    """

    def __init__(self):
        self.reset_period_accumulators()
        # Running totals across the whole training run
        self.total_num_tokens: int = 0
        self.total_num_samples: int = 0
        self.total_num_valid_tokens: int = 0
        self.total_data_source_tokens: Dict[str, int] = collections.defaultdict(int)

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    def reset_period_accumulators(self) -> None:
        """Reset metrics for the current logging period."""
        self.period_sum_loss:              float = 0.0
        self.period_sum_itemic_token_loss: float = 0.0
        self.period_sum_text_token_loss:   float = 0.0
        self.period_num_tokens:            int   = 0
        self.period_num_itemic_tokens:     int   = 0
        self.period_num_text_tokens:       int   = 0
        self.period_num_samples:           int   = 0
        self.period_num_valid_tokens:      int   = 0
        self.period_num_steps:             int   = 0
        # Per-data-source accumulators (local, before cross-rank reduction)
        self.period_data_source_loss          = collections.defaultdict(float)
        self.period_data_source_itemic_loss   = collections.defaultdict(float)
        self.period_data_source_text_loss     = collections.defaultdict(float)
        self.period_data_source_tokens        = collections.defaultdict(int)
        self.period_data_source_itemic_tokens = collections.defaultdict(int)
        self.period_data_source_text_tokens   = collections.defaultdict(int)
        self.period_valid_data_source_tokens  = collections.defaultdict(int)
        self.local_period_data_source_samples = collections.defaultdict(int)

    def update(self, num_tokens: int, num_samples: int, num_valid_tokens: int) -> None:
        """Update both period and total token / sample counters."""
        self.period_num_tokens       += num_tokens
        self.period_num_samples      += num_samples
        self.period_num_valid_tokens += num_valid_tokens
        self.total_num_tokens        += num_tokens
        self.total_num_samples       += num_samples
        self.total_num_valid_tokens  += num_valid_tokens

    # ------------------------------------------------------------------
    # Cross-rank reduction
    # ------------------------------------------------------------------

    def reduce_datasource_metrics(self) -> DataSourceMetrics:
        """All-reduce per-data-source accumulators across ranks and return result.

        Side effect: updates ``self.total_data_source_tokens`` with the
        reduced token counts for the current period.
        """
        from onerec_llm.utils.common import dist_reduce_dict   # local import to avoid circulars

        ds = DataSourceMetrics(
            loss          = dist_reduce_dict(self.period_data_source_loss),
            itemic_loss   = dist_reduce_dict(self.period_data_source_itemic_loss),
            text_loss     = dist_reduce_dict(self.period_data_source_text_loss),
            tokens        = dist_reduce_dict(self.period_data_source_tokens),
            itemic_tokens = dist_reduce_dict(self.period_data_source_itemic_tokens),
            text_tokens   = dist_reduce_dict(self.period_data_source_text_tokens),
            valid_tokens  = dist_reduce_dict(self.period_valid_data_source_tokens),
            samples       = dist_reduce_dict(self.local_period_data_source_samples, group=None),
        )
        # Accumulate into training-wide totals
        for key, count in ds.tokens.items():
            self.total_data_source_tokens[key] += count
        return ds

    # ------------------------------------------------------------------
    # Log-dict construction
    # ------------------------------------------------------------------

    def build_log_dict(
        self,
        *,
        global_step:       int,
        grad_norm:         float,
        learning_rate:     float,
        period_duration:   float,
        training_duration: float,
        mfu_stats,                   # MFUStats instance
        epoch_idx:         int,
    ) -> dict:
        """Build the standard training-step log dictionary.

        All perf calculations (current / avg rates, loss averages) are
        encapsulated here so the training script does not need to expose
        intermediate variables.
        """
        world_size   = dist.get_world_size()
        period_steps = max(self.period_num_steps, 1)
        total_tokens = max(self.total_num_tokens, 1)

        # ── Current-period rates ────────────────────────────────────────
        sec_per_step                       = period_duration / period_steps
        tokens_per_sec_per_gpu_current     = self.period_num_tokens       / period_duration / world_size
        samples_per_sec_per_gpu_current    = self.period_num_samples      / period_duration / world_size
        samples_per_step_per_gpu_current   = self.period_num_samples      / period_steps    / world_size
        valid_toks_per_sec_per_gpu_current = self.period_num_valid_tokens / period_duration / world_size

        # ── Training-average rates ──────────────────────────────────────
        active_steps                 = max(global_step, 1)
        training_secs                = max(training_duration, 1e-6)
        samples_per_step_per_gpu_avg = self.total_num_samples / world_size / active_steps
        samples_per_sec_per_gpu_avg  = self.total_num_samples / world_size / training_secs
        tokens_per_step_per_gpu_avg  = self.total_num_tokens  / world_size / active_steps
        tokens_per_sec_per_gpu_avg   = self.total_num_tokens  / world_size / training_secs

        # ── Loss averages ───────────────────────────────────────────────
        avg_loss              = self.period_sum_loss              / period_steps
        avg_itemic_token_loss = self.period_sum_itemic_token_loss / period_steps
        avg_text_token_loss   = self.period_sum_text_token_loss   / period_steps

        return {
            # Training losses
            "training/loss":               avg_loss,
            "training/itemic_token_loss":  avg_itemic_token_loss,
            "training/text_token_loss":    avg_text_token_loss,
            "training/itemic_token_ratio": (
                self.period_num_itemic_tokens /
                max(self.period_num_itemic_tokens + self.period_num_text_tokens, 1)
            ),
            "training/grad_norm":          grad_norm,
            "training/learning_rate":      learning_rate,
            # Perf — current period
            "perf/sec_per_step":                           sec_per_step,
            "perf/tokens_per_sec_per_gpu_current":         tokens_per_sec_per_gpu_current,
            "perf/samples_per_sec_per_gpu_current":        samples_per_sec_per_gpu_current,
            "perf/samples_per_step_per_gpu_current":       samples_per_step_per_gpu_current,
            "perf/num_sample_per_sec_per_gpu":             samples_per_sec_per_gpu_current,  # compat alias
            "perf/valid_tokens_per_sec_per_gpu_current":   valid_toks_per_sec_per_gpu_current,
            # Perf — running totals
            "perf/total_num_tokens":   self.total_num_tokens,
            "perf/total_num_samples":  self.total_num_samples,
            "perf/num_sample_per_gpu": self.total_num_samples / world_size,
            "perf/valid_total_num_tokens": self.total_num_valid_tokens,
            "perf/valid_token_ratio":  self.total_num_valid_tokens / total_tokens,
            # MFU
            **mfu_stats.mfu(period_duration, global_step),
            # Perf — training averages
            "perf/samples_per_step_per_gpu_avg": samples_per_step_per_gpu_avg,
            "perf/samples_per_sec_per_gpu_avg":  samples_per_sec_per_gpu_avg,
            "perf/tokens_per_step_per_gpu_avg":  tokens_per_step_per_gpu_avg,
            "perf/tokens_per_sec_per_gpu_avg":   tokens_per_sec_per_gpu_avg,
            "perf/epoch_idx":                    epoch_idx,
        }


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------

class TensorBoardLogger:
    """Manages TensorBoard logging in a background thread."""

    def __init__(self, tb_writer: Optional[SummaryWriter]) -> None:
        self.tb_writer = tb_writer
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread: Optional[threading.Thread] = None

        if tb_writer is not None and dist.get_rank() == 0:
            self._thread = threading.Thread(
                target=self._write_loop,
                args=(tb_writer, self._queue),
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        global_step:    int,
        log_dict:       dict,
        ticker_stats:   dict,
        ds_metrics:     DataSourceMetrics | dict | None = None,
        *legacy_dicts:  dict,
        monitor_loss:   bool = False,
        monitor_cnt:    bool = False,
    ) -> None:
        """Queue one logging event for async writing.

        Supports **two calling conventions** for backward compatibility:

        **New style** (tok2)::

            tb_logger.log(global_step, log_dict, ticker_stats,
                          ds_metrics, monitor_loss=..., monitor_cnt=...)

        **Legacy style** (sp)::

            tb_logger.log(global_step, log_dict, ticker_stats,
                          period_data_source_loss,
                          period_data_source_itemic_loss,
                          period_data_source_text_loss,
                          period_data_source_itemic_tokens,
                          period_data_source_text_tokens,
                          period_data_source_tokens,
                          total_data_source_samples)

        When *ds_metrics* is a ``DataSourceMetrics`` the new path is used;
        when it is a plain ``dict`` (or ``None``) the legacy dicts are
        assembled into a ``DataSourceMetrics`` internally.
        """
        if self.tb_writer is None:
            return

        # ── Detect calling convention ────────────────────────────────────
        if isinstance(ds_metrics, DataSourceMetrics):
            # New style: ds_metrics already a DataSourceMetrics object
            final_ds = ds_metrics
        else:
            # Legacy style: 7 positional dicts passed after ticker_stats
            #   [loss, itemic_loss, text_loss, itemic_tokens, text_tokens, tokens, samples]
            if ds_metrics is not None:
                legacy_dicts = (ds_metrics,) + legacy_dicts  # shift first dict back in
            # Build DataSourceMetrics from the 7 legacy dicts
            _loss, _itemic_loss, _text_loss, _itemic_tokens, _text_tokens, _tokens, _samples = (
                legacy_dicts if len(legacy_dicts) == 7
                else ({}, {}, {}, {}, {}, {}, {})
            )
            final_ds = DataSourceMetrics(
                loss          = _loss,
                itemic_loss   = _itemic_loss,
                text_loss     = _text_loss,
                tokens        = _tokens,
                itemic_tokens = _itemic_tokens,
                text_tokens   = _text_tokens,
                samples       = _samples,
            )

        self._queue.put((
            global_step, log_dict, ticker_stats,
            final_ds, monitor_loss, monitor_cnt,
        ))

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    @staticmethod
    def _write_loop(tb_writer: SummaryWriter, q: queue.Queue) -> None:
        while True:
            global_step, log_dict, ticker_stats, ds, monitor_loss, monitor_cnt = q.get()
            total_samples      = log_dict.get("perf/total_num_samples", 0)
            total_valid_tokens = log_dict.get("perf/valid_total_num_tokens", 0)

            # Main scalar metrics
            for name, value in log_dict.items():
                if value is None:
                    continue
                tb_writer.add_scalar(name, value, global_step=global_step, new_style=True)
                if name.startswith("training/"):
                    tb_writer.add_scalar(
                        f"x_token_{name}", value,
                        global_step=total_valid_tokens, new_style=True,
                    )

            # Timing breakdown
            for name, value in ticker_stats.items():
                tb_writer.add_scalar(
                    f"ticker/{name}", value, global_step=global_step, new_style=True
                )

            if monitor_loss:
                for key, loss_sum in ds.loss.items():
                    tb_writer.add_scalar(
                        f"data_source_loss/{key}",
                        loss_sum / (ds.tokens.get(key, 0) + 1e-6),
                        global_step=global_step, new_style=True,
                    )
                for key, loss_sum in ds.itemic_loss.items():
                    tb_writer.add_scalar(
                        f"data_source_itemic_loss/{key}",
                        loss_sum / (ds.itemic_tokens.get(key, 0) + 1e-6),
                        global_step=global_step, new_style=True,
                    )
                for key, loss_sum in ds.text_loss.items():
                    tb_writer.add_scalar(
                        f"data_source_text_loss/{key}",
                        loss_sum / (ds.text_tokens.get(key, 0) + 1e-6),
                        global_step=global_step, new_style=True,
                    )

            if monitor_cnt:
                total_tokens = sum(ds.tokens.values()) or 1
                for key, samples in ds.samples.items():
                    tb_writer.add_scalar(
                        f"data_source_sample_ratio/{key}",
                        samples / (total_samples or 1),
                        global_step=global_step, new_style=True,
                    )
                for key, num_tokens in ds.tokens.items():
                    tb_writer.add_scalar(
                        f"data_source_token_ratio/{key}",
                        num_tokens / total_tokens,
                        global_step=global_step, new_style=True,
                    )


# ---------------------------------------------------------------------------
# StepLogger — orchestrates one logging event
# ---------------------------------------------------------------------------

class StepLogger:
    """Orchestrates a single training-step logging event.

    Owns the TensorBoard logger and knows how to format the console output.
    The training loop just calls ``log(step, metrics, ...)``.

    Usage::

        step_logger = StepLogger(tb_logger, args)
        period_start = step_logger.log(
            global_step, metrics, lr_scheduler, grad_norm,
            period_start_time, training_start_time,
            mfu_stats, step_tracker, iter_tracker, epoch_idx,
            chunked_loss_computer,
        )
    """

    def __init__(self, tb_logger: TensorBoardLogger, args) -> None:
        self.tb_logger = tb_logger
        self.args      = args

    def log(
        self,
        global_step:            int,
        metrics:                "TrainingMetrics",
        lr_scheduler,
        grad_norm:              float,
        period_start_time:      float,
        training_start_time:    float,
        mfu_stats,
        step_time_tracker,
        iteration_time_tracker,
        epoch_idx:              int,
        chunked_loss_computer=None,
        lr_mgr=None,
    ) -> float:
        """Run one logging event and return the new *period_start_time*.

        All perf / loss calculations are delegated to
        ``TrainingMetrics.build_log_dict`` and
        ``TrainingMetrics.reduce_datasource_metrics``.
        """
        import time
        from onerec_llm.utils.common import print_rank_0

        end_time      = time.time()
        staged_lrs = lr_mgr.current_lrs() if lr_mgr is not None else None
        learning_rate = staged_lrs["llm"] if staged_lrs is not None else lr_scheduler.get_last_lr()[0]

        log_dict  = metrics.build_log_dict(
            global_step       = global_step,
            grad_norm         = grad_norm,
            learning_rate     = learning_rate,
            period_duration   = end_time - period_start_time,
            training_duration = end_time - training_start_time,
            mfu_stats         = mfu_stats,
            epoch_idx         = epoch_idx,
        )

        # ── Staged LR: merge kmeans / vq / llm into log_dict ──────────
        if staged_lrs is not None:
            log_dict["staged_lr/kmeans"] = staged_lrs["kmeans"]
            log_dict["staged_lr/vq"]     = staged_lrs["vq"]
            log_dict["staged_lr/llm"]    = staged_lrs["llm"]
        ds_metrics   = metrics.reduce_datasource_metrics()
        ticker_stats = {
            k: v
            for t in [step_time_tracker, iteration_time_tracker]
            for k, v in t.stat().items()
        }

        self.tb_logger.log(
            global_step,
            log_dict,
            ticker_stats,
            ds_metrics,
            monitor_loss = self.args.monitor_datasource_loss,
            monitor_cnt  = self.args.monitor_datasource_cnt,
        )

        try:
            from onerec_llm.utils.ds_utils import format_dict_or_list
            lr_str = (
                f"LR[kmeans={staged_lrs['kmeans']:.2e}, vq={staged_lrs['vq']:.2e}, "
                f"llm={staged_lrs['llm']:.2e}]"
                if staged_lrs is not None
                else f"LR: {learning_rate:.2e}"
            )
            print_rank_0(
                f"Step: {global_step}, Loss: {log_dict['training/loss']:.4f}, "
                f"{lr_str}, Grad Norm: {grad_norm:.4f}, "
                f"Sec/Step: {log_dict['perf/sec_per_step']:.4f}",
                format_dict_or_list(log_dict),
                "\n",
                format_dict_or_list({
                    "mfu_stats":         mfu_stats.mfu_per_step_per_gpu,
                    "step_time_tracker": step_time_tracker.stat(),
                }),
                "\n",
                chunked_loss_computer.ticker.stat() if chunked_loss_computer else "",
            )
        except Exception:
            print_rank_0(
                f"Step: {global_step}, Loss: {log_dict['training/loss']:.4f}"
            )

        return end_time
