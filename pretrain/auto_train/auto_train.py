#!/usr/bin/env python3
"""Auto-training + auto-evaluation orchestrator for OpenOneRec.

This script is adapted from the original OnerecPretrain/auto_train.py. It lives
inside ``OpenOneRec/pretrain/auto_train/`` and orchestrates a sequence of
training jobs defined in a YAML config, then -- once all training is done --
boots up the Ray-based benchmark cluster and evaluates every (job, think-mode)
combination with ``OpenOneRec/benchmarks/eval_rec_domains.sh``.

High-level flow
---------------
1. Load ``config.yaml`` (positional arg).
2. Pre-flight: all OUTPUT_DIRs must be unique. For any OUTPUT_DIR that
   already exists, detect existing ``step{N}`` checkpoints and offer to
   **skip** training for that job (keeping the existing checkpoints so the
   eval phase can still run on them); if the user declines to skip, prompt
   to delete the OUTPUT_DIR and retrain. ``--yes-delete-existing`` forces
   deletion without prompting (never skips).
3. For each experiment:
     a. Launch its ``run_script`` from ``REPO_ROOT`` (the ``pretrain`` dir).
     b. Poll ``OUTPUT_DIR`` every ``--poll-seconds`` for new ``step{N}`` dirs.
     c. For every step that is a multiple of ``--save_checkpoint_per_step``,
        convert the checkpoint via ``auto_train/convert.sh``.
     d. When a checkpoint at ``step >= max_step`` is converted, kill the
        ``mpirun --allow-run-as-root`` process bound to this OUTPUT_DIR.
     e. Record the run in ``auto_train/finished/{label}.txt``.
4. After all training jobs finish, run the evaluation phase:
     a. ``cd OpenOneRec/benchmarks && bash scripts/mpi_install.sh``
     b. ``bash scripts/init_ray_cluster.sh``
     c. For every job whose ``eval_think``/``eval_nothink`` is true, run
        ``bash eval_rec_domains.sh <converted_model> <eval_name> <true|false>``.
        By default only the final converted checkpoint is evaluated. Set
        ``eval_all_checkpoints: true`` (or pass ``--eval-all-checkpoints``) to
        evaluate **every** converted checkpoint; the step number is appended to
        ``eval_name`` (e.g. ``rec_demo_step240``) so result dirs don't collide.
     d. After each eval, parse its ``eval_results.json`` and append a row to
        ``auto_train/eval_results_summary.log``.
5. Finally, print a combined stdout summary of every eval result.

Note: ``./to_eval/{label}.txt`` is intentionally NOT written (the TODO in
``requirements.txt`` removed it).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# This file lives at OpenOneRec/pretrain/auto_train/auto_train.py
AUTO_TRAIN_DIR = Path(__file__).resolve().parent
# REPO_ROOT is the dir training scripts are launched from (the "pretrain" dir,
# which contains recipes/, examples/, set_env.sh, ...).
REPO_ROOT = AUTO_TRAIN_DIR.parent
# Benchmarks live next to pretrain/ (i.e. OpenOneRec/benchmarks).
BENCHMARK_DIR = REPO_ROOT.parent / "benchmarks"

CONVERT_SCRIPT = AUTO_TRAIN_DIR / "convert.sh"
EVAL_REC_DOMAINS_SCRIPT = BENCHMARK_DIR / "eval_rec_domains.sh"
MPI_INSTALL_SCRIPT = BENCHMARK_DIR / "scripts" / "mpi_install.sh"
INIT_RAY_CLUSTER_SCRIPT = BENCHMARK_DIR / "scripts" / "init_ray_cluster.sh"

LOGS_DIR = AUTO_TRAIN_DIR / "logs"
FINISHED_DIR = AUTO_TRAIN_DIR / "finished"
EVAL_SUMMARY_LOG = AUTO_TRAIN_DIR / "eval_results_summary.log"

DEFAULT_POLL_SECONDS = 60
LAUNCH_TIMEOUT_SECONDS = 300
TERMINATE_TIMEOUT_SECONDS = 15
DEFAULT_DATA_VERSION = "v1.0"
STEP_DIR_RE = re.compile(r"^step(\d+)$")

LOG_FILE_HANDLE: Any = None


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Job dataclass
# --------------------------------------------------------------------------- #
@dataclass
class TrainJob:
    label: str
    run_script: Path
    max_step: int
    output_dir_expr: str
    output_dir: Path
    save_checkpoint_per_step: int
    convert_script: Path
    # Eval config (TODO additions)
    eval_name: str
    eval_think: bool = False
    eval_nothink: bool = False
    # When True, evaluate every converted checkpoint at the end (not just the
    # last one). The step number is appended to eval_name to keep result dirs
    # unique.
    eval_all_checkpoints: bool = False
    # Optional explicit list of step numbers to evaluate. When set (non-empty),
    # overrides ``eval_all_checkpoints``: only the requested steps that actually
    # got converted are evaluated. Useful for long runs where you only care
    # about a few specific checkpoints (e.g. [6600, 8250] out of 9900).
    eval_steps: Optional[list[int]] = None
    # Optional comma-separated/list domain subset for eval_rec_domains.sh.
    # Supported values: ad, product, video. None means evaluate all three domains.
    eval_domains: Optional[list[str]] = None
    # Mutable state filled in during monitoring
    last_converted_step: Optional[int] = None
    last_step: int = 0
    # Ordered list of all steps that were successfully converted.
    converted_steps: list[int] = field(default_factory=list)
    # Set by pre-flight when the user chooses to keep existing checkpoints and
    # skip launching training for this job; the eval phase still runs on the
    # existing converted checkpoints.
    skip_training: bool = False

    @property
    def finished_file(self) -> Path:
        return FINISHED_DIR / f"{self.label}.txt"

    def step_dir(self, step: int) -> Path:
        return self.output_dir / f"step{step}"

    def checkpoint_input_dir(self, step: int) -> Path:
        return self.step_dir(step) / f"global_step{step}"

    def converted_dir(self, step: int) -> Path:
        return self.step_dir(step) / "converted"

    @property
    def has_eval(self) -> bool:
        return bool(self.eval_name) and (self.eval_think or self.eval_nothink)

    def eval_model_path(self, step: Optional[int] = None) -> Optional[Path]:
        if step is None:
            if self.last_converted_step is None:
                return None
            step = self.last_converted_step
        return self.converted_dir(step)

    def eval_name_for_step(self, step: Optional[int]) -> str:
        """Build the eval_name passed to eval_rec_domains.sh.

        ``step=None``  -> plain ``eval_name`` (used when eval_all_checkpoints is
        False, i.e. only the final checkpoint is evaluated).
        ``step=N``    -> ``f"{eval_name}_step{N}"`` so result dirs don't collide
        when evaluating multiple checkpoints.
        """
        if step is None:
            return self.eval_name
        return f"{self.eval_name}_step{step}"

    def result_name(self, think_mode: str, step: Optional[int] = None) -> str:
        return f"{self.eval_name_for_step(step)}_{think_mode}"

    def eval_results_path(
        self, think_mode: str, data_version: str, step: Optional[int] = None
    ) -> Path:
        return (
            BENCHMARK_DIR
            / "results"
            / data_version
            / f"results_{self.result_name(think_mode, step)}"
            / "eval_results.json"
        )

    def steps_to_eval(self) -> list[Optional[int]]:
        """Return the list of steps (or [None]) to evaluate for this job.

        * ``eval_steps`` set (non-empty) -> only those requested steps that
          actually got converted (intersection with ``converted_steps``).
          ``eval_all_checkpoints`` is ignored in this case.
        * ``eval_all_checkpoints=True``  -> every step in ``converted_steps``
          (as actual ints; ``eval_name_for_step`` appends ``_step{N}``).
        * ``eval_all_checkpoints=False`` -> ``[None]`` (or ``[]`` if nothing
          was converted). ``step=None`` makes ``eval_name_for_step`` return the
          plain ``eval_name`` and ``eval_model_path`` fall back to
          ``last_converted_step``.
        """
        if self.eval_steps:
            requested = sorted(set(self.eval_steps))
            converted = set(self.converted_steps)
            return [s for s in requested if s in converted]
        if self.eval_all_checkpoints:
            return list(self.converted_steps)
        if self.last_converted_step is None:
            return []
        return [None]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(message: str) -> None:
    global LOG_FILE_HANDLE
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = message.splitlines() or [""]
    for line in lines:
        formatted = f"[{timestamp}] {line}"
        print(formatted, flush=True)
        if LOG_FILE_HANDLE is not None:
            LOG_FILE_HANDLE.write(formatted + "\n")
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.flush()


def init_log_file() -> Path:
    global LOG_FILE_HANDLE
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_FILE_HANDLE = log_path.open("a", encoding="utf-8")
    return log_path


def close_log_file() -> None:
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None


# --------------------------------------------------------------------------- #
# Minimal YAML parser (fallback when PyYAML is unavailable)
# --------------------------------------------------------------------------- #
def unquote_shell_value(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        return value
    if len(parts) == 1:
        return parts[0]
    return value


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value.rstrip()


def parse_scalar(value: str, lineno: int) -> Any:
    value = strip_inline_comment(value).strip()
    if not value:
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ConfigError(f"Line {lineno}: invalid quoted string: {value}") from exc
    return value


def parse_key_value(text: str, lineno: int) -> tuple[str, Any]:
    if ":" not in text:
        raise ConfigError(f"Line {lineno}: expected 'key: value', got: {text}")
    key, raw_value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Line {lineno}: empty key is not allowed")
    return key, parse_scalar(raw_value, lineno)


def parse_experiments_list(
    lines: list[tuple[int, int, str]], *, start_index: int, list_indent: int
) -> tuple[list[dict[str, Any]], int]:
    index = start_index
    experiments: list[dict[str, Any]] = []

    while index < len(lines):
        lineno, indent, text = lines[index]
        if indent < list_indent:
            break
        if indent != list_indent or not text.startswith("- "):
            raise ConfigError(f"Line {lineno}: expected list item with indent {list_indent}")

        item: dict[str, Any] = {}
        inline = text[2:].strip()
        if inline:
            key, value = parse_key_value(inline, lineno)
            item[key] = value
        index += 1

        while index < len(lines):
            next_lineno, next_indent, next_text = lines[index]
            if next_indent <= list_indent:
                break
            key, value = parse_key_value(next_text, next_lineno)
            item[key] = value
            index += 1

        experiments.append(item)

    return experiments, index


def parse_simple_yaml(config_path: Path) -> Any:
    lines: list[tuple[int, int, str]] = []
    for lineno, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(stripped)
        lines.append((lineno, indent, stripped))

    if not lines:
        raise ConfigError(f"{config_path} is empty")

    first_lineno, first_indent, first_text = lines[0]
    if first_text.startswith("- "):
        experiments, next_index = parse_experiments_list(
            lines, start_index=0, list_indent=first_indent
        )
        if next_index != len(lines):
            lineno, _, _ = lines[next_index]
            raise ConfigError(f"Line {lineno}: unexpected content after experiments list")
        return {"experiments": experiments}

    root: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        lineno, indent, text = lines[index]
        if indent != 0:
            raise ConfigError(f"Line {lineno}: expected top-level key with zero indentation")

        if text == "experiments:":
            index += 1
            if index >= len(lines):
                raise ConfigError(f"Line {lineno}: 'experiments:' must contain a list")
            list_indent = lines[index][1]
            experiments, index = parse_experiments_list(
                lines, start_index=index, list_indent=list_indent
            )
            root["experiments"] = experiments
            continue

        key, value = parse_key_value(text, lineno)
        root[key] = value
        index += 1

    return root


def load_yaml(config_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        data = parse_simple_yaml(config_path)
    else:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return {"experiments": data}
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")
    return data


# --------------------------------------------------------------------------- #
# Config -> TrainJob
# --------------------------------------------------------------------------- #
def resolve_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_training_script(run_script: Path) -> tuple[str, int]:
    content = run_script.read_text(encoding="utf-8")

    output_match = re.search(r"(?m)^\s*OUTPUT_DIR=(.+?)\s*$", content)
    if not output_match:
        raise ConfigError(f"Failed to parse OUTPUT_DIR from {run_script}")
    output_dir_expr = unquote_shell_value(output_match.group(1))

    save_match = re.search(r"--save_checkpoint_per_step\s+(\d+)", content)
    if not save_match:
        raise ConfigError(f"Failed to parse --save_checkpoint_per_step from {run_script}")
    save_step = int(save_match.group(1))
    if save_step <= 0:
        raise ConfigError(f"{run_script}: --save_checkpoint_per_step must be positive")

    return output_dir_expr, save_step


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer")
    return value


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a boolean (true/false)")
    return value


def _optional_int_list(value: Any, where: str) -> Optional[list[int]]:
    """Validate ``eval_steps`` style fields. Accepts null/missing or a list of
    positive ints. Returns ``None`` when unset, or a sorted deduped list.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list of positive integers (or null)")
    out: list[int] = []
    seen: set[int] = set()
    for idx, v in enumerate(value):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ConfigError(f"{where}[{idx}] must be a positive integer, got {v!r}")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return sorted(out)


def _optional_domain_list(value: Any, where: str) -> Optional[list[str]]:
    """Validate eval domain subset. Accepts null/missing, comma string, or list."""
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [x.strip() for x in value.split(",")]
    elif isinstance(value, list):
        raw_items = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise ConfigError(f"{where}[{idx}] must be a domain string, got {item!r}")
            raw_items.extend(x.strip() for x in item.split(","))
    else:
        raise ConfigError(f"{where} must be a comma-separated string/list of domains or null")

    allowed = {"ad", "product", "video"}
    out: list[str] = []
    seen: set[str] = set()
    for domain in raw_items:
        if not domain:
            continue
        if domain not in allowed:
            raise ConfigError(f"{where} contains unsupported domain {domain!r}; allowed: ad, product, video")
        if domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
    if not out:
        raise ConfigError(f"{where} must contain at least one domain")
    return out


def load_jobs(config_path: Path) -> tuple[list[TrainJob], dict[str, Any]]:
    data = load_yaml(config_path)
    experiments = data.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ConfigError("Config must contain a non-empty 'experiments' list")

    default_label = data.get("default_label")
    default_max_step = data.get("default_max_step")
    default_eval_name = data.get("default_eval_name")
    default_eval_think = data.get("default_eval_think", False)
    default_eval_nothink = data.get("default_eval_nothink", False)
    default_eval_all_checkpoints = data.get("default_eval_all_checkpoints", False)
    default_eval_steps = _optional_int_list(
        data.get("default_eval_steps"), "default_eval_steps"
    )
    default_eval_domains = _optional_domain_list(
        data.get("default_eval_domains"), "default_eval_domains"
    )
    data_version = data.get("data_version", DEFAULT_DATA_VERSION)

    if default_label is not None and (not isinstance(default_label, str) or not default_label.strip()):
        raise ConfigError("default_label must be a non-empty string")
    if default_max_step is not None and (not isinstance(default_max_step, int) or default_max_step <= 0):
        raise ConfigError("default_max_step must be a positive integer")
    if default_eval_name is not None and (not isinstance(default_eval_name, str) or not default_eval_name.strip()):
        raise ConfigError("default_eval_name must be a non-empty string")
    _require_bool(default_eval_think, "default_eval_think")
    _require_bool(default_eval_nothink, "default_eval_nothink")
    _require_bool(default_eval_all_checkpoints, "default_eval_all_checkpoints")
    if not isinstance(data_version, str) or not data_version.strip():
        raise ConfigError("data_version must be a non-empty string")

    jobs: list[TrainJob] = []
    output_dirs: dict[Path, str] = {}

    for index, item in enumerate(experiments, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"experiments[{index}] must be a mapping")

        label_value = item.get("label", default_label)
        run_script_text = item.get("run_script")
        max_step_value = item.get("max_step", default_max_step)
        eval_name_value = item.get("eval_name", default_eval_name)
        eval_think_value = item.get("eval_think", default_eval_think)
        eval_nothink_value = item.get("eval_nothink", default_eval_nothink)
        eval_all_checkpoints_value = item.get(
            "eval_all_checkpoints", default_eval_all_checkpoints
        )
        eval_steps_value = item.get("eval_steps", default_eval_steps)
        eval_domains_value = item.get("eval_domains", default_eval_domains)

        label = _require_str(label_value, f"experiments[{index}].label (or default_label)")
        run_script_text = _require_str(
            run_script_text, f"experiments[{index}].run_script"
        )
        max_step = _require_int(
            max_step_value, f"experiments[{index}].max_step (or default_max_step)"
        )
        if eval_name_value is None:
            eval_name_value = label
        eval_name = _require_str(
            eval_name_value, f"experiments[{index}].eval_name (or default_eval_name)"
        )
        eval_think = _require_bool(eval_think_value, f"experiments[{index}].eval_think")
        eval_nothink = _require_bool(
            eval_nothink_value, f"experiments[{index}].eval_nothink"
        )
        eval_all_checkpoints = _require_bool(
            eval_all_checkpoints_value, f"experiments[{index}].eval_all_checkpoints"
        )
        eval_steps = _optional_int_list(
            eval_steps_value, f"experiments[{index}].eval_steps (or default_eval_steps)"
        )
        eval_domains = _optional_domain_list(
            eval_domains_value, f"experiments[{index}].eval_domains (or default_eval_domains)"
        )

        run_script_path = resolve_path(run_script_text, base_dir=REPO_ROOT)
        if not run_script_path.exists():
            raise ConfigError(f"Run script path does not exist: {run_script_path}")

        run_scripts: list[Path]
        if run_script_path.is_dir():
            run_scripts = sorted(
                child
                for child in run_script_path.iterdir()
                if child.is_file() and child.suffix == ".sh"
            )
            if not run_scripts:
                raise ConfigError(f"Run script directory has no .sh files: {run_script_path}")
        else:
            run_scripts = [run_script_path]

        for run_script in run_scripts:
            actual_label = label
            output_dir_expr, save_step = parse_training_script(run_script)
            output_dir = resolve_path(output_dir_expr, base_dir=REPO_ROOT)
            if output_dir in output_dirs:
                prev_label = output_dirs[output_dir]
                raise ConfigError(
                    f"Duplicate OUTPUT_DIR detected between labels '{prev_label}' "
                    f"and '{actual_label}': {output_dir}"
                )
            output_dirs[output_dir] = actual_label

            jobs.append(
                TrainJob(
                    label=actual_label,
                    run_script=run_script,
                    max_step=max_step,
                    output_dir_expr=output_dir_expr,
                    output_dir=output_dir,
                    save_checkpoint_per_step=save_step,
                    convert_script=CONVERT_SCRIPT,
                    eval_name=eval_name,
                    eval_think=eval_think,
                    eval_nothink=eval_nothink,
                    eval_all_checkpoints=eval_all_checkpoints,
                    eval_steps=eval_steps,
                    eval_domains=eval_domains,
                )
            )

    return jobs, {"data_version": data_version}


def scan_converted_step_dirs(job: TrainJob) -> dict[int, Path]:
    """Return existing ``step{N}/converted`` dirs, even if raw checkpoints were cleaned.

    Auto-cleanup removes ``global_step{N}/`` after a successful conversion to save
    disk, leaving ``step{N}/converted`` as the only artifact needed for eval.  The
    eval-only flows (``--skip-train`` and the pre-flight "skip existing" choice)
    must therefore discover converted models independently of raw checkpoint
    inputs.
    """
    converted_dirs: dict[int, Path] = {}
    for step in scan_step_dirs(job.output_dir):
        converted_dir = job.converted_dir(step)
        if converted_dir.is_dir():
            converted_dirs[step] = converted_dir
    return converted_dirs


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #
def detect_existing_steps(job: TrainJob) -> dict[str, Any]:
    """Scan ``job.output_dir`` for existing checkpoint / converted step dirs.

    Returns a dict with:
      * ``step_dirs``:        ``{step: Path}`` for every ``step{N}`` dir found.
      * ``checkpoint_dirs``:  ``{step: Path}`` for every step that has a usable
                              ``global_step{N}`` (or fallback) checkpoint dir.
      * ``converted_steps``:  sorted list of steps whose ``converted/`` dir exists.
      * ``reached_max_step``: True if any checkpoint/converted step >= ``job.max_step``.
    """
    step_dirs = scan_step_dirs(job.output_dir)
    checkpoint_dirs = scan_checkpoint_input_dirs(job)
    converted_steps = sorted(scan_converted_step_dirs(job))
    reusable_steps = set(checkpoint_dirs) | set(converted_steps)
    reached_max_step = any(step >= job.max_step for step in reusable_steps)
    return {
        "step_dirs": step_dirs,
        "checkpoint_dirs": checkpoint_dirs,
        "converted_steps": converted_steps,
        "reached_max_step": reached_max_step,
    }


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def handle_existing_output_dirs(
    jobs: list[TrainJob], *, assume_yes: bool
) -> None:
    """Pre-flight handling of jobs whose ``OUTPUT_DIR`` already exists.

    For each such job, first detect whether any ``step{N}`` checkpoint dirs are
    already present:

      * If checkpoints exist, offer to **skip** training for this job (keeping
        the existing checkpoints so the eval phase can still consume them). If
        the user declines to skip, prompt to **delete** the ``OUTPUT_DIR`` and
        retrain; declining that as well aborts the whole run.

      * If no checkpoints exist (stale / empty ``OUTPUT_DIR``), fall back to the
        original behavior: prompt to delete it (or just delete when
        ``assume_yes`` is set).

    In ``assume_yes`` mode (``--yes-delete-existing``) nothing is skipped --
    every existing ``OUTPUT_DIR`` is removed unconditionally, matching the
    previous semantics.
    """
    for job in jobs:
        if not job.output_dir.exists():
            continue

        info = detect_existing_steps(job)
        step_dirs = info["step_dirs"]
        converted_steps = info["converted_steps"]

        if not step_dirs:
            # OUTPUT_DIR exists but no checkpoints yet -- stale dir, ask to remove.
            if not assume_yes:
                should_delete = prompt_yes_no(
                    f"OUTPUT_DIR already exists for label '{job.label}' (no "
                    f"checkpoints yet): {job.output_dir}\n"
                    "Delete it and continue? [y/N]: "
                )
                if not should_delete:
                    raise RuntimeError(
                        f"User declined to delete existing OUTPUT_DIR: {job.output_dir}"
                    )

            log(f"Removing existing OUTPUT_DIR for {job.label}: {job.output_dir}")
            remove_path(job.output_dir)
            continue

        # Existing checkpoints found -> offer to skip training for this job.
        step_summary = ", ".join(f"step{n}" for n in sorted(step_dirs))
        extras: list[str] = []
        if converted_steps:
            extras.append(
                "converted: " + ", ".join(f"step{n}" for n in converted_steps)
            )
        if info["reached_max_step"]:
            extras.append(f"max_step={job.max_step} reached")
        extra_tag = f" ({'; '.join(extras)})" if extras else ""

        if assume_yes:
            should_skip = False
        else:
            should_skip = prompt_yes_no(
                f"Existing checkpoints found for label '{job.label}': "
                f"{step_summary}{extra_tag}\n"
                f"  output_dir={job.output_dir}\n"
                "Skip training for this job and keep existing checkpoints? [y/N]: "
            )

        if should_skip:
            job.skip_training = True
            discover_converted_steps(job, log_tag="[skip-existing]")
            log(
                f"Skipping training for {job.label}; will reuse existing "
                f"checkpoints (converted_steps={job.converted_steps})."
            )
            continue

        # User declined to skip -> confirm deletion before retraining.
        if not assume_yes:
            should_delete = prompt_yes_no(
                f"Delete existing OUTPUT_DIR for '{job.label}' and retrain? [y/N]: "
            )
            if not should_delete:
                raise RuntimeError(
                    f"User declined both skip and delete for existing OUTPUT_DIR: "
                    f"{job.output_dir}"
                )

        log(f"Removing existing OUTPUT_DIR for {job.label}: {job.output_dir}")
        remove_path(job.output_dir)


def validate_runtime_prerequisites() -> None:
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(f"Convert script not found: {CONVERT_SCRIPT}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT.parent))
    except ValueError:
        return str(path)


def prompt_yes_no(prompt: str) -> bool:
    print(prompt, end="", flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(prompt)
        LOG_FILE_HANDLE.flush()
    response = input().strip().lower()
    log(f"Prompt response: {response or '<empty>'}")
    return response in {"y", "yes"}


# --------------------------------------------------------------------------- #
# Command execution
# --------------------------------------------------------------------------- #
def run_command_logged(
    command: list[str], *, cwd: Path, prefix: str, check: bool = True,
    env: Optional[dict[str, str]] = None,
) -> int:
    log(f"{prefix} command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        log(f"{prefix} | {line.rstrip()}")

    return_code = process.wait()
    if return_code != 0:
        msg = f"{prefix} failed with exit code {return_code}"
        if check:
            raise subprocess.CalledProcessError(return_code, command)
        log(msg)
    return return_code


# --------------------------------------------------------------------------- #
# Training launch + monitor
# --------------------------------------------------------------------------- #
def launch_training(job: TrainJob) -> None:
    log(f"Launching {job.label}: {display_path(job.run_script)}")
    process = subprocess.Popen(
        ["bash", str(job.run_script)],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        output, _ = process.communicate(input="\n", timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise RuntimeError(
            f"Timed out waiting for launch script to finish bootstrap stage: {job.run_script}"
        ) from exc

    if output.strip():
        log(f"Launch output for {job.label}:\n{output.rstrip()}")

    if process.returncode != 0:
        raise RuntimeError(
            f"Launch script failed for {job.label} with exit code {process.returncode}"
        )


def scan_step_dirs(output_dir: Path) -> dict[int, Path]:
    steps: dict[int, Path] = {}
    if not output_dir.exists():
        return steps

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = STEP_DIR_RE.match(child.name)
        if not match:
            continue
        steps[int(match.group(1))] = child
    return steps


def scan_checkpoint_input_dirs(job: TrainJob) -> dict[int, Path]:
    checkpoint_dirs: dict[int, Path] = {}
    for step, step_dir in scan_step_dirs(job.output_dir).items():
        checkpoint_dir = job.checkpoint_input_dir(step)
        if checkpoint_dir.is_dir():
            checkpoint_dirs[step] = checkpoint_dir
            continue

        matches = sorted(
            child
            for child in step_dir.iterdir()
            if child.is_dir() and child.name.startswith("global_step")
        )
        if len(matches) == 1:
            checkpoint_dirs[step] = matches[0]
    return checkpoint_dirs


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def run_conversion(job: TrainJob, step: int, checkpoint_dir: Path) -> None:
    log(f"Converting {job.label} checkpoint step{step}: {checkpoint_dir}")
    converted_dir = job.converted_dir(step)
    if converted_dir.exists():
        shutil.rmtree(converted_dir)

    command = ["bash", str(job.convert_script), str(checkpoint_dir), str(converted_dir)]
    run_command_logged(command, cwd=REPO_ROOT, prefix=f"convert[{job.label}:step{step}]")
    # NOTE: intentionally do NOT write ./to_eval/{label}.txt (removed per TODO).


# --------------------------------------------------------------------------- #
# Checkpoint cleanup (disk space reclamation)
# --------------------------------------------------------------------------- #
# Directory layout per step (under job.output_dir/step{N}/):
#   global_step{N}/   — raw DCP checkpoint (input to convert.sh; large)
#   optimizer_ckpt/   — optimizer + scheduler state (only needed to resume
#                        training; auto_train never resumes, so always safe
#                        to delete)
#   converted/         — HF format model (output of convert.sh; used by eval)
#
# Cleanup rules (same as scripts/clean_ckpts.py):
#   - converted/ exists  → delete global_step{N}/ + optimizer_ckpt/
#     (converted/ is kept for eval; raw DCP + optimizer state no longer needed)
#   - converted/ absent → delete optimizer_ckpt/ only
#     (keep global_step{N}/ so conversion can still run on a later poll/retry)
#
# optimizer_ckpt/ is always safe to delete: auto_train never resumes training,
# and convert.sh only reads global_step{N}/ (not optimizer_ckpt/).


def cleanup_step_dir(job: TrainJob, step: int, *, quiet: bool = False) -> None:
    """Free disk by deleting redundant checkpoint artifacts for one step.

    See module docstring above for the directory layout and cleanup rules.
    """
    step_dir = job.step_dir(step)
    if not step_dir.is_dir():
        return

    is_converted = job.converted_dir(step).is_dir()

    # Always delete optimizer_ckpt/ (not needed for eval or conversion)
    opt_path = step_dir / "optimizer_ckpt"
    if opt_path.is_dir():
        try:
            shutil.rmtree(opt_path)
            if not quiet:
                log(f"  [clean] {job.label} step{step}: removed optimizer_ckpt/")
        except Exception as exc:
            log(f"  [clean] WARNING: failed to remove {opt_path}: {exc}")

    # Delete global_step{N}/ only after conversion (eval uses converted/)
    if is_converted:
        gs_path = job.checkpoint_input_dir(step)
        if gs_path.is_dir():
            try:
                shutil.rmtree(gs_path)
                if not quiet:
                    log(
                        f"  [clean] {job.label} step{step}: "
                        f"removed {gs_path.name}/ (converted/ kept for eval)"
                    )
            except Exception as exc:
                log(f"  [clean] WARNING: failed to remove {gs_path}: {exc}")


def cleanup_all_steps(job: TrainJob) -> None:
    """Sweep every step dir under job.output_dir and clean each one.

    Called once at the end of training (or in --skip-train mode) to reclaim
    disk from optimizer_ckpt/ on unconverted steps that were never cleaned
    during the per-step conversion loop.
    """
    steps = scan_step_dirs(job.output_dir)
    if not steps:
        return
    cleaned = 0
    for step in sorted(steps):
        cleanup_step_dir(job, step, quiet=True)
        cleaned += 1
    log(f"[clean] {job.label}: swept {cleaned} step dir(s) for disk reclamation")


def find_mpirun_pids(job: TrainJob) -> list[int]:
    result = subprocess.run(
        ["ps", "-eww", "-o", "pid=", "-o", "args="],
        check=True,
        capture_output=True,
        text=True,
    )

    pids: list[int] = []
    output_dir_strings = {job.output_dir_expr, str(job.output_dir)}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s+(.*)$", line)
        if not match:
            continue
        pid = int(match.group(1))
        args = match.group(2)
        if "mpirun" not in args or "--allow-run-as-root" not in args:
            continue
        if any(
            output_dir_text and output_dir_text in args
            for output_dir_text in output_dir_strings
        ):
            pids.append(pid)
    return pids


def terminate_training(job: TrainJob) -> None:
    pids = find_mpirun_pids(job)
    if not pids:
        log(
            f"No matching mpirun process found for {job.label}; assuming it already exited."
        )
        return

    log(
        f"Stopping {job.label} mpirun process(es): "
        f"{', '.join(str(pid) for pid in pids)}"
    )
    for pid in pids:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + TERMINATE_TIMEOUT_SECONDS
    remaining = set(pids)
    while remaining and time.time() < deadline:
        time.sleep(1)
        still_running: set[int] = set()
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            still_running.add(pid)
        remaining = still_running

    for pid in remaining:
        log(f"Force killing stubborn pid {pid}")
        os.kill(pid, signal.SIGKILL)


def record_finished(job: TrainJob, last_step: int) -> None:
    append_line(
        job.finished_file,
        f"{datetime.now().isoformat(timespec='seconds')}\tstep={last_step}"
        f"\toutput_dir={job.output_dir}",
    )


def monitor_training(job: TrainJob, *, poll_seconds: int, clean_ckpts: bool = True) -> int:
    log(
        f"Monitoring {job.label}: output_dir={job.output_dir}, "
        f"save_checkpoint_per_step={job.save_checkpoint_per_step}, "
        f"max_step={job.max_step}, clean_ckpts={clean_ckpts}"
    )
    known_steps: set[int] = set()
    converted_steps: set[int] = set()
    last_step = 0
    idle_polls_without_mpirun = 0

    while True:
        checkpoint_dirs = scan_checkpoint_input_dirs(job)
        if checkpoint_dirs:
            last_step = max(last_step, max(checkpoint_dirs))

        new_steps = sorted(step for step in checkpoint_dirs if step not in known_steps)
        for step in new_steps:
            log(f"Detected new checkpoint for {job.label}: step{step}")
            known_steps.add(step)

        eligible_steps = sorted(
            step
            for step in checkpoint_dirs
            if step % job.save_checkpoint_per_step == 0 and step not in converted_steps
        )
        for step in eligible_steps:
            try:
                run_conversion(job, step, checkpoint_dirs[step])
            except subprocess.CalledProcessError as exc:
                log(
                    f"Conversion failed for {job.label} step{step} with exit code "
                    f"{exc.returncode}; will retry on the next poll."
                )
                continue
            converted_steps.add(step)
            job.converted_steps.append(step)
            job.last_converted_step = step
            log(f"Converted {job.label} step{step} -> {job.converted_dir(step)}")
            # ★ Disk reclamation: after conversion succeeds, the raw global_step{N}/
            #   and optimizer_ckpt/ are no longer needed (eval uses converted/).
            #   Delete them immediately to free disk during long training runs.
            if clean_ckpts:
                cleanup_step_dir(job, step)

        reached_target_steps = sorted(
            step for step in checkpoint_dirs if step >= job.max_step
        )
        if reached_target_steps:
            target_step = reached_target_steps[0]
            if (
                target_step % job.save_checkpoint_per_step != 0
                or target_step in converted_steps
            ):
                log(
                    f"{job.label} reached max_step via step{target_step}; "
                    f"stopping training."
                )
                terminate_training(job)
                job.last_step = max(last_step, target_step)
                # ★ Final sweep: reclaim optimizer_ckpt/ from any unconverted steps
                #   (e.g. steps whose conversion failed and never retried).
                if clean_ckpts:
                    cleanup_all_steps(job)
                return job.last_step

        pids = find_mpirun_pids(job)
        if pids:
            idle_polls_without_mpirun = 0
        else:
            idle_polls_without_mpirun += 1
            if idle_polls_without_mpirun >= 2 and last_step < job.max_step:
                raise RuntimeError(
                    f"Training process for {job.label} disappeared before reaching "
                    f"max_step. Last observed step: {last_step}"
                )
            if idle_polls_without_mpirun >= 2 and last_step >= job.max_step:
                job.last_step = last_step
                # ★ Final sweep: reclaim optimizer_ckpt/ from any unconverted steps.
                if clean_ckpts:
                    cleanup_all_steps(job)
                return last_step

        time.sleep(poll_seconds)


# --------------------------------------------------------------------------- #
# Evaluation phase
# --------------------------------------------------------------------------- #
def parse_eval_results(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    try:
        with results_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"Failed to read eval results {results_path}: {exc}")
        return {}


def summarize_eval_results(results: dict) -> list[tuple[str, str, str, str, str, str]]:
    """Return rows of (task, pass@32, pos1@32, recall@32, pid_pass@32, pid_recall@32)."""
    rows: list[tuple[str, str, str, str, str, str]] = []

    def fmt(v: Any) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    for model_name, model_results in results.items():
        if model_name.startswith("_"):
            continue
        if not isinstance(model_results, dict):
            continue
        for task_name in ["ad", "product", "video"]:
            if task_name not in model_results:
                continue
            for _split, m in model_results[task_name].items():
                if not isinstance(m, dict):
                    continue
                sid_pass = m.get("pass@32", "?")
                sid_pos1 = m.get("position1_pass@32", "?")
                sid_recall = m.get("recall@32", "?")
                pid_pass = m.get("pid_pass@32", "-")
                pid_recall = m.get("pid_recall@32", "-")
                rows.append(
                    (
                        task_name,
                        fmt(sid_pass),
                        fmt(sid_pos1),
                        fmt(sid_recall),
                        fmt(pid_pass),
                        fmt(pid_recall),
                    )
                )
    return rows


def append_eval_summary(
    job: TrainJob,
    think_mode: str,
    rows: list[tuple[str, str, str, str, str, str]],
    results_path: Path,
    *,
    step: Optional[int] = None,
) -> None:
    EVAL_SUMMARY_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_tag = f" | step{step}" if step is not None else ""
    model_path = job.eval_model_path(step)
    header = (
        f"\n========== [{timestamp}] {job.label}{step_tag} | {think_mode} ==========\n"
        f"model_path: {model_path}\n"
        f"results_path: {results_path}\n"
    )
    body = ""
    if rows:
        body = (
            f"{'Task':<12} {'pass@32':>10} {'pos1@32':>10} {'recall@32':>10}"
            f"   |   {'pid_pass@32':>12} {'pid_recall@32':>14}\n"
        )
        body += "-" * 80 + "\n"
        for task, sid_pass, sid_pos1, sid_recall, pid_pass, pid_recall in rows:
            body += (
                f"{task:<12} {sid_pass:>10} {sid_pos1:>10} {sid_recall:>10}"
                f"   |   {pid_pass:>12} {pid_recall:>14}\n"
            )
    else:
        body = "(no results rows parsed)\n"
    append_line(EVAL_SUMMARY_LOG, header + body.rstrip())


def run_eval_phase(
    jobs: list[TrainJob], *, data_version: str, dry_run: bool = False
) -> None:
    eval_jobs = [job for job in jobs if job.has_eval]
    if not eval_jobs:
        log("No jobs have evaluation configured (eval_think/eval_nothink both false "
            "or eval_name empty); skipping eval phase.")
        return

    log("=" * 70)
    log("Starting evaluation phase (all training jobs finished)")
    log("=" * 70)

    if dry_run:
        log("Dry run: would run mpi_install.sh, init_ray_cluster.sh, then "
            "eval_rec_domains.sh for:")
        for job in eval_jobs:
            steps = job.steps_to_eval()
            if not steps:
                log(f"  - {job.label}: no converted checkpoints to eval.")
                continue
            mode_desc = "ALL" if job.eval_all_checkpoints else "last-only"
            steps_desc = f", eval_steps={job.eval_steps}" if job.eval_steps else ""
            log(f"  - {job.label} (eval_all_checkpoints={mode_desc}{steps_desc}, "
                f"{len(steps)} checkpoint(s)):")
            for step in steps:
                model_path = job.eval_model_path(step)
                eval_name = job.eval_name_for_step(step)
                for think_mode, enabled in (
                    ("think", job.eval_think), ("nothink", job.eval_nothink)
                ):
                    if not enabled:
                        continue
                    domains_desc = ",".join(job.eval_domains) if job.eval_domains else "ad,product,video"
                    log(
                        f"      - [{think_mode}] step={step} model={model_path} "
                        f"eval_name={eval_name} enable_thinking="
                        f"{'true' if think_mode == 'think' else 'false'} "
                        f"domains={domains_desc}"
                    )
        return

    # Step 1: install benchmarks requirements via MPI
    if MPI_INSTALL_SCRIPT.exists():
        log(f"Running mpi_install.sh in {display_path(BENCHMARK_DIR)} ...")
        run_command_logged(
            ["bash", str(MPI_INSTALL_SCRIPT)],
            cwd=BENCHMARK_DIR,
            prefix="mpi_install",
            check=False,
        )
    else:
        log(f"mpi_install.sh not found at {MPI_INSTALL_SCRIPT}; skipping.")

    # Step 2: init ray cluster
    # NOTE: init_ray_cluster.sh uses `set -e` and its final command
    # (`ssh ... ray status`) sometimes returns non-zero even though the cluster
    # actually started fine (e.g. the SSH "Permanently added ... to known hosts"
    # warning, or `ray status` being flaky). So we treat a non-zero exit as a
    # warning, not a hard abort -- the user can still proceed to eval.
    if INIT_RAY_CLUSTER_SCRIPT.exists():
        log(f"Running init_ray_cluster.sh in {display_path(BENCHMARK_DIR)} ...")
        rc = run_command_logged(
            ["bash", str(INIT_RAY_CLUSTER_SCRIPT)],
            cwd=BENCHMARK_DIR,
            prefix="init_ray_cluster",
            check=False,
        )
        if rc != 0:
            log(
                f"WARNING: init_ray_cluster.sh exited with code {rc}. "
                f"Ray processes may still have started (the script's final "
                f"`ray status` step is known to be flaky). Proceeding with "
                f"eval anyway -- if eval fails to connect to Ray, check the "
                f"cluster logs at "
                f"{BENCHMARK_DIR / 'scripts' / 'logs' / 'ray'}."
            )
    else:
        log(f"init_ray_cluster.sh not found at {INIT_RAY_CLUSTER_SCRIPT}; skipping.")

    # Step 3: eval each (job, step, think-mode)
    if not EVAL_REC_DOMAINS_SCRIPT.exists():
        log(f"eval_rec_domains.sh not found at {EVAL_REC_DOMAINS_SCRIPT}; cannot run eval.")
        return

    # all_summaries: list of (job, step, think_mode, rows)
    all_summaries: list[tuple[TrainJob, Optional[int], str, list[tuple[str, str, str, str, str, str]]]] = []

    for job in eval_jobs:
        steps = job.steps_to_eval()
        if not steps:
            eval_steps_desc = (
                f", eval_steps={job.eval_steps}" if job.eval_steps else ""
            )
            log(
                f"Skipping eval for {job.label}: no converted checkpoints match "
                f"(last_converted_step={job.last_converted_step}, "
                f"converted_steps={job.converted_steps}{eval_steps_desc})."
            )
            continue

        mode_desc = "ALL" if job.eval_all_checkpoints else "last-only"
        steps_desc = f", eval_steps={job.eval_steps}" if job.eval_steps else ""
        log(
            f"Eval plan for {job.label}: {len(steps)} checkpoint(s) "
            f"(eval_all_checkpoints={mode_desc}{steps_desc}) "
            f"steps={steps} think={job.eval_think} nothink={job.eval_nothink} "
            f"domains={','.join(job.eval_domains) if job.eval_domains else 'ad,product,video'}"
        )

        for step in steps:
            model_path = job.eval_model_path(step)
            if model_path is None or not model_path.exists():
                log(
                    f"Skipping eval for {job.label} step={step}: converted model "
                    f"not found at {model_path}."
                )
                continue

            actual_eval_name = job.eval_name_for_step(step)

            for think_mode, enabled in (
                ("think", job.eval_think), ("nothink", job.eval_nothink)
            ):
                if not enabled:
                    continue
                enable_thinking = "true" if think_mode == "think" else "false"
                step_tag = f" step={step}" if step is not None else ""
                log(
                    f"Evaluating {job.label}{step_tag} ({think_mode}) "
                    f"model={model_path} eval_name={actual_eval_name} "
                    f"enable_thinking={enable_thinking}"
                )
                eval_env = os.environ.copy()
                if job.eval_domains:
                    eval_env["EVAL_DOMAINS"] = ",".join(job.eval_domains)
                try:
                    run_command_logged(
                        [
                            "bash",
                            str(EVAL_REC_DOMAINS_SCRIPT),
                            str(model_path),
                            actual_eval_name,
                            enable_thinking,
                        ],
                        cwd=BENCHMARK_DIR,
                        prefix=f"eval[{job.label}:step={step}:{think_mode}]",
                        check=True,
                        env=eval_env,
                    )
                except subprocess.CalledProcessError as exc:
                    log(
                        f"eval_rec_domains.sh failed for {job.label} "
                        f"step={step} ({think_mode}) with exit code {exc.returncode}; "
                        f"continuing."
                    )

                results_path = job.eval_results_path(
                    think_mode, data_version, step=step
                )
                results = parse_eval_results(results_path)
                rows = summarize_eval_results(results)
                append_eval_summary(
                    job, think_mode, rows, results_path, step=step
                )
                all_summaries.append((job, step, think_mode, rows))

    # Final stdout summary
    print()
    log("=" * 70)
    log("Evaluation summary")
    log("=" * 70)
    if not all_summaries:
        log("No evaluation results were collected.")
        return

    for job, step, think_mode, rows in all_summaries:
        step_tag = f" step={step}" if step is not None else ""
        result_name = job.result_name(think_mode, step=step)
        log(
            f"\n--- {job.label}{step_tag} | {think_mode} | "
            f"results={result_name} ---"
        )
        if rows:
            header = (
                f"{'Task':<12} {'pass@32':>10} {'pos1@32':>10} {'recall@32':>10}"
                f"   |   {'pid_pass@32':>12} {'pid_recall@32':>14}"
            )
            log(header)
            log("-" * 80)
            for task, sid_pass, sid_pos1, sid_recall, pid_pass, pid_recall in rows:
                log(
                    f"{task:<12} {sid_pass:>10} {sid_pos1:>10} {sid_recall:>10}"
                    f"   |   {pid_pass:>12} {pid_recall:>14}"
                )
        else:
            log("(no results rows parsed)")
    log(f"\nDetailed per-run summary log: {EVAL_SUMMARY_LOG}")
    log("Published baselines (OpenOneRec-8B, README.md):")
    log("  Video   Recall@32 = 0.0355")
    log("  Ad      Recall@32 = 0.0877")
    log("  Product Recall@32 = 0.0470")


# --------------------------------------------------------------------------- #
# Plan printing
# --------------------------------------------------------------------------- #
def print_plan(
    jobs: list[TrainJob], *, poll_seconds: int, data_version: str,
    clean_ckpts: bool = True,
) -> None:
    log(f"Loaded {len(jobs)} experiment(s); poll_seconds={poll_seconds}; "
        f"data_version={data_version}")
    log(f"REPO_ROOT={REPO_ROOT}")
    log(f"BENCHMARK_DIR={BENCHMARK_DIR}")
    log(f"CONVERT_SCRIPT={CONVERT_SCRIPT}")
    clean_desc = (
        "auto (delete global_step{N}/ + optimizer_ckpt/ after conversion; "
        "delete optimizer_ckpt/ for unconverted steps)"
        if clean_ckpts else "DISABLED (--no-clean-ckpts)"
    )
    log(f"Checkpoint cleanup: {clean_desc}")
    for job in jobs:
        eval_modes = []
        if job.eval_think:
            eval_modes.append("think")
        if job.eval_nothink:
            eval_modes.append("nothink")
        eval_desc = ",".join(eval_modes) if eval_modes else "(none)"
        all_ckpts = "ALL" if job.eval_all_checkpoints else "last-only"
        steps_desc = f" steps={job.eval_steps}" if job.eval_steps else ""
        domains_desc = f" domains={','.join(job.eval_domains)}" if job.eval_domains else " domains=ad,product,video"
        log(
            f"Plan label={job.label} run_script={display_path(job.run_script)} "
            f"output_dir={job.output_dir} save_every={job.save_checkpoint_per_step} "
            f"max_step={job.max_step} eval_name={job.eval_name} eval=[{eval_desc}] "
            f"eval_all_checkpoints={all_ckpts}{steps_desc}{domains_desc}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential auto training + eval launcher")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help=f"Polling interval in seconds (default: {DEFAULT_POLL_SECONDS})",
    )
    parser.add_argument(
        "--yes-delete-existing",
        action="store_true",
        help="Delete existing OUTPUT_DIR paths without prompting (never "
             "skips, even if checkpoints already exist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the execution plan without launching anything",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training phase (assume checkpoints already exist); "
             "go straight to evaluation. Requires prior converted checkpoints.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation phase (only run training).",
    )
    parser.add_argument(
        "--eval-all-checkpoints",
        dest="eval_all_checkpoints",
        action="store_true",
        default=None,
        help="Evaluate every converted checkpoint at the end (overrides config; "
             "appends _step{N} to eval_name so result dirs don't collide).",
    )
    parser.add_argument(
        "--no-eval-all-checkpoints",
        dest="eval_all_checkpoints",
        action="store_false",
        help="Only evaluate the final converted checkpoint at the end (overrides config).",
    )
    parser.add_argument(
        "--eval-steps",
        type=str,
        default=None,
        help="Comma-separated list of step numbers to evaluate (overrides config "
             "'eval_steps'; e.g. --eval-steps 6600,8250). Only steps that "
             "actually got converted are evaluated.",
    )
    parser.add_argument(
        "--eval-domains",
        type=str,
        default=None,
        help="Comma-separated domain subset to evaluate (overrides config "
             "'eval_domains'; e.g. --eval-domains ad or --eval-domains ad,video). "
             "Default is all three domains.",
    )
    parser.add_argument(
        "--no-clean-ckpts",
        dest="clean_ckpts",
        action="store_false",
        default=True,
        help="Disable automatic checkpoint cleanup. By default, after each "
             "checkpoint is successfully converted, auto_train deletes the raw "
             "global_step{N}/ and optimizer_ckpt/ dirs (converted/ is kept for "
             "eval); for unconverted steps, optimizer_ckpt/ is still deleted. "
             "Use --no-clean-ckpts to keep all raw checkpoints (e.g. for "
             "debugging conversion issues).",
    )
    return parser


def discover_converted_steps(job: TrainJob, *, log_tag: str = "[skip-train]") -> None:
    """Populate ``converted_steps`` + ``last_converted_step`` from existing
    converted dirs on disk.

    Used by ``--skip-train`` mode and by the pre-flight "skip this job" path
    (``log_tag="[skip-existing]"``) so the eval phase can still consume the
    checkpoints that are already on disk.  Discovery intentionally scans
    ``step{N}/converted`` directly because checkpoint cleanup deletes the raw
    ``global_step{N}/`` dirs after conversion succeeds.
    """
    job.converted_steps.clear()
    job.last_converted_step = None
    for step in sorted(scan_converted_step_dirs(job)):
        if step % job.save_checkpoint_per_step != 0:
            continue
        job.converted_steps.append(step)
        job.last_converted_step = step
        job.last_step = max(job.last_step, step)
    if job.last_converted_step is not None:
        log(
            f"{log_tag} {job.label}: found {len(job.converted_steps)} converted "
            f"checkpoint(s): {job.converted_steps} "
            f"(last=step{job.last_converted_step})"
        )
    else:
        log(
            f"{log_tag} {job.label}: no converted checkpoint found under "
            f"{job.output_dir} (will be skipped at eval time)."
        )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    log_path = init_log_file()
    log(f"Logging to {display_path(log_path)}")
    log(f"Command line: {' '.join(sys.argv)}")

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    validate_runtime_prerequisites()
    config_path = args.config.resolve()
    jobs, extras = load_jobs(config_path)
    data_version = extras["data_version"]

    # CLI override for eval_all_checkpoints (None = use config value)
    if args.eval_all_checkpoints is not None:
        for job in jobs:
            job.eval_all_checkpoints = args.eval_all_checkpoints
        log(f"CLI override: eval_all_checkpoints={args.eval_all_checkpoints} "
            f"applied to all {len(jobs)} job(s).")

    # CLI override for eval_steps (None = use config value)
    if args.eval_steps is not None:
        try:
            steps_list = _optional_int_list(
                [int(s.strip()) for s in args.eval_steps.split(",") if s.strip()],
                "args.eval_steps",
            )
        except (ConfigError, ValueError) as exc:
            log(f"Invalid --eval-steps value: {exc}")
            return 2
        for job in jobs:
            job.eval_steps = steps_list
        log(f"CLI override: eval_steps={steps_list} "
            f"applied to all {len(jobs)} job(s).")

    # CLI override for eval_domains (None = use config value)
    if args.eval_domains is not None:
        try:
            domains_list = _optional_domain_list(args.eval_domains, "args.eval_domains")
        except ConfigError as exc:
            log(f"Invalid --eval-domains value: {exc}")
            return 2
        for job in jobs:
            job.eval_domains = domains_list
        log(f"CLI override: eval_domains={domains_list} "
            f"applied to all {len(jobs)} job(s).")

    print_plan(
        jobs, poll_seconds=args.poll_seconds, data_version=data_version,
        clean_ckpts=args.clean_ckpts,
    )

    if args.dry_run:
        if args.skip_train:
            log("Dry run + --skip-train: scanning existing converted checkpoints for eval plan.")
            for job in jobs:
                discover_converted_steps(job, log_tag="[dry-run skip-train]")
        log("Dry run: would launch training for the above, then run eval phase:")
        run_eval_phase(jobs, data_version=data_version, dry_run=True)
        log("Dry run completed successfully.")
        return 0

    if not args.skip_train:
        handle_existing_output_dirs(jobs, assume_yes=args.yes_delete_existing)
        for job in jobs:
            if job.skip_training:
                log(f"Skipping training for {job.label} (existing checkpoints kept).")
                continue
            launch_training(job)
            last_step = monitor_training(
                job, poll_seconds=args.poll_seconds, clean_ckpts=args.clean_ckpts
            )
            record_finished(job, last_step)
            log(f"Finished {job.label} at step {last_step}")
        log("All training experiments completed.")
    else:
        log("--skip-train: skipping training phase; scanning existing checkpoints.")
        for job in jobs:
            discover_converted_steps(job)
            # ★ Reclaim disk from existing checkpoints (optimizer_ckpt/ on all
            #   steps, global_step{N}/ on already-converted steps).
            if args.clean_ckpts:
                cleanup_all_steps(job)

    if not args.skip_eval:
        run_eval_phase(jobs, data_version=data_version)
    else:
        log("--skip-eval: skipping evaluation phase.")

    log("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"Fatal error: {exc}")
        stack = traceback.format_exc().rstrip()
        if stack:
            log(stack)
        sys.exit(1)
    finally:
        close_log_file()
