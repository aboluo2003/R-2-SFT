#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a (ppl_eval): 多机多卡 PPL 评估。

读取 1_preprocess.py 产出的统一 schema parquet，对每个样本计算两个 PPL：
  - PPL(target | input)         -> ppl_y_given_x        (P(y|x) 的 PPL 形式)
  - PPL(target | input, cot)    -> ppl_y_given_x_cot    (P(y|x, cot) 的 PPL 形式)
并同时给出对应的 log 概率与打分 token 数，方便下游做 CoT 筛选。

设计要点：
  - 不使用 vllm：vllm 旧版本无法只返回目标 token 的 logits，返回全序列 logits 会 OOM。
  - 使用 transformers + flash_attention_2 (失败回退 sdpa) + logits_to_keep，
    只物化目标 token 对应的 logits，避免全序列 logits 炸显存。
  - 支持 caption (Qwen3-8B) 与 sid (OpenOneRec-8B) 两种 representation，
    通过 --representation_type 与 --model_path 切换，复用同一份主脚本。
  - sid 模式下 <|sid_begin|> 与 <|sid_end|> 各自是一个 token，从打分中剔除，
    只对 <s_a_*><s_b_*><s_c_*> 等 itemic token 计算 log 概率；caption 模式打分全部目标 token。
  - MPI 多机多卡：每个 rank 绑定一张 GPU（CUDA_VISIBLE_DEVICES 由 2_run_mpi.sh 设置），
    按 parquet 分片 round-robin 分配，每个 rank 独立写出自己的 parquet 分片。

输入 schema（来自 1_preprocess.py）：
  ppl_sample_key, sample_key, uid, target_domain, target_idx, target_pid,
  representation_type, system_text, input_text, input_with_cot_text,
  target_text, cot, history_values, history_pids, history_length,
  history_meta, render_meta

输出 schema（输入 schema + 下列新增列）：
  logp_y_given_x          : float   sum_j log P(target_j | input,          target_<j)
  logp_y_given_x_cot      : float   sum_j log P(target_j | input+cot,     target_<j)
  ppl_y_given_x           : float   exp(mean_j -log P(target_j | input))
  ppl_y_given_x_cot       : float   exp(mean_j -log P(target_j | input+cot))
  num_target_tokens       : int     实际参与打分的目标 token 数（两个变体共享同一 target，故一致）
  eval_meta                : str(JSON) prefix 长度 / 是否截断 / 打分 token 表等调试信息
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pa_fs
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "transformers is required. 请在运行环境安装 transformers (推荐 >= 4.51, "
        "集群默认 5.12.1)。"
    ) from exc

try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None


logger = logging.getLogger(__name__)

REPRESENTATION_CAPTION = "caption"
REPRESENTATION_SID = "sid"
SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"

# ---------------------------------------------------------------------------
# ETA / 耗时格式化
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """格式化耗时：<60s 显示 Xs，<1h 显示 Xm Ys，否则 Xh Ym。nan/inf 显示 N/A。"""
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "N/A"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60}s"
    return f"{total // 3600}h{(total % 3600) // 60}m"


def _format_eta(seconds: float) -> str:
    """格式化剩余时间：>=1 天显示 Xd Yh，否则沿用 _format_duration。"""
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "N/A"
    total = int(seconds)
    if total >= 86400:
        return f"{total // 86400}d{(total % 86400) // 3600}h"
    return _format_duration(total)


def _format_clock(eta_seconds: float) -> str:
    """把剩余秒数转成"预计 HH:MM:SS 完成"的时钟点。"""
    if eta_seconds is None or math.isnan(eta_seconds) or math.isinf(eta_seconds) or eta_seconds < 0:
        return "N/A"
    return (datetime.now() + timedelta(seconds=int(eta_seconds))).strftime("%H:%M:%S")


def _format_throughput(tokens_per_sec: float) -> str:
    """token 吞吐量：K/s 或 M/s。"""
    if tokens_per_sec is None or math.isnan(tokens_per_sec) or math.isinf(tokens_per_sec) or tokens_per_sec < 0:
        return "N/A"
    if tokens_per_sec >= 1e6:
        return f"{tokens_per_sec / 1e6:.2f}M tok/s"
    if tokens_per_sec >= 1e3:
        return f"{tokens_per_sec / 1e3:.1f}K tok/s"
    return f"{tokens_per_sec:.1f} tok/s"

# ★ 与 5_data_format/main.py build_cot_answer 对齐：CoT 放在 assistant 的  Sofia... Sofia 块里
#    训练侧 qwen3_dataset.py add_think_pattern=True 识别这两个 tag，有非空 think body 时给 user 加 /think。
#    PPL eval 的 xcot 变体复用该格式：assistant 前缀 =  Sofia\ncot\nSofia\n\n，target 紧随其后。
THINK_OPEN = "<" + "think" + ">"    #  Sofia
THINK_CLOSE = "<" + "/" + "think" + ">"  #  Sofia

ADDED_COLUMNS = (
    "logp_y_given_x",
    "logp_y_given_x_cot",
    "ppl_y_given_x",
    "ppl_y_given_x_cot",
    "num_target_tokens",
    "eval_meta",
)


# ---------------------------------------------------------------------------
# 日志 / 分布式
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool, rank: int) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s %(levelname)s [rank{rank:05d}] %(message)s",
        force=True,
    )


def init_distributed() -> Tuple[int, int, int, Optional[Any], str]:
    """初始化分布式环境。

    - mpirun 启动（OMPI_COMM_WORLD_RANK 在环境变量中）：必须使用 mpi4py，**不 fallback**。
      若 mpi4py 不可用，立即报错——否则每个进程都会误认为自己是 rank 0 跑全集，
      N 卡变成 N 份重复劳动且输出互相覆盖（这正是 2026-08-16 排查到的 bug）。
    - torchrun 启动（torch.distributed 已初始化或 RANK/WORLD_SIZE 在环境变量中）：用 torch。
    - 直接 python3 启动（无任何分布式环境变量）：单进程模式，用于本地调试。
    """
    under_mpirun = "OMPI_COMM_WORLD_RANK" in os.environ
    if under_mpirun:
        if MPI is None:
            raise RuntimeError(
                "检测到 mpirun 环境（OMPI_COMM_WORLD_RANK 已设置）但 mpi4py 未安装。"
                "多进程下若 fallback 到单进程，每个进程都会当自己是 rank 0 跑全集，"
                "浪费算力且输出互相覆盖。请安装 mpi4py：\n"
                "  pip install mpi4py\n"
                "并确保 mpi4py 链接的 MPI 库与 mpirun 使用的 OpenMPI 版本一致。"
            )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        return rank, world_size, local_rank, comm, "mpi"

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank, None, "torch"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1 and torch.distributed.is_available():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
        return rank, world_size, local_rank, None, "torch"
    return rank, world_size, local_rank, None, "single"


def resolve_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        # - MPI (2_run_mpi.sh): CUDA_VISIBLE_DEVICES 已限制为单卡，cuda:0 即绑定卡。
        # - torchrun: 一个进程可见多卡，需要按 local_rank 选卡。
        visible = torch.cuda.device_count()
        idx = 0 if visible <= 1 else min(int(local_rank), visible - 1)
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 文件 IO (本地 + HDFS/ViewFS)
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
    # 目录：递归列出
    selector = pa_fs.FileSelector(info.path, recursive=True)
    infos = filesystem.get_file_info(selector)
    files = [i.path for i in infos if i.is_file and _is_parquet_data_file(i.path)]
    files.sort()
    if not files:
        raise FileNotFoundError(f"未在 {uri} 找到 parquet 文件")
    return filesystem, files


def read_parquet(filesystem: Any, file_path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    with filesystem.open_input_file(file_path) as f:
        table = pq.read_table(f, columns=columns)
    return table.to_pandas()


def write_parquet_auto(df: pd.DataFrame, output_file_path: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    parsed = urlparse(output_file_path)
    if parsed.scheme and parsed.scheme != "file":
        filesystem, fs_path = get_filesystem_and_path(output_file_path)
        parent = os.path.dirname(fs_path.rstrip("/"))
        try:
            filesystem.create_dir(parent, recursive=True)
        except Exception as exc:  # 目录可能已存在，忽略
            logger.debug("create_dir(%s) 忽略: %s", parent, exc)
        pq.write_table(table, fs_path, filesystem=filesystem)
    else:
        local_path = parsed.path if parsed.scheme == "file" else output_file_path
        local_path = os.path.abspath(local_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        pq.write_table(table, local_path)


def chunk_for_rank(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    return [item for idx, item in enumerate(items) if idx % world_size == rank]


# ---------------------------------------------------------------------------
# 文本归一化 (对齐 test_sid_ppl.py)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n")
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    if "\\t" in text:
        text = text.replace("\\t", "\t")
    return text.strip()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"不支持的 dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any]:
    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    base_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": bool(args.local_files_only),
    }
    if dtype != "auto":
        base_kwargs["torch_dtype"] = dtype
    if device.type == "cuda":
        base_kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    else:
        base_kwargs["device_map"] = "cpu"

    attn_candidates: List[str] = []
    if args.attn_implementation and args.attn_implementation != "auto":
        attn_candidates.append(args.attn_implementation)
    attn_candidates.extend([x for x in ("flash_attention_2", "sdpa", "eager") if x not in attn_candidates])

    last_error: Optional[Exception] = None
    for impl in attn_candidates:
        kwargs = dict(base_kwargs)
        if impl != "auto":
            kwargs["attn_implementation"] = impl
        try:
            logger.info("加载模型: model_path=%s dtype=%s attn=%s", args.model_path, args.dtype, impl)
            model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
            model.eval()
            # ★ 关键修复：无论是否使用 device_map，都显式 model.to(device)。
            # 原因：device_map={"": 0} 在某些 transformers/accelerate 版本下不会移动
            # 非持久化 buffer（如 Qwen3RotaryEmbedding 的 inv_freq，注册时 persistent=False），
            # 导致 forward 时出现 "Expected all tensors to be on the same device,
            # but found at least two devices, cpu and cuda:0! (when checking argument
            # for argument index in method wrapper_CUDA__index_select)"。
            # 显式 model.to(device) 能把所有参数和 buffer（包括非持久化的）都搬到目标设备。
            model.to(device)
            _log_model_device(model, device)
            logger.info("模型加载成功: attn_implementation=%s", impl)
            return tokenizer, model
        except Exception as exc:  # pragma: no cover
            last_error = exc
            logger.warning("attn_implementation=%s 加载失败: %s；尝试下一个", impl, exc)
    raise RuntimeError(f"模型加载失败，已尝试 {attn_candidates}") from last_error


def _log_model_device(model: Any, expected_device: torch.device) -> None:
    """诊断日志：打印模型参数 / buffer 的实际设备分布，便于排查 device 不一致问题。"""
    try:
        param_devices: Dict[str, int] = {}
        buffer_devices: Dict[str, int] = {}
        for name, param in model.named_parameters():
            param_devices.setdefault(str(param.device), 0)
            param_devices[str(param.device)] += 1
        for name, buf in model.named_buffers():
            buffer_devices.setdefault(str(buf.device), 0)
            buffer_devices[str(buf.device)] += 1
        # 取第一个参数的设备作为代表
        first_param = next(iter(model.parameters()), None)
        first_buf = next(iter(model.named_buffers()), None)
        logger.info(
            "模型设备分布: params=%s buffers=%s expected=%s first_param=%s first_buf=%s",
            param_devices, buffer_devices, expected_device,
            str(first_param.device) if first_param is not None else "N/A",
            (str(first_buf[1].device) if first_buf else "N/A"),
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("模型设备诊断失败（忽略）: %s", exc)


# ---------------------------------------------------------------------------
# SID token 处理
# ---------------------------------------------------------------------------

def get_sid_wrapper_ids(tokenizer: Any) -> Tuple[Optional[int], Optional[int]]:
    """返回 (<|sid_begin|> id, <|sid_end|> id)，并断言各自是单 token。"""
    begin_ids = tokenizer.encode(SID_BEGIN, add_special_tokens=False)
    end_ids = tokenizer.encode(SID_END, add_special_tokens=False)
    if len(begin_ids) == 1 and len(end_ids) == 1:
        return begin_ids[0], end_ids[0]
    # 兼容某些 tokenizer 把 <|sid_end|> 拆成多 token 的情况：返回 None 表示不做剔除
    if len(begin_ids) == 1 and len(end_ids) != 1:
        logger.warning(
            "<|sid_end|> 被切成 %d 个 token：%s；将不再剔除 <|sid_end|>",
            len(end_ids), end_ids,
        )
        return begin_ids[0], None
    logger.warning(
        "<|sid_begin|> 被切成 %d 个 token：%s；sid 剔除逻辑将被关闭",
        len(begin_ids), begin_ids,
    )
    return None, None


# ---------------------------------------------------------------------------
# chat template / 序列构建
# ---------------------------------------------------------------------------

def apply_chat_template_safe(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    signature = inspect.signature(tokenizer.apply_chat_template)
    if "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_prefix_ids(
    tokenizer: Any,
    system_text: str,
    input_text: str,
    use_system_message: bool,
    enable_thinking: bool,
    cot_text: Optional[str] = None,
) -> List[int]:
    """构造 prefix token ids（target 紧随其后）。

    ★ 与训练侧 qwen3_dataset.py add_think_pattern 对齐（显式构造，不依赖模板 enable_thinking）：
    - cot_text 为 None（x 变体）：user content 末尾加 /no_think，assistant 前缀
      = 空 think 块（ Sofia\n\nSofia\n\n），target 紧随其后。
      与训练侧非 CoT 样本对齐（add_think_pattern 检测无 think body → /no_think + 空 think 块）。
    - cot_text 非空（xcot 变体）：user content 末尾加 /think，assistant 前缀
      =  Sofia\ncot_body\nSofia\n\n，target 紧随其后。
      与训练侧 build_cot_answer( Sofia + cot + Sofia + target) 对齐，测 PPL(target | input, cot)。

    ★ 显式构造的原因：训练侧 add_think_pattern 在 _convert_messages 里手动加 /think /no_think
      + 包空/实 think 块，然后 apply_chat_template(add_generation_prompt=False) 仅做角色标签包装。
      这里复刻该逻辑：用 add_generation_prompt=False 渲染完整对话（含 assistant think 块），
      再把 target 拼到 think 块之后；避免依赖模型模板的 enable_thinking 参数（不同模板行为不一致）。
    """
    cot_body = safe_str(cot_text) if cot_text else ""
    # 与训练侧 add_think_pattern 一致：有非空 think body → /think，否则 /no_think
    think_suffix = "/think" if cot_body else "/no_think"
    if cot_body and not cot_body.endswith("\n"):
        cot_body += "\n"
    # ★ 与训练侧 build_cot_answer 一致的 assistant 内容：
    #   x 变体（无 cot）:  Sofia\n\nSofia\n\n  (空 think 块，add_think_pattern 检测无 body 时填)
    #   xcot 变体（有 cot）:  Sofia\ncot\nSofia\n\n
    assistant_content = THINK_OPEN + "\n" + cot_body + THINK_CLOSE + "\n\n"

    messages: List[Dict[str, str]] = []
    if use_system_message and system_text:
        messages.append({"role": "system", "content": system_text})
    # user content 末尾显式加 /think 或 /no_think（不靠模板）
    messages.append({"role": "user", "content": input_text + think_suffix})
    # assistant 内容 = think 块（空或实），target 在调用方拼接（不放进 assistant_content，避免被模板包 <|im_end|>）
    messages.append({"role": "assistant", "content": assistant_content})

    # ★ add_generation_prompt=False：训练侧用 False 渲染完整对话（含 assistant），
    #   这里也用 False，渲染出 ...<|im_start|>assistant\n{think_block}<|im_end|>\n
    #   然后 strip 掉结尾的 <|im_end|>\n，让 target 紧接 think 块之后。
    prefix_text = apply_chat_template_full(tokenizer, messages)
    # 去掉模板追加的 <|im_end|>\n（assistant 段结尾），让 target 紧接 think 块
    im_end_nl = "<|im_end|>\n"
    if prefix_text.endswith(im_end_nl):
        prefix_text = prefix_text[: -len(im_end_nl)]
    # ★ add_special_tokens=True 与训练侧 self.tokenizer(text) 对齐（Qwen3 add_bos_token=False 时为 no-op；
    #   若某模型 add_bos_token=True 则训练也会加 BOS，这里同样加，保持一致）
    return tokenizer.encode(prefix_text, add_special_tokens=True)


def apply_chat_template_full(
    tokenizer: Any,
    messages: List[Dict[str, str]],
) -> str:
    """用 add_generation_prompt=False 渲染完整对话（与训练侧 _process_chat 一致）。"""
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_score_mask(
    target_ids: List[int],
    representation_type: str,
    sid_begin_id: Optional[int],
    sid_end_id: Optional[int],
) -> List[bool]:
    if representation_type != REPRESENTATION_SID:
        return [True] * len(target_ids)
    exclude = {tid for tid in (sid_begin_id, sid_end_id) if tid is not None}
    if not exclude:
        return [True] * len(target_ids)
    return [tid not in exclude for tid in target_ids]


def prepare_sequences(
    frame: pd.DataFrame,
    tokenizer: Any,
    args: argparse.Namespace,
    sid_begin_id: Optional[int],
    sid_end_id: Optional[int],
    rank: int = 0,
) -> List[Dict[str, Any]]:
    """把一个 parquet 文件里的所有样本展开成 (sample_idx, variant) 序列列表。

    ★ 0724 ver3: cot 放在 assistant 的  Sofia... Sofia 块里（不再塞 user message）。
      - x 变体:  user(input_text) + assistant 空 think 块  → PPL(target | input)
      - xcot 变体: user(input_text) + assistant  Sofia\\ncot\\nSofia\\n\\n  → PPL(target | input, cot)
      两个变体的 user message 相同（都是 input_text），差异只在 assistant 前缀。
    """
    sequences: List[Dict[str, Any]] = []
    skipped = 0
    truncated = 0
    first_sample_logged = False
    for sample_idx, row in enumerate(frame.to_dict(orient="records")):
        system_text = normalize_text(safe_str(row.get("system_text")))
        input_text = normalize_text(safe_str(row.get("input_text")))
        target_text = normalize_text(safe_str(row.get("target_text")))
        cot_text = safe_str(row.get("cot"))
        representation_type = safe_str(row.get("representation_type")) or args.representation_type

        if not target_text:
            skipped += 1
            continue

        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            skipped += 1
            continue

        score_mask = build_score_mask(target_ids, representation_type, sid_begin_id, sid_end_id)
        if not any(score_mask):
            skipped += 1
            continue

        # ★ rank 0 打印第一个样本的完整输入内容供 check（两个变体的 prefix_text + target_text）
        if rank == 0 and not first_sample_logged:
            first_sample_logged = True
            _log_first_sample(
                tokenizer=tokenizer,
                system_text=system_text,
                input_text=input_text,
                cot_text=cot_text,
                target_text=target_text,
                use_system_message=bool(args.use_system_message),
                enable_thinking=bool(args.enable_thinking),
            )

        for variant in ("x", "xcot"):
            # 两个变体都用 input_text 作为 user message；xcot 把 cot 放 assistant think 块
            variant_cot = cot_text if (variant == "xcot" and cot_text) else None
            if not input_text:
                sequences.append({
                    "sample_idx": sample_idx,
                    "variant": variant,
                    "prefix_ids": None,
                    "target_ids": target_ids,
                    "score_mask": score_mask,
                    "skipped": True,
                    "truncated": False,
                })
                continue
            prefix_ids = build_prefix_ids(
                tokenizer=tokenizer,
                system_text=system_text,
                input_text=input_text,
                use_system_message=bool(args.use_system_message),
                enable_thinking=bool(args.enable_thinking),
                cot_text=variant_cot,
            )
            truncated_flag = False
            total_len = len(prefix_ids) + len(target_ids)
            if total_len > args.max_seq_len:
                # 截断 prefix 左侧，保留紧邻 target 的上下文 + 完整 target
                keep_prefix = max(0, args.max_seq_len - len(target_ids))
                if keep_prefix < len(prefix_ids):
                    prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []
                    truncated_flag = True
                    truncated += 1
            sequences.append({
                "sample_idx": sample_idx,
                "variant": variant,
                "prefix_ids": prefix_ids,
                "target_ids": target_ids,
                "score_mask": score_mask,
                "skipped": False,
                "truncated": truncated_flag,
            })
    if skipped:
        logger.warning("跳过 %d 条空 target / 无打分 token 的样本", skipped)
    if truncated:
        logger.warning("截断 %d 条超长 prefix (max_seq_len=%d)", truncated, args.max_seq_len)
    return sequences


def _log_first_sample(
    tokenizer: Any,
    system_text: str,
    input_text: str,
    cot_text: str,
    target_text: str,
    use_system_message: bool,
    enable_thinking: bool,
) -> None:
    """rank 0 打印第一个样本的完整 prefix_text + target_text，供人工 check 格式。"""
    sep = "=" * 80
    print(f"\n{sep}", flush=True)
    print("[rank0 first sample] PPL eval 输入内容 check（cot 在 assistant think 块）", flush=True)
    print(sep, flush=True)
    print(f"[system_text]\n{system_text}", flush=True)
    print(f"\n[input_text (user message)]\n{input_text}", flush=True)
    print(f"\n[cot (raw)]\n{cot_text}", flush=True)
    print(f"\n[target_text]\n{target_text}", flush=True)

    # x 变体 prefix（无 cot，enable_thinking=False → /no_think + 空 think 块）
    x_prefix = build_prefix_ids(
        tokenizer=tokenizer, system_text=system_text, input_text=input_text,
        use_system_message=use_system_message, enable_thinking=enable_thinking, cot_text=None,
    )
    x_prefix_text = tokenizer.decode(x_prefix, skip_special_tokens=False)
    print(f"\n[x variant prefix_text] (enable_thinking=False, 空 think 块):\n{x_prefix_text}", flush=True)

    # xcot 变体 prefix（cot 在 assistant think 块, enable_thinking=True → /think）
    xcot_prefix = build_prefix_ids(
        tokenizer=tokenizer, system_text=system_text, input_text=input_text,
        use_system_message=use_system_message, enable_thinking=enable_thinking, cot_text=cot_text,
    )
    xcot_prefix_text = tokenizer.decode(xcot_prefix, skip_special_tokens=False)
    print(f"\n[xcot variant prefix_text] (enable_thinking=True,  Sofia cot Sofia 块):\n{xcot_prefix_text}", flush=True)
    print(f"\n[target_text appended after prefix]\n{target_text}", flush=True)
    print(sep, flush=True)


# ---------------------------------------------------------------------------
# 批量 PPL 计算 (左 padding + logits_to_keep)
# ---------------------------------------------------------------------------

def _detect_logits_to_keep_kwarg(model: Any) -> str:
    """检测 forward 支持的 logits 限制参数名。"""
    forward_signature = inspect.signature(model.forward)
    if "logits_to_keep" in forward_signature.parameters:
        return "logits_to_keep"
    if "num_logits_to_keep" in forward_signature.parameters:
        return "num_logits_to_keep"
    return ""


def compute_batch_ppl(
    model: Any,
    batch: List[Dict[str, Any]],
    pad_id: int,
    device: torch.device,
    logits_to_keep_kwarg: str,
) -> List[Dict[str, Any]]:
    """对一个 micro-batch 做左 padding 前向，逐序列计算 logp / ppl。返回结果与 batch 同序。"""
    n = len(batch)
    results: List[Optional[Dict[str, Any]]] = [None] * n
    valid_indices: List[int] = []
    for i, s in enumerate(batch):
        if s.get("skipped") or s.get("prefix_ids") is None:
            results[i] = {
                "logp": float("nan"),
                "ppl": float("nan"),
                "num_tokens": 0,
                "truncated": s.get("truncated", False),
            }
        else:
            valid_indices.append(i)
    if not valid_indices:
        return results  # type: ignore[return-value]

    valid = [batch[i] for i in valid_indices]
    total_lens = [len(s["prefix_ids"]) + len(s["target_ids"]) for s in valid]
    target_lens = [len(s["target_ids"]) for s in valid]
    max_total = max(total_lens)
    max_target = max(target_lens)
    n_valid = len(valid)

    input_ids = torch.full((n_valid, max_total), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n_valid, max_total), dtype=torch.long)
    for j, s in enumerate(valid):
        ids = list(s["prefix_ids"]) + list(s["target_ids"])
        offset = max_total - len(ids)
        input_ids[j, offset:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[j, offset:] = 1

    # 左 padding 下 position_ids 必须按 attention_mask 重新累计，否则 RoPE 位置错乱
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0).long()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)

    # ★ 防御性断言：确保输入与模型在同一设备，避免 "Expected all tensors to be on
    # the same device, but found at least two devices, cpu and cuda:0!" 这类报错
    # 在 forward 内部才暴露（错误信息不直观）。
    try:
        model_param_device = next(model.parameters()).device
    except StopIteration:
        model_param_device = device
    if model_param_device != device:
        logger.warning(
            "设备不一致: model 在 %s, 输入在 %s；尝试 model.to(device) 兜底",
            model_param_device, device,
        )
        model.to(device)

    # logits_to_keep = max_target + 1：多保留一格用于预测第一个 target token
    logits_to_keep = min(max_target + 1, max_total)

    with torch.no_grad():
        forward_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
        }
        if logits_to_keep_kwarg:
            forward_kwargs[logits_to_keep_kwarg] = logits_to_keep
        output = model(**forward_kwargs)
        logits = output.logits  # [n_valid, K, vocab]

    for j, s in enumerate(valid):
        i = valid_indices[j]
        target_ids = s["target_ids"]
        score_mask = s["score_mask"]
        t_i = len(target_ids)
        # 返回 logits 的相对索引：r_j = max_target - t_i + j
        start = max_target - t_i
        end = max_target
        seq_logits = logits[j, start:end, :].float()  # [t_i, vocab]
        seq_log_probs = F.log_softmax(seq_logits, dim=-1)  # [t_i, vocab]
        target_ids_t = torch.tensor(target_ids, device=seq_log_probs.device, dtype=torch.long)
        token_lp = seq_log_probs.gather(1, target_ids_t.unsqueeze(1)).squeeze(1)  # [t_i]
        mask_t = torch.tensor(score_mask, device=token_lp.device, dtype=torch.bool)
        scored = token_lp[mask_t]
        if scored.numel() == 0:
            logp = float("nan")
            ppl = float("nan")
            num_tokens = 0
        else:
            logp = float(scored.sum().item())
            nll = float((-scored.mean()).item())
            ppl = float(math.exp(nll)) if nll < 50 else float("inf")
            num_tokens = int(scored.numel())
        results[i] = {
            "logp": logp,
            "ppl": ppl,
            "num_tokens": num_tokens,
            "truncated": s.get("truncated", False),
        }

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 单文件处理
# ---------------------------------------------------------------------------

def process_file(
    file_path: str,
    filesystem: Any,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    pad_id: int,
    device: torch.device,
    logits_to_keep_kwarg: str,
    sid_begin_id: Optional[int],
    sid_end_id: Optional[int],
    rank: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = read_parquet(filesystem, file_path)
    stats = {
        "rows_seen": len(df),
        "rows_scored": 0,
        "rows_partial_nan": 0,
        "rows_full_nan": 0,
        "micro_batches": 0,
        "tokens_processed": 0,
    }
    if df.empty:
        return _attach_ppl_columns(df, {}, {}), stats

    sequences = prepare_sequences(df, tokenizer, args, sid_begin_id, sid_end_id, rank=rank)

    # 按总长度排序后切 micro-batch，尽量减少 padding
    valid_seqs = [s for s in sequences if not s.get("skipped") and s["prefix_ids"] is not None]
    valid_seqs.sort(key=lambda s: len(s["prefix_ids"]) + len(s["target_ids"]), reverse=True)

    seq_results: Dict[int, Dict[str, Any]] = {}
    # ★ 动态 batch：valid_seqs 已按总长度降序排序（上方 sort），按 token 预算切块。
    #   长 seq → 自动小 batch（防 OOM），短 seq → 自动大 batch（保吞吐）。
    #   token_budget 默认 131072 (32×4096)；max_tokens_per_batch<=0 时退化为固定 batch_size。
    #   例（Qwen3-8B bf16，intermediate_size=12288）：seq=15600 → bs≈8（MLP intermediate ~2.9G，安全）；
    #                                                  seq=4096  → bs=32（满 batch_size 上限）。
    token_budget = args.max_tokens_per_batch if args.max_tokens_per_batch > 0 else (1 << 62)
    max_bs = max(1, args.batch_size)
    bs_hist: Dict[int, int] = {}
    # 文件内 ETA：预先算出文件总 token 数（用于进度百分比 + 文件内剩余时间）
    file_total_tokens = sum(len(s["prefix_ids"]) + len(s["target_ids"]) for s in valid_seqs)
    file_t0 = time.time()
    last_log_t = file_t0
    processed_tokens = 0
    start = 0
    while start < len(valid_seqs):
        # 本批最长序列 = 当前窗口第一条（已降序），以此为基准算 token 预算
        first = valid_seqs[start]
        max_len = len(first["prefix_ids"]) + len(first["target_ids"])
        bs_by_tokens = max(1, token_budget // max(max_len, 1))
        bs = min(max_bs, bs_by_tokens, len(valid_seqs) - start)
        batch = valid_seqs[start:start + bs]
        batch_results = compute_batch_ppl(model, batch, pad_id, device, logits_to_keep_kwarg)
        for s, r in zip(batch, batch_results):
            seq_results[id(s)] = r
        # 累计 token（用实际序列长度和，比 bs×max_len 更准，反映真实计算量）
        batch_tokens = sum(len(s["prefix_ids"]) + len(s["target_ids"]) for s in batch)
        processed_tokens += batch_tokens
        stats["tokens_processed"] += batch_tokens
        stats["micro_batches"] += 1
        bs_hist[bs] = bs_hist.get(bs, 0) + 1
        # 批内进度日志：每 50 个 micro-batch 或最后一个打一次（间隔>=30s 避免刷屏）
        is_last = start + bs >= len(valid_seqs)
        now = time.time()
        if is_last or (stats["micro_batches"] % 50 == 0 and now - last_log_t >= 30):
            file_elapsed = now - file_t0
            pct = (100.0 * processed_tokens / file_total_tokens) if file_total_tokens > 0 else 0.0
            tok_s = processed_tokens / file_elapsed if file_elapsed > 0 else 0.0
            if is_last:
                logger.info(
                    "rank%05d 文件完成: %s batches=%d tokens=%d (%.1f%%) 耗时=%s %s",
                    rank, os.path.basename(file_path), stats["micro_batches"],
                    processed_tokens, pct, _format_duration(file_elapsed), _format_throughput(tok_s),
                )
            else:
                file_eta = (file_elapsed / processed_tokens * (file_total_tokens - processed_tokens)) if processed_tokens > 0 else 0.0
                logger.info(
                    "rank%05d 文件内进度: %s batches=%d tokens=%d/%d (%.1f%%) 耗时=%s 剩余≈%s %s",
                    rank, os.path.basename(file_path), stats["micro_batches"],
                    processed_tokens, file_total_tokens, pct,
                    _format_duration(file_elapsed), _format_eta(file_eta), _format_throughput(tok_s),
                )
            last_log_t = now
        start += bs
    if bs_hist:
        logger.log(
            logging.INFO if rank == 0 else logging.DEBUG,
            "动态 batch 分布: %s (token_budget=%s, max_bs=%d, n_valid=%d)",
            dict(sorted(bs_hist.items())),
            args.max_tokens_per_batch if args.max_tokens_per_batch > 0 else "disabled",
            max_bs, len(valid_seqs),
        )

    # 组装每个 (sample_idx, variant) 的结果
    per_sample: Dict[int, Dict[str, Any]] = {}
    for s in sequences:
        sample_idx = s["sample_idx"]
        variant = s["variant"]
        if s.get("skipped") or s["prefix_ids"] is None:
            res = {"logp": float("nan"), "ppl": float("nan"),
                   "num_tokens": 0, "truncated": s.get("truncated", False)}
        else:
            res = seq_results.get(id(s), {"logp": float("nan"), "ppl": float("nan"),
                                          "num_tokens": 0, "truncated": False})
        per_sample.setdefault(sample_idx, {})[variant] = res
        per_sample[sample_idx].setdefault("truncated", False)
        per_sample[sample_idx]["truncated"] = per_sample[sample_idx]["truncated"] or res.get("truncated", False)

    return _attach_ppl_columns(df, per_sample, {}), _finalize_stats(stats, per_sample)


def _attach_ppl_columns(
    df: pd.DataFrame,
    per_sample: Dict[int, Dict[str, Any]],
    _unused: Dict[str, Any],
) -> pd.DataFrame:
    n = len(df)
    logp_x = [float("nan")] * n
    logp_xcot = [float("nan")] * n
    ppl_x = [float("nan")] * n
    ppl_xcot = [float("nan")] * n
    num_tokens = [0] * n
    eval_meta = [""] * n
    for sample_idx, info in per_sample.items():
        if sample_idx >= n:
            continue
        x = info.get("x", {})
        xcot = info.get("xcot", {})
        logp_x[sample_idx] = x.get("logp", float("nan"))
        logp_xcot[sample_idx] = xcot.get("logp", float("nan"))
        ppl_x[sample_idx] = x.get("ppl", float("nan"))
        ppl_xcot[sample_idx] = xcot.get("ppl", float("nan"))
        # 两个变体共享同一 target，num_tokens 取有效的一方
        nt = x.get("num_tokens", 0) or xcot.get("num_tokens", 0) or 0
        num_tokens[sample_idx] = int(nt)
        eval_meta[sample_idx] = json.dumps({
            "truncated": info.get("truncated", False),
            "num_tokens": nt,
        }, ensure_ascii=False)
    df = df.copy()
    df["logp_y_given_x"] = logp_x
    df["logp_y_given_x_cot"] = logp_xcot
    df["ppl_y_given_x"] = ppl_x
    df["ppl_y_given_x_cot"] = ppl_xcot
    df["num_target_tokens"] = num_tokens
    df["eval_meta"] = eval_meta
    return df


def _finalize_stats(stats: Dict[str, int], per_sample: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    full = 0
    partial = 0
    scored = 0
    for info in per_sample.values():
        x_ok = not math.isnan(info.get("x", {}).get("ppl", float("nan")))
        xcot_ok = not math.isnan(info.get("xcot", {}).get("ppl", float("nan")))
        if x_ok and xcot_ok:
            full += 1
            scored += 1
        elif x_ok or xcot_ok:
            partial += 1
            scored += 1
    stats["rows_scored"] = scored
    stats["rows_full_nan"] = len(per_sample) - scored
    stats["rows_partial_nan"] = partial
    return stats


# ---------------------------------------------------------------------------
# 输出文件命名
# ---------------------------------------------------------------------------

def make_output_path(input_file_path: str, output_uri: str, rank: int, file_idx: int) -> str:
    base = os.path.basename(input_file_path)
    stem = base.rsplit(".", 1)[0] if base.endswith(".parquet") else base
    out_name = f"part-rank{rank:05d}-f{file_idx:05d}-{stem}.parquet"
    parsed = urlparse(output_uri)
    if parsed.scheme and parsed.scheme != "file":
        return f"{output_uri.rstrip('/')}/{out_name}"
    local = parsed.path if parsed.scheme == "file" else output_uri
    return os.path.join(os.path.abspath(local), out_name)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    rank, world_size, local_rank, comm, distributed_mode = init_distributed()
    setup_logging(args.verbose, rank)

    device = resolve_device(local_rank)
    logger.info(
        "启动: rank=%s world_size=%s local_rank=%s mode=%s device=%s",
        rank, world_size, local_rank, distributed_mode, device,
    )
    logger.info(
        "input_uri=%s output_uri=%s model_path=%s representation_type=%s",
        args.input_uri, args.output_uri, args.model_path, args.representation_type,
    )

    if args.representation_type not in (REPRESENTATION_CAPTION, REPRESENTATION_SID):
        raise ValueError(f"不支持的 representation_type: {args.representation_type}")

    filesystem, parquet_files = list_parquet_files(args.input_uri)
    if args.max_files > 0:
        parquet_files = parquet_files[: args.max_files]
    my_files = chunk_for_rank(parquet_files, rank, world_size)
    logger.info("文件分配: total=%s my_files=%s", len(parquet_files), len(my_files))

    tokenizer, model = load_model_and_tokenizer(args, device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    logger.info("pad_id=%s", pad_id)

    sid_begin_id, sid_end_id = (None, None)
    if args.representation_type == REPRESENTATION_SID:
        sid_begin_id, sid_end_id = get_sid_wrapper_ids(tokenizer)
        logger.info("sid_begin_id=%s sid_end_id=%s", sid_begin_id, sid_end_id)

    logits_to_keep_kwarg = _detect_logits_to_keep_kwarg(model)
    if not logits_to_keep_kwarg:
        logger.warning(
            "model.forward 不支持 logits_to_keep / num_logits_to_keep，"
            "将退化为全序列 logits（长序列可能 OOM）。"
        )

    total_stats = {
        "files_seen": 0,
        "files_ok": 0,
        "rows_seen": 0,
        "rows_scored": 0,
        "rows_full_nan": 0,
        "rows_partial_nan": 0,
        "micro_batches": 0,
        "tokens_processed": 0,
    }
    outputs_written: List[str] = []

    # ★ 文件级 ETA 跟踪：记录每个文件耗时，用滑动平均外推剩余时间
    rank_t0 = time.time()
    file_times: List[float] = []
    eta_log_every = max(1, args.eta_log_interval)

    for file_idx, file_path in enumerate(tqdm(my_files, desc=f"rank{rank:05d}", disable=rank != 0 and not args.verbose)):
        if args.limit > 0 and file_idx >= args.limit:
            logger.info("命中 --limit=%s，停止本 rank 处理", args.limit)
            break
        total_stats["files_seen"] += 1
        file_t0 = time.time()
        try:
            out_df, stats = process_file(
                file_path=file_path,
                filesystem=filesystem,
                args=args,
                model=model,
                tokenizer=tokenizer,
                pad_id=pad_id,
                device=device,
                logits_to_keep_kwarg=logits_to_keep_kwarg,
                sid_begin_id=sid_begin_id,
                sid_end_id=sid_end_id,
                rank=rank,
            )
        except Exception as exc:
            logger.warning("处理文件失败，跳过: %s error=%s", file_path, exc)
            if args.verbose:
                logger.exception("详细堆栈:")
            continue

        total_stats["files_ok"] += 1
        for k in ("rows_seen", "rows_scored", "rows_full_nan", "rows_partial_nan", "micro_batches", "tokens_processed"):
            total_stats[k] += stats.get(k, 0)

        out_path = make_output_path(file_path, args.output_uri, rank, file_idx)
        if not args.dry_run:
            write_parquet_auto(out_df, out_path)
            outputs_written.append(out_path)
            logger.info("写出 %s rows=%s", out_path, len(out_df))

        # 记录本文件耗时（含写出），用于 ETA 滑动平均
        file_dt = time.time() - file_t0
        file_times.append(file_dt)
        done = file_idx + 1
        total_files = len(my_files)
        is_last = done == total_files
        # 日志频率：eta_log_interval<=0 → 仅最后文件打；>0 → 首文件 + 每 N 个 + 最后一个都打
        log_now = is_last
        if args.eta_log_interval > 0:
            log_now = log_now or done == 1 or done % eta_log_every == 0
        if log_now:
            elapsed = time.time() - rank_t0
            remaining_files = total_files - done
            # 滑动平均：最近 min(10, done) 个文件，反映当前速度（比全程均值更准）
            recent = file_times[-min(10, done):]
            avg_per_file = sum(recent) / len(recent) if recent else 0.0
            eta_seconds = avg_per_file * remaining_files if avg_per_file > 0 else 0.0
            tok_total = total_stats["tokens_processed"]
            tok_s = tok_total / elapsed if elapsed > 0 else 0.0
            pct = (100.0 * done / total_files) if total_files > 0 else 0.0
            if is_last:
                logger.info(
                    "rank%05d 全部文件完成: %s/%s rows=%s tokens=%s 耗时=%s %s",
                    rank, done, total_files, total_stats["rows_scored"], tok_total,
                    _format_duration(elapsed), _format_throughput(tok_s),
                )
            else:
                logger.info(
                    "rank%05d 进度 %s/%s (%.1f%%) rows=%s tokens=%s "
                    "耗时=%s 均/文件=%s 剩余≈%s ETA=%s %s",
                    rank, done, total_files, pct, total_stats["rows_scored"], tok_total,
                    _format_duration(elapsed), _format_duration(avg_per_file),
                    _format_eta(eta_seconds), _format_clock(eta_seconds),
                    _format_throughput(tok_s),
                )

    rank_elapsed = time.time() - rank_t0
    summary = {
        "rank": rank,
        "world_size": world_size,
        "distributed_mode": distributed_mode,
        "representation_type": args.representation_type,
        "model_path": args.model_path,
        "input_uri": args.input_uri,
        "output_uri": args.output_uri,
        **total_stats,
        "total_elapsed_seconds": round(rank_elapsed, 2),
        "tokens_per_second": round(total_stats["tokens_processed"] / rank_elapsed, 2) if rank_elapsed > 0 else 0.0,
        "outputs_written": outputs_written,
    }
    logger.info("rank=%s 完成: %s", rank, json.dumps({k: v for k, v in summary.items() if k != "outputs_written"}, ensure_ascii=False))

    if comm is not None:
        gathered = comm.gather(summary, root=0)
        if rank == 0:
            total_tokens = int(sum(x.get("tokens_processed", 0) for x in gathered))
            # 全局耗时取所有 rank 中的最大值（并行执行，墙上时间 = 最慢 rank）
            total_elapsed = max((x.get("total_elapsed_seconds", 0) or 0) for x in gathered) if gathered else 0.0
            merged = {
                "representation_type": args.representation_type,
                "model_path": args.model_path,
                "input_uri": args.input_uri,
                "output_uri": args.output_uri,
                "world_size": world_size,
                "files_seen": int(sum(x["files_seen"] for x in gathered)),
                "files_ok": int(sum(x["files_ok"] for x in gathered)),
                "rows_seen": int(sum(x["rows_seen"] for x in gathered)),
                "rows_scored": int(sum(x["rows_scored"] for x in gathered)),
                "rows_full_nan": int(sum(x["rows_full_nan"] for x in gathered)),
                "rows_partial_nan": int(sum(x["rows_partial_nan"] for x in gathered)),
                "tokens_processed": total_tokens,
                "total_elapsed_seconds": round(total_elapsed, 2),
                "tokens_per_second": round(total_tokens / total_elapsed, 2) if total_elapsed > 0 else 0.0,
                "rank_summaries": gathered,
            }
            logger.info(
                "全局汇总: %s",
                json.dumps({k: v for k, v in merged.items() if k != "rank_summaries"}, ensure_ascii=False),
            )
            logger.info(
                "全局吞吐: rows=%s tokens=%s 耗时=%s %s",
                merged["rows_scored"], total_tokens,
                _format_duration(total_elapsed), _format_throughput(merged["tokens_per_second"]),
            )
            return merged
        return summary

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 3a PPL 评估（多机多卡，transformers + flash_attention_2）")
    parser.add_argument("--input_uri", type=str, required=True,
                        help="1_preprocess.py 的输出目录或单文件（本地或 viewfs://hdfs://）")
    parser.add_argument("--output_uri", type=str, required=True,
                        help="PPL 评估输出目录（本地或 viewfs://hdfs://）")
    parser.add_argument("--representation_type", type=str,
                        choices=[REPRESENTATION_CAPTION, REPRESENTATION_SID], required=True,
                        help="caption=使用 Qwen3-8B 打分 caption；sid=使用 OpenOneRec-8B 打分 sid token")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF 模型路径。caption: Qwen3-8B；sid: OpenOneRec pretrain-8B/converted")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"],
                        help="注意力实现，FA2 失败会自动回退 sdpa/eager")
    parser.add_argument("--local_files_only", type=int, default=1, choices=[0, 1],
                        help="是否仅使用本地模型文件（默认 1，离线加载）")
    parser.add_argument("--use_system_message", type=int, default=1, choices=[0, 1],
                        help="是否把 system_text 作为 system 消息拼进 chat template（默认 1）")
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1],
                        help="chat template 的 enable_thinking；默认 0（直接预测 target，不走 thinking）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="单个 micro-batch 的最大序列数上限（动态 batch 下实际可能小于此值；"
                             "每个样本产生 x / xcot 两条序列）")
    parser.add_argument("--max_tokens_per_batch", type=int, default=131072,
                        help="动态 batch 的 token 预算上限（默认 131072 = 32×4096）。"
                             "valid_seqs 按长度降序排序后，按当前批最长序列反推单批 bs = "
                             "min(batch_size, token_budget // max_len)。长 seq 自动小 batch 防 OOM，"
                             "短 seq 自动大 batch 保吞吐。设为 0 禁用动态 batch，退化为固定 batch_size。")
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="单条序列最大长度，超出则从左侧截断 prefix（保留完整 target）")
    parser.add_argument("--rows_per_shard", type=int, default=0,
                        help="保留兼容参数，2_eval_ppl 按 1 输入文件 → 1 输出分片写出")
    parser.add_argument("--max_files", type=int, default=0,
                        help="仅调试用；限制总输入文件数，0 表示处理全部")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅调试用；限制每个 rank 处理的文件数，0 表示不限制")
    parser.add_argument("--eta_log_interval", type=int, default=10,
                        help="文件级 ETA 日志频率：每处理 N 个文件打一次进度+ETA（默认 10）。"
                             "设为 1 每个文件都打，设为 0 仅在结束时打。批内进度独立按 50 micro-batch 间隔。")
    parser.add_argument("--dry_run", action="store_true",
                        help="只计算不写出")
    parser.add_argument("--verbose", action="store_true",
                        help="打印更详细日志")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
