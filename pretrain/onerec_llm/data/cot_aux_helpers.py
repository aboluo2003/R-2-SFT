import re
from typing import Iterable, Set, Tuple


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_BEGIN_TOKEN_RE = re.compile(r"<\|[^|>]+_begin\|>")


def parse_source_names(source_names) -> Set[str]:
    if source_names is None:
        return set()
    if isinstance(source_names, str):
        return {x.strip() for x in source_names.split(",") if x.strip()}
    if isinstance(source_names, Iterable):
        return {str(x).strip() for x in source_names if str(x).strip()}
    return set()


def source_enabled(source_name, source_names) -> bool:
    if isinstance(source_names, set):
        return str(source_name) in source_names
    return str(source_name) in parse_source_names(source_names)


def filter_sid_end_tokens(text: str) -> str:
    text = text.replace("<|sid_end|>", "")
    text = text.replace("<|goods_sid_end|>", "")
    return text


def build_empty_think_response(text: str) -> str:
    if not isinstance(text, str):
        return text

    match = _THINK_BLOCK_RE.search(text)
    if match is None:
        return text

    answer = text[match.end():]
    return "<think>\n</think>\n" + answer.lstrip("\n")


def split_think_answer(text: str) -> Tuple[str, str]:
    if not isinstance(text, str):
        return "", ""

    match = _THINK_BLOCK_RE.search(text)
    if match is None:
        return "", text
    return text[:match.end()], text[match.end():]


def find_cot_target_relative_positions(
    assistant_content: str,
    *,
    tokenizer,
    filter_sid_end_tokens: bool,
):
    """Find relative positions of ALL target SID code tokens in assistant content.

    Supports multi-target outputs. Finds every ``<|..._begin|>`` token in the
    answer part (after the optional think block) and returns the 3 code tokens
    (s_a, s_b, s_c) that follow each begin token.

    The returned positions are relative to the start of ``assistant_content``
    (i.e., include the think prefix length). The list length is always a
    multiple of 3 (``3 * num_complete_sids``). Incomplete SIDs (begin token
    near the end with fewer than 3 code tokens following) are skipped.

    Returns ``[]`` if no complete SID is found.
    """
    if filter_sid_end_tokens:
        assistant_content = globals()["filter_sid_end_tokens"](assistant_content)
    prefix_text, answer_text = split_think_answer(assistant_content)
    if not answer_text:
        return []

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)

    # Find ALL begin tokens (e.g. <|sid_begin|>, <|video_begin|>, <|goods_sid_begin|>)
    begin_indices = [
        idx for idx, token in enumerate(answer_tokens)
        if isinstance(token, str) and _BEGIN_TOKEN_RE.fullmatch(token)
    ]
    if not begin_indices:
        return []

    target_positions = []
    prefix_len = len(prefix_ids)
    answer_len = len(answer_ids)
    for begin_idx in begin_indices:
        # Each SID has exactly 3 code tokens (s_a, s_b, s_c) after the begin.
        # Skip incomplete SIDs (truncated at the end of the answer).
        if begin_idx + 3 >= answer_len:
            continue
        for offset in range(1, 4):
            rel_idx = begin_idx + offset
            target_positions.append(prefix_len + rel_idx)
    return target_positions
