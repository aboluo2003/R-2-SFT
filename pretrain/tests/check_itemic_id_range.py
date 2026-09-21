#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_vocab_from_tokenizer_json(model_dir: str):
    tk_path = Path(model_dir) / "tokenizer.json"
    if not tk_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tk_path}")

    data = json.loads(tk_path.read_text(encoding="utf-8"))

    vocab = {}
    # 主词表
    model_vocab = data.get("model", {}).get("vocab", {})
    if isinstance(model_vocab, dict):
        for tok, tid in model_vocab.items():
            vocab[tok] = int(tid)

    # added_tokens（有些特殊 token 在这里）
    for item in data.get("added_tokens", []):
        content = item.get("content")
        tid = item.get("id")
        if content is not None and tid is not None:
            vocab[content] = int(tid)

    return vocab


def minmax(ids):
    if not ids:
        return None, None
    return min(ids), max(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="HF model dir, e.g. .../converted",
    )
    parser.add_argument(
        "--configured_start",
        type=int,
        default=151669,
        help="configured itemic_id_range start",
    )
    parser.add_argument(
        "--configured_end",
        type=int,
        default=176246,
        help="configured itemic_id_range end",
    )
    args = parser.parse_args()

    vocab = load_vocab_from_tokenizer_json(args.model_dir)

    sid_begin = "<|sid_begin|>"
    sid_end = "<|sid_end|>"

    sid_begin_id = vocab.get(sid_begin, None)
    sid_end_id = vocab.get(sid_end, None)

    # 支持任意层：<s_a_x> / <s_b_x> / <s_c_x> / <s_d_x> ...
    p_itemic = re.compile(r"^<s_([a-z])_(\d+)>$")

    layer2ids = {}
    layer2count = {}
    all_itemic_ids = []

    for tok, tid in vocab.items():
        m = p_itemic.match(tok)
        if not m:
            continue
        layer = m.group(1)
        layer2ids.setdefault(layer, []).append(tid)
        layer2count[layer] = layer2count.get(layer, 0) + 1
        all_itemic_ids.append(tid)

    itemic_min, itemic_max = minmax(all_itemic_ids)

    print("=" * 80)
    print(f"model_dir: {args.model_dir}")
    print(f"vocab_size(observed): {len(vocab)}")
    print("-" * 80)
    print(f"{sid_begin}: {sid_begin_id}")
    print(f"{sid_end}:   {sid_end_id}")

    sid_range = None
    if sid_begin_id is not None and sid_end_id is not None:
        sid_range = [min(sid_begin_id, sid_end_id), max(sid_begin_id, sid_end_id)]
        print(f"sid token range: {sid_range}")

    print("-" * 80)
    sorted_layers = sorted(layer2ids.keys())
    if not sorted_layers:
        print("No itemic tokens found (<s_[a-z]_*>).")
    else:
        for layer in sorted_layers:
            l_min, l_max = minmax(layer2ids[layer])
            print(f"s_{layer} count={layer2count[layer]}, id_range=[{l_min}, {l_max}]")

    if sorted_layers:
        layers_display = "/".join([f"s_{x}" for x in sorted_layers])
        print(
            f"s_[{''.join(sorted_layers)}] total count={len(all_itemic_ids)}, "
            f"id_range=[{itemic_min}, {itemic_max}] ({layers_display})"
        )

    # 建议区间
    only_itemic_range = None
    with_sid_range = None
    if itemic_min is not None and itemic_max is not None:
        only_itemic_range = [itemic_min, itemic_max]

    if (
        itemic_min is not None
        and sid_begin_id is not None
        and sid_end_id is not None
    ):
        with_sid_range = [
            min(itemic_min, sid_begin_id, sid_end_id),
            max(itemic_max, sid_begin_id, sid_end_id),
        ]

    print("-" * 80)
    print(f"configured itemic_id_range = [{args.configured_start}, {args.configured_end}]")
    if only_itemic_range:
        print(f"recommended (only itemic <s_[a-z]_*>) : {only_itemic_range}")
    if with_sid_range:
        print(f"recommended (include sid begin/end): {with_sid_range}")

    cfg_ok_only = only_itemic_range == [args.configured_start, args.configured_end]
    cfg_ok_with_sid = with_sid_range == [args.configured_start, args.configured_end]

    print("-" * 80)
    if cfg_ok_only:
        print("MATCH: configured range == ONLY itemic range")
    elif cfg_ok_with_sid:
        print("MATCH: configured range == itemic + <|sid_begin|>/<|sid_end|> range")
    else:
        print("NOT MATCH: configured range differs from both recommendations")
    print("=" * 80)


if __name__ == "__main__":
    main()
