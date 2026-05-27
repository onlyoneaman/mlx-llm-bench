#!/usr/bin/env python3
"""Run one model against the classification dataset and write results.json.

Invoked by cli.py via the appropriate pipx venv. Not usually run directly.
"""
import argparse
import json
import re
import time
from pathlib import Path

PROMPT_TEMPLATES = {
    "sentiment": (
        "Classify the sentiment of the following text as 'positive' or 'negative'.\n"
        "Reply with exactly one word: positive or negative.\n\n"
        "Text: {text}\n"
        "Answer:"
    ),
    "topic": (
        "Classify the topic of the following text as one of: 'world', 'sports', 'business', 'tech'.\n"
        "Reply with exactly one word.\n\n"
        "Text: {text}\n"
        "Answer:"
    ),
    "spam": (
        "Classify the following message as 'spam' or 'ham'.\n"
        "Reply with exactly one word: spam or ham.\n\n"
        "Message: {text}\n"
        "Answer:"
    ),
}

VALID = {
    "sentiment": {"positive", "negative"},
    "topic": {"world", "sports", "business", "tech"},
    "spam": {"spam", "ham"},
}


_THINK_RE = re.compile(r"<think>.*?</think>|<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)


def parse_answer(raw, task):
    # Reasoning models (Phi-4-reasoning, DeepSeek-R1 distills, Qwen3 with thinking)
    # wrap chain-of-thought in <think>...</think>. Strip it so the parser sees
    # only the post-reasoning answer.
    raw = _THINK_RE.sub("", raw)
    words = re.findall(r"[A-Za-z]+", raw.lower())
    for w in words:
        if w in VALID[task]:
            return w
    return words[0] if words else ""


def run_mlx_lm(data, model_id, max_tokens):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"[mlx-lm] loading {model_id}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_id)
    load_s = time.time() - t0
    print(f"[mlx-lm] loaded in {load_s:.1f}s", flush=True)

    sampler = make_sampler(temp=0.0)
    results = []
    for i, ex in enumerate(data):
        prompt = PROMPT_TEMPLATES[ex["task"]].format(text=ex["text"])
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        t0 = time.time()
        raw = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
        dt = time.time() - t0
        pred = parse_answer(raw, ex["task"])
        ok = pred == ex["label"]
        results.append({
            "i": i, "task": ex["task"], "text": ex["text"],
            "label": ex["label"], "pred": pred, "raw": raw,
            "correct": ok, "time_s": round(dt, 3),
        })
        print(f"  [{i+1:>2}/{len(data)}] {ex['task']:9s} {ex['label']:8s} -> {pred:8s} {'OK' if ok else 'XX'} ({dt:.2f}s)", flush=True)
    return results, load_s


def run_mlx_vlm(data, model_id, max_tokens):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"[mlx-vlm] loading {model_id}", flush=True)
    t0 = time.time()
    model, processor = load(model_id)
    load_s = time.time() - t0
    print(f"[mlx-vlm] loaded in {load_s:.1f}s", flush=True)

    config = getattr(model, "config", None)
    results = []
    for i, ex in enumerate(data):
        prompt = PROMPT_TEMPLATES[ex["task"]].format(text=ex["text"])
        formatted = apply_chat_template(
            processor, config, prompt, num_images=0, enable_thinking=False,
        )
        t0 = time.time()
        out = generate(
            model, processor, formatted,
            max_tokens=max_tokens, temperature=0.0, verbose=False,
        )
        dt = time.time() - t0
        raw = out if isinstance(out, str) else getattr(out, "text", str(out))
        pred = parse_answer(raw, ex["task"])
        ok = pred == ex["label"]
        results.append({
            "i": i, "task": ex["task"], "text": ex["text"],
            "label": ex["label"], "pred": pred, "raw": raw,
            "correct": ok, "time_s": round(dt, 3),
        })
        print(f"  [{i+1:>2}/{len(data)}] {ex['task']:9s} {ex['label']:8s} -> {pred:8s} {'OK' if ok else 'XX'} ({dt:.2f}s)", flush=True)
    return results, load_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--backend", choices=["mlx-lm", "mlx-vlm"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=250)
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text())
    runner = run_mlx_lm if args.backend == "mlx-lm" else run_mlx_vlm
    results, load_s = runner(data, args.model_id, args.max_tokens)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model_id": args.model_id,
        "backend": args.backend,
        "load_s": load_s,
        "max_tokens": args.max_tokens,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
