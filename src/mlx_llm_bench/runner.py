#!/usr/bin/env python3
"""Run one model against the classification dataset and write results.json.

Invoked by cli.py via the appropriate pipx venv. Not usually run directly.

Supports:
  --format {text,json}  free-word reply vs '{"label":"x"}' JSON
  --seeds N [N ...]     one or more seeds; results are tagged per-seed
"""
import argparse
import json
import re
import time
from pathlib import Path

PROMPT_TEMPLATES = {
    "text": {
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
    },
    "json": {
        "sentiment": (
            "Classify the sentiment of the following text as 'positive' or 'negative'.\n"
            "Respond with valid JSON in this exact format:\n"
            "  {{\"label\": \"positive\"}}  or  {{\"label\": \"negative\"}}\n"
            "Output ONLY the JSON object, nothing else.\n\n"
            "Text: {text}\n"
            "JSON:"
        ),
        "topic": (
            "Classify the topic of the following text as one of: 'world', 'sports', 'business', 'tech'.\n"
            "Respond with valid JSON in this exact format:\n"
            "  {{\"label\": \"world\"}}  (use the chosen label).\n"
            "Output ONLY the JSON object, nothing else.\n\n"
            "Text: {text}\n"
            "JSON:"
        ),
        "spam": (
            "Classify the following message as 'spam' or 'ham'.\n"
            "Respond with valid JSON in this exact format:\n"
            "  {{\"label\": \"spam\"}}  or  {{\"label\": \"ham\"}}\n"
            "Output ONLY the JSON object, nothing else.\n\n"
            "Message: {text}\n"
            "JSON:"
        ),
    },
}

# Scoring functions are the canonical implementation in rescore.py — runner.py
# and the `bench rescore` command both use the same logic so a re-scoring pass
# is identical to a re-run.
from mlx_llm_bench.rescore import (  # noqa: E402
    load_dataset_with_validators,
    parse_answer,
    validate_ifeval,
)


def _seed_mlx(seed):
    """Seed mlx + python random + numpy where available."""
    import random as _random
    _random.seed(seed)
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _record(i, ex, pred, raw, ok, format_ok, dt, seed, validator_results=None):
    rec = {
        "i": i, "task": ex["task"], "text": ex["text"],
        "difficulty": ex.get("difficulty"),
        "label": ex["label"], "pred": pred, "raw": raw,
        "correct": ok, "format_ok": format_ok,
        "time_s": round(dt, 3), "seed": seed,
    }
    if validator_results is not None:
        rec["validators"] = validator_results
    return rec


def _build_prompt(ex, fmt):
    """For ifeval items the example's `text` is the full prompt. For
    classification items we wrap with the format-specific template."""
    if ex["task"] == "ifeval":
        return ex["text"]
    return PROMPT_TEMPLATES[fmt][ex["task"]].format(text=ex["text"])


def _score(ex, raw, fmt):
    """Return (pred_string, format_ok, correct, validator_results_or_none)."""
    if ex["task"] == "ifeval":
        all_pass, vres = validate_ifeval(raw, ex.get("validators", []))
        passed = sum(1 for r in vres if r["pass"])
        pred = f"{passed}/{len(vres)} validators"
        return pred, True, all_pass, vres
    pred, format_ok = parse_answer(raw, ex["task"], fmt=fmt)
    return pred, format_ok, pred == ex["label"], None


def run_mlx_lm(data, model_id, max_tokens, fmt, seeds):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"[mlx-lm] loading {model_id}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_id)
    load_s = time.time() - t0
    print(f"[mlx-lm] loaded in {load_s:.1f}s", flush=True)

    results = []
    for seed in seeds:
        _seed_mlx(seed)
        sampler = make_sampler(temp=0.0)
        if len(seeds) > 1:
            print(f"--- seed {seed} ---", flush=True)
        for i, ex in enumerate(data):
            prompt = _build_prompt(ex, fmt)
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
            pred, format_ok, ok, vres = _score(ex, raw, fmt)
            results.append(_record(i, ex, pred, raw, ok, format_ok, dt, seed, vres))
            mark = "OK" if ok else "XX"
            fmt_flag = "" if format_ok else " !fmt"
            label_str = ex.get("label", "")[:10] if isinstance(ex.get("label"), str) else "?"
            pred_str = str(pred)[:18]
            print(f"  [{i+1:>2}/{len(data)}] {ex['task']:9s} {label_str:10s} -> {pred_str:18s} {mark}{fmt_flag} ({dt:.2f}s)", flush=True)
    return results, load_s


def run_mlx_vlm(data, model_id, max_tokens, fmt, seeds):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"[mlx-vlm] loading {model_id}", flush=True)
    t0 = time.time()
    model, processor = load(model_id)
    load_s = time.time() - t0
    print(f"[mlx-vlm] loaded in {load_s:.1f}s", flush=True)

    config = getattr(model, "config", None)
    results = []
    for seed in seeds:
        _seed_mlx(seed)
        if len(seeds) > 1:
            print(f"--- seed {seed} ---", flush=True)
        for i, ex in enumerate(data):
            prompt = _build_prompt(ex, fmt)
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
            pred, format_ok, ok, vres = _score(ex, raw, fmt)
            results.append(_record(i, ex, pred, raw, ok, format_ok, dt, seed, vres))
            mark = "OK" if ok else "XX"
            fmt_flag = "" if format_ok else " !fmt"
            label_str = ex.get("label", "")[:10] if isinstance(ex.get("label"), str) else "?"
            pred_str = str(pred)[:18]
            print(f"  [{i+1:>2}/{len(data)}] {ex['task']:9s} {label_str:10s} -> {pred_str:18s} {mark}{fmt_flag} ({dt:.2f}s)", flush=True)
    return results, load_s


def run_openai(data, endpoint, model_name, max_tokens, fmt, seeds):
    """OpenAI-compatible HTTP backend — works for any server speaking
    /v1/chat/completions (Apple FM via afm, Ollama, LM Studio, mlx-lm.server).
    """
    import json as _json
    import urllib.error
    import urllib.request

    print(f"[openai] endpoint={endpoint} model={model_name}", flush=True)
    chat_url = endpoint.rstrip("/") + "/chat/completions"

    results = []
    for seed in seeds:
        if len(seeds) > 1:
            print(f"--- seed {seed} ---", flush=True)
        for i, ex in enumerate(data):
            prompt = _build_prompt(ex, fmt)
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": seed,
            }
            req = urllib.request.Request(
                chat_url,
                data=_json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = _json.loads(resp.read())
                raw = body["choices"][0]["message"]["content"]
            except urllib.error.URLError as e:
                raw = f"__HTTP_ERROR__: {e}"
            except Exception as e:
                raw = f"__ERROR__: {e}"
            dt = time.time() - t0

            pred, format_ok, ok, vres = _score(ex, raw, fmt)
            results.append(_record(i, ex, pred, raw, ok, format_ok, dt, seed, vres))
            mark = "OK" if ok else "XX"
            fmt_flag = "" if format_ok else " !fmt"
            label_str = ex.get("label", "")[:10] if isinstance(ex.get("label"), str) else "?"
            pred_str = str(pred)[:18]
            print(f"  [{i+1:>2}/{len(data)}] {ex['task']:9s} {label_str:10s} -> {pred_str:18s} {mark}{fmt_flag} ({dt:.2f}s)", flush=True)
    return results, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--backend", choices=["mlx-lm", "mlx-vlm", "openai"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--format", choices=["text", "json"], default="json")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--endpoint", default=None,
                   help="OpenAI-compatible base URL (e.g. http://localhost:9999/v1); required for openai backend")
    p.add_argument("--remote-model", default=None,
                   help="model name the remote server expects in the 'model' field (required for openai backend)")
    args = p.parse_args()

    data = load_dataset_with_validators(args.data)
    if args.backend == "mlx-lm":
        results, load_s = run_mlx_lm(data, args.model_id, args.max_tokens, args.format, args.seeds)
    elif args.backend == "mlx-vlm":
        results, load_s = run_mlx_vlm(data, args.model_id, args.max_tokens, args.format, args.seeds)
    elif args.backend == "openai":
        if not args.endpoint or not args.remote_model:
            raise SystemExit("openai backend requires --endpoint and --remote-model")
        results, load_s = run_openai(data, args.endpoint, args.remote_model, args.max_tokens, args.format, args.seeds)
    else:
        raise SystemExit(f"unknown backend: {args.backend}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model_id": args.model_id,
        "backend": args.backend,
        "load_s": load_s,
        "max_tokens": args.max_tokens,
        "format": args.format,
        "seeds": args.seeds,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
