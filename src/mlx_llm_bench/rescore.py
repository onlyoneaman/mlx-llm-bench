"""Re-score saved runs in place with the current scoring rules.

Why: results.json stores the full raw model output per example. The scoring
(parse_answer, validate_ifeval) can change without rerunning the models.
This module is the canonical scoring implementation; runner.py imports from
here so live runs use the same logic.

Fixes applied vs the original scoring:
1. Word count uses `\b[\w']+\b` regex (matches em-dash-joined words like
   "there—my" as 2 words, not 1) instead of `str.split()`.
2. Cleanup strips special chat tokens (<|im_end|>, <|endoftext|>, etc.) in
   addition to <think>/<thought> blocks. Fixes Qwen2.5-Coder which leaks
   `<|im_end|>` literal text into every response.
"""
import json
import re
from pathlib import Path

VALID = {
    "sentiment": {"positive", "negative"},
    "topic": {"world", "sports", "business", "tech"},
    "spam": {"spam", "ham"},
}

# Strip CoT blocks and chat special tokens before any scoring.
_THINK_RE = re.compile(r"<think>.*?</think>|<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_SPECIAL_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|file_separator\|>"
    r"|<\|fim_prefix\|>|<\|fim_middle\|>|<\|fim_suffix\|>"
    r"|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>"
    r"|<\|begin_of_text\|>|<\|end_of_text\|>",
    re.IGNORECASE,
)
_JSON_LABEL_RE = re.compile(r'\{\s*"label"\s*:\s*"([A-Za-z]+)"\s*\}', re.IGNORECASE)
_WORD_RE = re.compile(r"\b[\w']+\b")


def clean_raw(raw):
    raw = _THINK_RE.sub("", raw)
    raw = _SPECIAL_RE.sub("", raw)
    return raw


def word_count(text):
    """Linguistically-defensible word count — handles em-dashes correctly."""
    return len(_WORD_RE.findall(text))


def parse_answer(raw, task, fmt="text"):
    """Return (pred, format_ok). Match runner.py semantics exactly."""
    raw_clean = clean_raw(raw)
    if fmt == "json":
        m = _JSON_LABEL_RE.search(raw_clean)
        if m:
            label = m.group(1).lower()
            if label in VALID[task]:
                return label, True
            return label, False
        words = re.findall(r"[A-Za-z]+", raw_clean.lower())
        for w in words:
            if w in VALID[task]:
                return w, False
        return (words[0] if words else ""), False
    words = re.findall(r"[A-Za-z]+", raw_clean.lower())
    for w in words:
        if w in VALID[task]:
            return w, True
    return (words[0] if words else ""), False


# ---------- IFEval validators ----------

def _v_word_count_exact(raw, v): return word_count(raw) == v["n"]
def _v_word_count_max(raw, v):   return word_count(raw) <= v["n"]
def _v_word_count_min(raw, v):   return word_count(raw) >= v["n"]
def _v_contains_word(raw, v):
    return re.search(rf"\b{re.escape(v['word'])}\b", raw, re.IGNORECASE) is not None
def _v_not_contains_word(raw, v):
    return re.search(rf"\b{re.escape(v['word'])}\b", raw, re.IGNORECASE) is None
def _v_not_contains_letter(raw, v):
    return v["letter"].lower() not in raw.lower()
def _v_not_contains_chars(raw, v):
    return not any(c in raw for c in v["chars"])
def _v_regex_match(raw, v):
    return re.match(v["pattern"], raw.strip(), re.DOTALL) is not None
def _v_starts_with(raw, v): return raw.strip().startswith(v["prefix"])
def _v_ends_with(raw, v):   return raw.strip().endswith(v["suffix"])
def _extract_json(raw, open_ch, close_ch):
    """Pull the first balanced [..] or {..} substring out of `raw` and json-load it.
    Handles nested brackets, ignores brackets inside strings. Returns the parsed
    Python object or None. Avoids the greedy/non-greedy regex pitfalls that
    misparse nested arrays and split across separate objects.
    """
    # Fast path: maybe the whole thing is valid JSON.
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start = raw.find(open_ch)
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except Exception:
                        break
        # didn't close cleanly — try next opener
        start = raw.find(open_ch, start + 1)
    return None


def _v_json_array_length(raw, v):
    arr = _extract_json(raw, "[", "]")
    return isinstance(arr, list) and len(arr) == v["n"]


def _v_json_has_keys(raw, v):
    obj = _extract_json(raw, "{", "}")
    return isinstance(obj, dict) and all(k in obj for k in v["keys"])
def _v_exact_word_set(raw, v):
    cleaned = re.sub(r'[^A-Za-z]', '', raw).upper()
    return cleaned in [o.upper() for o in v["options"]]
def _v_sentence_count_exact(raw, v):
    sents = re.split(r'(?<=[.!?])\s+', raw.strip())
    return len([s for s in sents if s.strip()]) == v["n"]
def _v_paragraph_count_exact(raw, v):
    paras = [p for p in raw.strip().split("\n\n") if p.strip()]
    return len(paras) == v["n"]
def _v_line_count_exact(raw, v):
    lines = [l for l in raw.strip().split("\n") if l.strip()]
    return len(lines) == v["n"]
def _v_contains_exact(raw, v): return v["text"] in raw

_VALIDATORS = {
    "word_count_exact": _v_word_count_exact,
    "word_count_max": _v_word_count_max,
    "word_count_min": _v_word_count_min,
    "contains_word": _v_contains_word,
    "not_contains_word": _v_not_contains_word,
    "not_contains_letter": _v_not_contains_letter,
    "not_contains_chars": _v_not_contains_chars,
    "regex_match": _v_regex_match,
    "starts_with": _v_starts_with,
    "ends_with": _v_ends_with,
    "json_array_length": _v_json_array_length,
    "json_has_keys": _v_json_has_keys,
    "exact_word_set": _v_exact_word_set,
    "sentence_count_exact": _v_sentence_count_exact,
    "paragraph_count_exact": _v_paragraph_count_exact,
    "line_count_exact": _v_line_count_exact,
    "contains_exact": _v_contains_exact,
}


def validate_ifeval(raw, validators):
    """Returns (all_pass, per_validator_results) after stripping CoT + special tokens."""
    raw_clean = clean_raw(raw).strip()
    results = []
    for v in validators:
        fn = _VALIDATORS.get(v["type"])
        if fn is None:
            results.append({"type": v["type"], "pass": False, "error": "unknown validator"})
            continue
        try:
            ok = bool(fn(raw_clean, v))
        except Exception:
            ok = False
        results.append({"type": v["type"], "pass": ok})
    return all(r["pass"] for r in results), results


def load_dataset_with_validators(data_path, validators_path=None, strict=True):
    """Load data.json and merge ifeval_validators.json into each ifeval row.

    Keeps data.json clean of nested validator structures (so HuggingFace's
    dataset viewer renders well) while still letting our runner/scorer find
    the validators for IFEval items.

    `validators_path` defaults to `ifeval_validators.json` next to data.json.
    `strict=True` (default) raises if:
      - any ifeval row has no validators attached
      - any validators key points to a non-ifeval row or out-of-range index
    Without these checks, `all([])` would silently mark unvalidated IFEval rows
    as passing.
    """
    data = json.loads(Path(data_path).read_text())
    if validators_path is None:
        validators_path = Path(data_path).parent / "ifeval_validators.json"
    p = Path(validators_path)
    vmap = {}
    if p.exists():
        vmap = json.loads(p.read_text())
        for i, ex in enumerate(data):
            key = str(i)
            if key in vmap:
                ex["validators"] = vmap[key]

    if strict:
        errors = []
        for i, ex in enumerate(data):
            if ex.get("task") == "ifeval":
                vs = ex.get("validators") or []
                if not vs:
                    errors.append(f"  ifeval row i={i} has no validators ({ex.get('text','')[:60]!r})")
        for key in vmap:
            try:
                i = int(key)
            except ValueError:
                errors.append(f"  validator key {key!r} is not a numeric index")
                continue
            if i < 0 or i >= len(data):
                errors.append(f"  validator key {i} out of range (dataset has {len(data)} rows)")
                continue
            if data[i].get("task") != "ifeval":
                errors.append(f"  validator key {i} points to a {data[i].get('task')!r} row, not ifeval")
        if errors:
            raise ValueError(
                "ifeval_validators.json / data.json validation failed:\n" + "\n".join(errors)
            )
    return data


def rescore_run(rdir, data_by_i):
    """Re-score one run's results.json in place. Returns (n_examples, n_changed)."""
    res_path = rdir / "results.json"
    if not res_path.exists():
        return 0, 0
    res = json.loads(res_path.read_text())
    fmt = res.get("format", "text")
    n_changed = 0
    for r in res["results"]:
        old_correct = r["correct"]
        if r["task"] == "ifeval":
            ex = data_by_i.get(r["i"])
            validators = (ex or {}).get("validators", []) if ex else []
            if not validators:
                continue
            new_correct, v_res = validate_ifeval(r["raw"], validators)
            passed = sum(1 for vr in v_res if vr["pass"])
            r["correct"] = new_correct
            r["validators"] = v_res
            r["pred"] = f"{passed}/{len(v_res)} validators"
            r["format_ok"] = True
        else:
            pred, format_ok = parse_answer(r["raw"], r["task"], fmt)
            r["pred"] = pred
            r["format_ok"] = format_ok
            r["correct"] = (pred == r["label"])
        if r["correct"] != old_correct:
            n_changed += 1
    res_path.write_text(json.dumps(res, indent=2))
    return len(res["results"]), n_changed


