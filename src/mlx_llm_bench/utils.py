"""Pure helpers used across the CLI.

Anything in here is side-effect-free except for reading files from the project
root and the HuggingFace cache. No subprocess, no run lifecycle — keep those
in cli.py where they belong.
"""
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime
from pathlib import Path

# ---------- paths & constants ----------

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "models.json"
DATA_FILE = ROOT / "data.json"
RUNS_DIR = ROOT / "runs"
ARCHIVE_DIR = ROOT / "archive"

import os as _os


def _venv_python(env_var, default_path):
    """Resolve a venv Python path, allowing env-var override.

    bench shell entry sets these via MLX_LM_PYTHON / MLX_VLM_PYTHON. If the
    var is unset we fall back to the pipx default. Path existence is checked
    by callers (not here — the value is allowed to be a placeholder for
    backends that aren't actively used)."""
    return Path(_os.environ.get(env_var, default_path))


VENVS = {
    "mlx-lm": _venv_python("MLX_LM_PYTHON", Path.home() / ".local/pipx/venvs/mlx-lm/bin/python"),
    "mlx-vlm": _venv_python("MLX_VLM_PYTHON", Path.home() / ".local/pipx/venvs/mlx-vlm/bin/python"),
    # `openai` backend has no model weights to load and no MLX dep. Uses urllib
    # from stdlib. Routed through whichever Python is convenient (mlx-lm venv
    # by default since it exists anyway).
    "openai": _venv_python("MLX_LM_PYTHON", Path.home() / ".local/pipx/venvs/mlx-lm/bin/python"),
}

TASKS = ["sentiment", "topic", "spam", "ifeval"]


# ---------- model registry ----------

def load_models():
    return json.loads(MODELS_FILE.read_text())


def get_model(key):
    models = load_models()
    if key not in models:
        raise KeyError(f"unknown model '{key}'. try `bench list`")
    return models[key]


# ---------- HuggingFace cache ----------

_is_cached_memo = {}


def is_cached(model_id):
    """Canonical 'do I have everything locally' check via HF Hub itself.

    Parsing index.json ourselves is fragile because HF cache snapshots can
    contain stale manifests from prior revisions that don't match the current
    weight file layout. snapshot_download(local_files_only=True) raises iff
    anything is missing, so it's the authoritative answer.
    """
    if model_id in _is_cached_memo:
        return _is_cached_memo[model_id]
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_files_only=True)
        result = True
    except Exception:
        result = False
    _is_cached_memo[model_id] = result
    return result


# ---------- dataset & run IDs ----------

def run_id(model_key):
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_key}"


def dataset_sha():
    """Content-based SHA over the canonical merged dataset.

    Hashes the load_dataset_with_validators output (data.json + sidecar
    validators) with sort_keys + compact separators. This is invariant to
    JSON formatting and to the file-split refactor — only actual content
    changes (task/text/label/difficulty/validators) move the SHA.
    """
    from mlx_llm_bench.rescore import load_dataset_with_validators
    data = load_dataset_with_validators(DATA_FILE)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------- hardware ----------

def detect_hardware():
    """Auto-detect Mac hardware fingerprint. Returns {} on non-macOS."""
    if platform.system() != "Darwin":
        return {"os": platform.system(), "machine": platform.machine()}
    info = {"os": f"macOS {platform.mac_ver()[0]}", "machine": platform.machine()}
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, timeout=5
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Model Name:"):
                info["model"] = line.split(":", 1)[1].strip()
            elif line.startswith("Chip:"):
                info["chip"] = line.split(":", 1)[1].strip()
            elif line.startswith("Memory:"):
                info["memory"] = line.split(":", 1)[1].strip()
            elif line.startswith("Total Number of Cores:"):
                info["cpu_cores"] = line.split(":", 1)[1].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return info


# ---------- statistics ----------

def wilson_ci(k, n, z=1.96):
    """Two-sided Wilson score interval for proportion k/n. Returns (lo%, hi%)
    rounded to 1 decimal. Well-behaved at small n where the normal approximation
    breaks down."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - half) * 100
    hi = min(1.0, center + half) * 100
    return (round(lo, 1), round(hi, 1))


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for paired binary outcomes.

    b = count where A correct, B wrong.
    c = count where A wrong, B correct.
    H0: b == c (no difference). Uses exact binomial since n_discordant is
    typically small.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    one_side = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(2 * one_side, 1.0)


def stats(results, task=None, difficulty=None):
    """Accuracy + Wilson CI + format-compliance over an optionally filtered subset.

    Filters: `task` matches `r["task"]`; `difficulty` matches `r.get("difficulty")`
    directly from each result row (which already carries that field). Returns
    None if subset is empty.

    Multi-seed aggregation: collapses per (task, i) via strict majority before
    computing CI. n_examples is the example count; n_calls = examples × seeds.
    Treating (example, seed) pairs as independent at temp=0 would shrink CIs by
    sqrt(N_seeds) — the seeds are highly correlated so it's invalid.

    Strict metric: correct AND format_ok. Reasoning-tuned/finetuned models
    often "guess right" while ignoring the requested response shape; strict
    accuracy makes the divergence visible (see Hermes-3 vs base Llama 3.2).
    """
    sub = results
    if task is not None:
        sub = [r for r in sub if r["task"] == task]
    if difficulty is not None:
        sub = [r for r in sub if r.get("difficulty") == difficulty]
    if not sub:
        return None

    # Single pass: collect (correct, format_ok) per (task, i).
    by_ex = {}
    for r in sub:
        key = (r["task"], r["i"])
        by_ex.setdefault(key, []).append((bool(r["correct"]), r.get("format_ok", True)))

    def majority(votes):
        return sum(votes) > len(votes) / 2

    n_examples = len(by_ex)
    correct_examples = sum(1 for v in by_ex.values() if majority([c for c, _ in v]))
    strict_correct = sum(1 for v in by_ex.values() if majority([c and f for c, f in v]))
    n_seeds = max(len(v) for v in by_ex.values()) if by_ex else 1

    n_calls = len(sub)
    time_s = sum(r["time_s"] for r in sub)
    format_ok = sum(1 for r in sub if r.get("format_ok", True))
    lo, hi = wilson_ci(correct_examples, n_examples)
    s_lo, s_hi = wilson_ci(strict_correct, n_examples)
    return {
        "acc": round(100 * correct_examples / n_examples, 1),
        "ci": [lo, hi],
        "strict_acc": round(100 * strict_correct / n_examples, 1),
        "strict_ci": [s_lo, s_hi],
        "n": n_examples,
        "n_calls": n_calls,
        "n_seeds": n_seeds,
        "correct": correct_examples,
        "strict_correct": strict_correct,
        "time_s": round(time_s, 2),
        "avg_s": round(time_s / n_calls, 3),
        "format_ok_rate": round(100 * format_ok / n_calls, 1),
    }


def breakdown(results):
    """Compute the canonical (overall, per_task_dict, easy, hard) quartet in one
    call. Used by `_write_summary` and `cmd_export` — keeps the field names
    consistent between markdown summaries and the leaderboard JSON schema.
    """
    return {
        "overall": stats(results),
        "per_task": {t: stats(results, task=t) for t in TASKS},
        "easy": stats(results, difficulty="easy"),
        "hard": stats(results, difficulty="hard"),
    }


def fmt_acc(s, decimals=1):
    """Render 'acc% [lo-hi]' from a stats dict. One place to control floating-
    point format so the leaderboard table doesn't drift between `:.0f` and `:.1f`."""
    return f"{s['acc']:.{decimals}f}% [{s['ci'][0]:.{decimals}f}–{s['ci'][1]:.{decimals}f}]"


def seed_stats(results):
    """Per-seed accuracy when 'seed' field is present and multi-seed.
    Returns dict {seed: acc%} or None if single seed / missing field."""
    seeds = sorted({r.get("seed") for r in results if "seed" in r})
    if len(seeds) <= 1:
        return None
    by_seed = {}
    for s in seeds:
        rs = [r for r in results if r.get("seed") == s]
        if rs:
            by_seed[s] = round(100 * sum(r["correct"] for r in rs) / len(rs), 1)
    return by_seed
