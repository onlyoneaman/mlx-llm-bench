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

VENVS = {
    "mlx-lm": Path.home() / ".local/pipx/venvs/mlx-lm/bin/python",
    "mlx-vlm": Path.home() / ".local/pipx/venvs/mlx-vlm/bin/python",
    # `openai` backend speaks any OpenAI-compatible HTTP server (Apple FM via
    # afm, Ollama, LM Studio, mlx-lm.server). No model load, no MLX dep —
    # uses urllib from stdlib. Routed through mlx-lm venv since it exists.
    "openai": Path.home() / ".local/pipx/venvs/mlx-lm/bin/python",
}

TASKS = ["sentiment", "topic", "spam"]


# ---------- model registry ----------

def load_models():
    return json.loads(MODELS_FILE.read_text())


def get_model(key):
    models = load_models()
    if key not in models:
        raise KeyError(f"unknown model '{key}'. try `bench list`")
    return models[key]


# ---------- HuggingFace cache ----------

def hf_cache_dir(model_id):
    return Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"


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
    """sha256(data.json) first 12 hex chars. Any change to the file (including
    schema-only edits like adding fields) shifts this. Snapshots with
    different SHAs are not directly comparable."""
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()[:12]


def difficulty_index_sets():
    """Return (easy_set, hard_set) of dataset row indices from the explicit
    `difficulty` field in data.json. Rows missing the field land in neither
    set (treated as no-difficulty)."""
    data = json.loads(DATA_FILE.read_text())
    easy, hard = set(), set()
    for i, ex in enumerate(data):
        d = ex.get("difficulty")
        if d == "easy":
            easy.add(i)
        elif d == "hard":
            hard.add(i)
    return easy, hard


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


def stats(results, task=None, difficulty=None, easy_idx=None, hard_idx=None):
    """Compute accuracy + 95% Wilson CI + format-compliance over a subset.

    Optional filters:
      task         — restrict to one of TASKS
      difficulty   — "easy" or "hard" (requires easy_idx / hard_idx)
    Returns None if subset is empty.
    """
    sub = results
    if task is not None:
        sub = [r for r in sub if r["task"] == task]
    if difficulty == "easy":
        sub = [r for r in sub if r["i"] in (easy_idx or set())]
    elif difficulty == "hard":
        sub = [r for r in sub if r["i"] in (hard_idx or set())]
    if not sub:
        return None
    correct = sum(r["correct"] for r in sub)
    total = len(sub)
    time_s = sum(r["time_s"] for r in sub)
    format_ok = sum(1 for r in sub if r.get("format_ok", True))
    lo, hi = wilson_ci(correct, total)
    return {
        "acc": round(100 * correct / total, 1),
        "ci": [lo, hi],
        "n": total,
        "correct": correct,
        "time_s": round(time_s, 2),
        "avg_s": round(time_s / total, 3),
        "format_ok_rate": round(100 * format_ok / total, 1),
    }


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
