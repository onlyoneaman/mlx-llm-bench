#!/usr/bin/env python3
"""mlx-llm-bench CLI.

Subcommands:
  list                   show registered models
  pull <model>           pre-download a model's weights
  run <model>            run benchmark, save to runs/<id>/
  run all [--cached]     run all (or only locally cached) models
  history [--model M]    list past runs
  show <run-id>          print one run's summary
  compare <id1> <id2>    side-by-side comparison
  report                 aggregate latest run per model into RESULTS.md
  serve <model>          start OpenAI-compatible server for a model
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "models.json"
DATA_FILE = ROOT / "data.json"
RUNS_DIR = ROOT / "runs"
RUNNER = Path(__file__).resolve().parent / "runner.py"

VENVS = {
    "mlx-lm": Path.home() / ".local/pipx/venvs/mlx-lm/bin/python",
    "mlx-vlm": Path.home() / ".local/pipx/venvs/mlx-vlm/bin/python",
}

TASKS = ["sentiment", "topic", "spam"]


def load_models():
    return json.loads(MODELS_FILE.read_text())


def get_model(key):
    models = load_models()
    if key not in models:
        sys.exit(f"unknown model '{key}'. try `bench list`")
    return models[key]


def hf_cache_dir(model_id):
    return Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"


_is_cached_memo = {}


def is_cached(model_id):
    """Use HF Hub's canonical "do I have everything locally" check.

    `snapshot_download(local_files_only=True)` succeeds iff every file the
    repo declares is present in cache. Parsing index.json ourselves is fragile
    because HF cache snapshots can contain stale manifests from prior revisions
    that don't match the current weight file layout.
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


def run_id(model_key):
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_key}"


def dataset_sha():
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()[:12]


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


def hard_index_set():
    """Identify which dataset rows are 'hard' by convention: last N per task.

    HARD_COUNTS reflects the structure used when data.json was authored
    (easy block first, hard block appended). If this assumption stops holding,
    add an explicit `difficulty` field to data.json and switch to that.
    """
    HARD_COUNTS = {"sentiment": 8, "topic": 8, "spam": 7}
    data = json.loads(DATA_FILE.read_text())
    by_task = {}
    for i, ex in enumerate(data):
        by_task.setdefault(ex["task"], []).append(i)
    hard = set()
    for task, idxs in by_task.items():
        n = HARD_COUNTS.get(task, 0)
        for i in idxs[-n:]:
            hard.add(i)
    return hard


# ----- subcommands -----

def cmd_list(_args):
    models = load_models()
    print(f"{'KEY':25s} {'SIZE':>6s}  {'CACHED':>6s}  BACKEND     MODEL_ID")
    print("-" * 100)
    for key, m in models.items():
        cached = "yes" if is_cached(m["model_id"]) else "no"
        print(f"{key:25s} {m['size_gb']:>5.1f}G  {cached:>6s}  {m['backend']:10s}  {m['model_id']}")
        if m.get("notes"):
            print(f"  {' ' * 23} {m['notes']}")


def cmd_pull(args):
    m = get_model(args.model)
    py = VENVS[m["backend"]]
    print(f"pulling {m['model_id']}...")
    subprocess.check_call([
        str(py), "-c",
        f"from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id='{m['model_id']}'))",
    ])


def _do_run(model_key):
    m = get_model(model_key)
    if not is_cached(m["model_id"]):
        print(f"model '{model_key}' not cached. run: bench pull {model_key}")
        return None

    rid = run_id(model_key)
    rdir = RUNS_DIR / rid
    rdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": rid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "model_id": m["model_id"],
        "backend": m["backend"],
        "host": socket.gethostname(),
        "data_file": str(DATA_FILE.name),
        "data_count": len(json.loads(DATA_FILE.read_text())),
    }
    (rdir / "meta.json").write_text(json.dumps(meta, indent=2))

    py = VENVS[m["backend"]]
    out = rdir / "results.json"
    print(f"\n=== {model_key} -> {rid} ===")
    t0 = time.time()
    rc = subprocess.call([
        str(py), str(RUNNER),
        "--model-id", m["model_id"],
        "--backend", m["backend"],
        "--data", str(DATA_FILE),
        "--out", str(out),
    ])
    wall_s = time.time() - t0
    if rc != 0:
        print(f"FAILED (rc={rc})")
        meta["status"] = "failed"
        meta["wall_s"] = round(wall_s, 1)
        (rdir / "meta.json").write_text(json.dumps(meta, indent=2))
        return rid

    meta["status"] = "ok"
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["wall_s"] = round(wall_s, 1)
    (rdir / "meta.json").write_text(json.dumps(meta, indent=2))
    _write_summary(rdir)
    return rid


def cmd_run(args):
    if args.model == "all":
        models = load_models()
        keys = list(models.keys())
        if args.cached:
            keys = [k for k in keys if is_cached(models[k]["model_id"])]
        if not keys:
            print("no models to run")
            return
        print(f"will run: {', '.join(keys)}")
        for k in keys:
            _do_run(k)
    else:
        _do_run(args.model)


def _stats(results, task=None):
    sub = [r for r in results if task is None or r["task"] == task]
    if not sub:
        return None
    correct = sum(r["correct"] for r in sub)
    total = len(sub)
    time_s = sum(r["time_s"] for r in sub)
    return {"acc": 100 * correct / total, "n": total, "correct": correct,
            "time_s": time_s, "avg_s": time_s / total}


def _write_summary(rdir):
    meta = json.loads((rdir / "meta.json").read_text())
    res = json.loads((rdir / "results.json").read_text())
    rs = res["results"]
    overall = _stats(rs)
    lines = [
        f"# {meta['run_id']}",
        "",
        f"- Model: `{meta['model_id']}` ({meta['model_key']})",
        f"- Backend: {meta['backend']}",
        f"- Started: {meta.get('started_at', '?')}",
        f"- Wall time: {meta.get('wall_s', '?')}s",
        f"- Load time: {res['load_s']:.1f}s",
        "",
        f"## Overall: {overall['correct']}/{overall['n']} = **{overall['acc']:.1f}%**",
        "",
        f"Inference: {overall['time_s']:.1f}s ({overall['avg_s']:.2f}s/example)",
        "",
        "## Per task",
        "",
        "| Task | Accuracy | Avg time |",
        "|---|---|---|",
    ]
    for t in TASKS:
        s = _stats(rs, t)
        if s:
            lines.append(f"| {t} | {s['correct']}/{s['n']} = {s['acc']:.1f}% | {s['avg_s']:.2f}s |")
    lines += ["", "## Misses", ""]
    for r in rs:
        if not r["correct"]:
            txt = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
            lines.append(f"- [{r['task']}] **{r['label']} → {r['pred']}**: {txt}")
    (rdir / "summary.md").write_text("\n".join(lines))


def _all_runs():
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        meta_p = d / "meta.json"
        res_p = d / "results.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            m["_dir"] = d
            m["_has_results"] = res_p.exists()
            runs.append(m)
    return runs


def cmd_history(args):
    runs = _all_runs()
    if args.model:
        runs = [r for r in runs if r.get("model_key") == args.model]
    if not runs:
        print("no runs yet")
        return
    print(f"{'RUN_ID':40s} {'MODEL':22s} {'ACC':>7s}  {'TIME':>7s}  STATUS")
    print("-" * 100)
    for r in runs:
        acc = "-"
        if r.get("_has_results"):
            res = json.loads((r["_dir"] / "results.json").read_text())
            s = _stats(res["results"])
            acc = f"{s['acc']:.1f}%"
        wall = f"{r.get('wall_s', 0):.0f}s"
        print(f"{r['run_id']:40s} {r.get('model_key', '?'):22s} {acc:>7s}  {wall:>7s}  {r.get('status', '?')}")


def cmd_show(args):
    rdir = RUNS_DIR / args.run_id
    summary = rdir / "summary.md"
    if not summary.exists():
        sys.exit(f"no summary at {summary}")
    print(summary.read_text())


def cmd_compare(args):
    a = RUNS_DIR / args.id1 / "results.json"
    b = RUNS_DIR / args.id2 / "results.json"
    if not (a.exists() and b.exists()):
        sys.exit("one or both run ids missing results.json")
    ra = json.loads(a.read_text())
    rb = json.loads(b.read_text())
    name_a, name_b = args.id1, args.id2
    print(f"\n{name_a}  vs  {name_b}\n")
    sa, sb = _stats(ra["results"]), _stats(rb["results"])
    print(f"{'Overall':12s}  {sa['acc']:5.1f}%  vs  {sb['acc']:5.1f}%   (delta {sb['acc']-sa['acc']:+.1f})")
    for t in TASKS:
        ta, tb = _stats(ra["results"], t), _stats(rb["results"], t)
        if ta and tb:
            print(f"  {t:10s}  {ta['acc']:5.1f}%  vs  {tb['acc']:5.1f}%   (delta {tb['acc']-ta['acc']:+.1f})")
    print()
    miss_a = {(r["task"], r["i"]) for r in ra["results"] if not r["correct"]}
    miss_b = {(r["task"], r["i"]) for r in rb["results"] if not r["correct"]}
    print(f"misses unique to {name_a}: {len(miss_a - miss_b)}")
    print(f"misses unique to {name_b}: {len(miss_b - miss_a)}")
    print(f"misses shared:            {len(miss_a & miss_b)}")


def cmd_serve(args):
    m = get_model(args.model)
    py = VENVS[m["backend"]]
    server_mod = "mlx_lm.server" if m["backend"] == "mlx-lm" else "mlx_vlm.server"
    cmd = [str(py), "-m", server_mod, "--model", m["model_id"],
           "--host", args.host, "--port", str(args.port)]
    print(f"starting {server_mod} for {m['model_id']} on http://{args.host}:{args.port}")
    print(f"  (Ctrl-C to stop. OpenAI-compatible: POST /v1/chat/completions)")
    os.execvp(cmd[0], cmd)


def cmd_export(_args):
    """Export latest run per model into leaderboard.json and leaderboard.csv.

    These are the canonical, committable artifacts. runs/ stays gitignored.
    """
    runs = _all_runs()
    latest_by_model = {}
    for r in runs:
        if r.get("status") == "ok" and r.get("_has_results"):
            latest_by_model[r["model_key"]] = r
    if not latest_by_model:
        print("no completed runs yet")
        return

    models = load_models()
    hw = detect_hardware()
    ds_sha = dataset_sha()
    hard = hard_index_set()

    entries = []
    for key, r in latest_by_model.items():
        res = json.loads((r["_dir"] / "results.json").read_text())
        rs = res["results"]
        overall = _stats(rs)
        per_task = {t: _stats(rs, t) for t in TASKS}
        easy = [x for x in rs if x["i"] not in hard]
        hardrs = [x for x in rs if x["i"] in hard]
        easy_correct = sum(x["correct"] for x in easy)
        hard_correct = sum(x["correct"] for x in hardrs)

        m = models.get(key, {})
        entries.append({
            "model_key": key,
            "model_id": r["model_id"],
            "backend": r["backend"],
            "size_gb": m.get("size_gb"),
            "overall_acc": round(overall["acc"], 1),
            "sentiment_acc": round(per_task["sentiment"]["acc"], 1) if per_task["sentiment"] else None,
            "topic_acc": round(per_task["topic"]["acc"], 1) if per_task["topic"] else None,
            "spam_acc": round(per_task["spam"]["acc"], 1) if per_task["spam"] else None,
            "easy_acc": round(100 * easy_correct / len(easy), 1) if easy else None,
            "hard_acc": round(100 * hard_correct / len(hardrs), 1) if hardrs else None,
            "avg_inference_s": round(overall["avg_s"], 3),
            "load_s": round(res["load_s"], 1),
            "run_id": r["run_id"],
            "run_date": r.get("started_at", "")[:10],
        })

    entries.sort(key=lambda e: -e["overall_acc"])

    payload = {
        "schema_version": 1,
        "dataset_sha": ds_sha,
        "dataset_file": DATA_FILE.name,
        "dataset_count": len(json.loads(DATA_FILE.read_text())),
        "hardware": hw,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "results": entries,
    }

    json_path = ROOT / "leaderboard.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = ROOT / "leaderboard.csv"
    if entries:
        fields = list(entries[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(entries)

    # Local archive (gitignored) — preserves snapshots over time.
    # Filename embeds timestamp + dataset_sha so different dataset versions
    # are visually distinguishable in the directory listing.
    archive_dir = ROOT / "archive"
    archive_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"leaderboard-{stamp}-{ds_sha}"
    (archive_dir / f"{base}.json").write_text(json.dumps(payload, indent=2))
    if csv_path.exists():
        (archive_dir / f"{base}.csv").write_text(csv_path.read_text())

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"archived to {archive_dir.name}/{base}.{{json,csv}}")
    print(f"\ndataset sha:  {ds_sha}")
    print(f"hardware:     {hw.get('model', '?')} · {hw.get('chip', '?')} · {hw.get('memory', '?')}")
    print(f"models:       {len(entries)}")


def cmd_report(_args):
    runs = _all_runs()
    latest_by_model = {}
    for r in runs:
        if r.get("status") == "ok" and r.get("_has_results"):
            latest_by_model[r["model_key"]] = r
    if not latest_by_model:
        print("no completed runs yet")
        return

    lines = ["# Classification benchmark — latest results", "",
             "Mac mini M4 base 16GB · mlx-lm 0.31.3 / mlx-vlm 0.5.0 · temp=0", "",
             "| Model | Overall | Sentiment | Topic | Spam | Run ID |",
             "|---|---|---|---|---|---|"]
    rows = []
    for key, r in latest_by_model.items():
        res = json.loads((r["_dir"] / "results.json").read_text())
        s = _stats(res["results"])
        per = {t: _stats(res["results"], t) for t in TASKS}
        rows.append((key, s, per, r["run_id"]))
    rows.sort(key=lambda x: -x[1]["acc"])
    for key, s, per, rid in rows:
        cells = [f"{per[t]['correct']}/{per[t]['n']} ({per[t]['acc']:.0f}%)" if per[t] else "-" for t in TASKS]
        lines.append(f"| {key} | **{s['acc']:.1f}%** | {cells[0]} | {cells[1]} | {cells[2]} | `{rid}` |")
    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(out.read_text())
    print(f"\nwrote {out}")


# ----- entry -----

def main():
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show registered models").set_defaults(fn=cmd_list)

    sp = sub.add_parser("pull", help="download a model")
    sp.add_argument("model")
    sp.set_defaults(fn=cmd_pull)

    sp = sub.add_parser("run", help="run benchmark (model name or 'all')")
    sp.add_argument("model")
    sp.add_argument("--cached", action="store_true", help="when 'all', only run cached models")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("history", help="list past runs")
    sp.add_argument("--model", default=None)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("show", help="show a run's summary")
    sp.add_argument("run_id")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("compare", help="compare two runs")
    sp.add_argument("id1")
    sp.add_argument("id2")
    sp.set_defaults(fn=cmd_compare)

    sub.add_parser("report", help="aggregate latest-per-model into RESULTS.md").set_defaults(fn=cmd_report)

    sub.add_parser("export", help="export leaderboard.{json,csv} for committing").set_defaults(fn=cmd_export)

    sp = sub.add_parser("serve", help="start OpenAI-compatible server for a model")
    sp.add_argument("model")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--host", default="127.0.0.1")
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
