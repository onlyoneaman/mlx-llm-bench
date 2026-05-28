#!/usr/bin/env python3
"""mlx-llm-bench CLI — subcommand dispatcher and run lifecycle.

Pure helpers live in utils.py. This file owns:
  - argparse + subcommand routing
  - run lifecycle (_do_run, _all_runs, _write_summary) — they orchestrate
    subprocess, filesystem, and meta.json updates
  - cmd_* handlers
"""
import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mlx_llm_bench.utils import (
    ARCHIVE_DIR, DATA_FILE, ROOT, RUNS_DIR, TASKS, VENVS,
    breakdown,
    dataset_sha,
    detect_hardware,
    fmt_acc,
    get_model,
    is_cached,
    load_models,
    mcnemar_p,
    run_id,
    seed_stats,
    stats,
)


def _endpoint_reachable(endpoint, timeout=2.0):
    """Light reachability check for openai-backend endpoints. Returns True if
    the host accepts a TCP connection on the given port — doesn't require a
    successful HTTP response."""
    import socket as _socket
    from urllib.parse import urlparse
    u = urlparse(endpoint)
    host = u.hostname or "localhost"
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with _socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

RUNNER = Path(__file__).resolve().parent / "runner.py"


# ---------- subcommands ----------

def cmd_list(_args):
    models = load_models()
    print(f"{'KEY':25s} {'SIZE':>6s}  {'STATUS':>7s}  BACKEND     MODEL_ID")
    print("-" * 100)
    for key, m in models.items():
        if m["backend"] == "openai":
            status = "live" if _endpoint_reachable(m.get("endpoint", "")) else "offline"
        else:
            status = "cached" if is_cached(m["model_id"]) else "missing"
        print(f"{key:25s} {m['size_gb']:>5.1f}G  {status:>7s}  {m['backend']:10s}  {m['model_id']}")
        if m.get("notes"):
            print(f"  {' ' * 23} {m['notes']}")


def cmd_pull(args):
    m = get_model(args.model)
    if m["backend"] == "openai":
        print(f"'{args.model}' is an openai-backend model — nothing to pull.")
        print(f"  endpoint:     {m.get('endpoint', '(unset)')}")
        print(f"  remote_model: {m.get('remote_model', '(unset)')}")
        if m.get("notes"):
            print(f"  notes:        {m['notes']}")
        return
    py = VENVS[m["backend"]]
    print(f"pulling {m['model_id']}...")
    subprocess.check_call([
        str(py), "-c",
        f"from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id='{m['model_id']}'))",
    ])


DEFAULT_RUN_TIMEOUT_S = int(os.environ.get("BENCH_RUN_TIMEOUT_S", 3600))


def _do_run(model_key, seeds=(1,), fmt="text", timeout_s=DEFAULT_RUN_TIMEOUT_S):
    try:
        m = get_model(model_key)
    except KeyError as e:
        sys.exit(str(e))

    backend = m["backend"]
    if backend == "openai":
        endpoint = m.get("endpoint")
        if not endpoint:
            print(f"model '{model_key}' missing 'endpoint' in models.json")
            return None
        if not _endpoint_reachable(endpoint):
            print(f"model '{model_key}' endpoint {endpoint} not reachable — start the server first")
            return None
    else:
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
        "backend": backend,
        "host": socket.gethostname(),
        "data_file": str(DATA_FILE.name),
        "data_count": len(json.loads(DATA_FILE.read_text())),
        "dataset_sha": dataset_sha(),
        "format": fmt,
        "seeds": list(seeds),
    }
    if backend == "openai":
        meta["endpoint"] = m.get("endpoint")
        meta["remote_model"] = m.get("remote_model")
    (rdir / "meta.json").write_text(json.dumps(meta, indent=2))

    py = VENVS[backend]
    out = rdir / "results.json"
    print(f"\n=== {model_key} -> {rid} ===")
    t0 = time.time()
    cmd = [
        str(py), str(RUNNER),
        "--model-id", m["model_id"],
        "--backend", backend,
        "--data", str(DATA_FILE),
        "--out", str(out),
        "--format", fmt,
        "--seeds", *[str(s) for s in seeds],
    ]
    if backend == "openai":
        cmd += ["--endpoint", m["endpoint"], "--remote-model", m.get("remote_model", model_key)]
    try:
        rc = subprocess.call(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        wall_s = time.time() - t0
        print(f"FAILED — timeout after {timeout_s}s. Override with BENCH_RUN_TIMEOUT_S env var.")
        meta["status"] = "timeout"
        meta["wall_s"] = round(wall_s, 1)
        meta["timeout_s"] = timeout_s
        (rdir / "meta.json").write_text(json.dumps(meta, indent=2))
        return rid
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


def _model_runnable(m):
    """Is this model ready to run on the local box right now?"""
    if m["backend"] == "openai":
        return _endpoint_reachable(m.get("endpoint", ""))
    return is_cached(m["model_id"])


def cmd_run(args):
    seeds = tuple(args.seeds)
    fmt = args.format
    if args.model == "all":
        models = load_models()
        keys = list(models.keys())
        if args.cached:
            keys = [k for k in keys if _model_runnable(models[k])]
        if not keys:
            print("no models to run")
            return
        print(f"will run: {', '.join(keys)} (seeds={list(seeds)} format={fmt})")
        for k in keys:
            _do_run(k, seeds=seeds, fmt=fmt)
    else:
        _do_run(args.model, seeds=seeds, fmt=fmt)


def _write_summary(rdir):
    meta = json.loads((rdir / "meta.json").read_text())
    res = json.loads((rdir / "results.json").read_text())
    rs = res["results"]
    overall = stats(rs)
    seed_acc = seed_stats(rs)

    lines = [
        f"# {meta['run_id']}",
        "",
        f"- Model: `{meta['model_id']}` ({meta['model_key']})",
        f"- Backend: {meta['backend']}",
        f"- Format: {meta.get('format', 'text')}",
        f"- Seeds: {meta.get('seeds', [1])}",
        f"- Dataset sha: `{meta.get('dataset_sha', '?')}`",
        f"- Started: {meta.get('started_at', '?')}",
        f"- Wall time: {meta.get('wall_s', '?')}s",
        f"- Load time: {res['load_s']:.1f}s",
        "",
        f"## Overall: {overall['correct']}/{overall['n']} = **{fmt_acc(overall)}** · format_ok={overall['format_ok_rate']:.0f}%",
        "",
        f"Inference: {overall['time_s']:.1f}s ({overall['avg_s']:.2f}s/example)",
        "",
    ]
    if seed_acc:
        per_seed_str = ", ".join(f"seed {s}={a}%" for s, a in seed_acc.items())
        mean = sum(seed_acc.values()) / len(seed_acc)
        std = (sum((a - mean) ** 2 for a in seed_acc.values()) / len(seed_acc)) ** 0.5
        lines += [
            f"Per-seed: {per_seed_str}  (mean {mean:.1f}%, std {std:.2f})",
            "",
        ]
    lines += [
        "## Per task",
        "",
        "| Task | Accuracy (95% CI) | format_ok | Avg time |",
        "|---|---|---|---|",
    ]
    for t in TASKS:
        s = stats(rs, task=t)
        if s:
            lines.append(f"| {t} | {s['correct']}/{s['n']} = {fmt_acc(s)} | {s['format_ok_rate']:.0f}% | {s['avg_s']:.2f}s |")
    lines += ["", "## Easy vs Hard", "",
              "| Difficulty | Accuracy (95% CI) |",
              "|---|---|"]
    for label in ("easy", "hard"):
        s = stats(rs, difficulty=label)
        if s:
            lines.append(f"| {label} | {s['correct']}/{s['n']} = {fmt_acc(s)} |")
    # Misses: aggregate per (task, i) via majority-vote so multi-seed runs don't
    # list examples that majority-passed but failed on one seed.
    lines += ["", "## Misses", ""]
    by_ex = {}
    for r in rs:
        by_ex.setdefault((r["task"], r["i"]), []).append(r)
    for (task, i), reps in by_ex.items():
        if sum(r["correct"] for r in reps) > len(reps) / 2:
            continue
        r = reps[0]  # representative for text + pred
        txt = r["text"][:80] + ("..." if len(r["text"]) > 80 else "")
        fmt_flag = "" if r.get("format_ok", True) else " [format_error]"
        lines.append(f"- [{r['task']}/{r.get('difficulty','?')}] **{r['label']} → {r['pred']}**{fmt_flag}: {txt}")
    (rdir / "summary.md").write_text("\n".join(lines))


def _all_runs():
    """Scan runs/ and return parsed meta.json contents. One pass per CLI
    invocation. Malformed meta.json files are skipped with a stderr warning."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        try:
            m = json.loads(meta_p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping {d.name} — meta.json unreadable: {e}", file=sys.stderr)
            continue
        m["_dir"] = d
        m["_has_results"] = (d / "results.json").exists()
        runs.append(m)
    return runs


def cmd_history(args):
    runs = _all_runs()
    if args.model:
        runs = [r for r in runs if r.get("model_key") == args.model]
    if not runs:
        print("no runs yet")
        return
    print(f"{'RUN_ID':40s} {'MODEL':22s} {'FMT':>5s}  {'ACC (95% CI)':>18s}  {'fmt_ok':>6s}  {'TIME':>7s}  STATUS")
    print("-" * 122)
    for r in runs:
        acc_str = "-"
        fmt_ok_str = "-"
        if r.get("_has_results"):
            try:
                res = json.loads((r["_dir"] / "results.json").read_text())
                s = stats(res["results"])
                if s:
                    acc_str = fmt_acc(s)
                    fmt_ok_str = f"{s['format_ok_rate']:.0f}%"
            except (json.JSONDecodeError, OSError):
                acc_str = "(err)"
        fmt = r.get("format", "text")[:5]
        wall = f"{r.get('wall_s', 0):.0f}s"
        print(f"{r['run_id']:40s} {r.get('model_key', '?'):22s} {fmt:>5s}  {acc_str:>18s}  {fmt_ok_str:>6s}  {wall:>7s}  {r.get('status', '?')}")


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
    rs_a = ra["results"]
    rs_b = rb["results"]

    # Check dataset_sha compatibility BEFORE printing any side-by-side numbers.
    # Different SHAs are not comparable by repo rule; printing deltas first
    # would mislead.
    sha_a = json.loads((RUNS_DIR / args.id1 / "meta.json").read_text()).get("dataset_sha")
    sha_b = json.loads((RUNS_DIR / args.id2 / "meta.json").read_text()).get("dataset_sha")
    if sha_a and sha_b and sha_a != sha_b:
        print(f"\n⚠️  dataset_sha differs ({sha_a} vs {sha_b}) — these runs are not comparable.")
        print(f"   Rerun one of them against the current dataset to compare.\n")
        return

    print(f"\n{args.id1}  vs  {args.id2}\n")
    def line(label, ta, tb):
        print(f"  {label:10s}  {fmt_acc(ta)}  vs  {fmt_acc(tb)}   delta {tb['acc']-ta['acc']:+.1f}")

    sa, sb = stats(rs_a), stats(rs_b)
    line("Overall", sa, sb)
    for t in TASKS:
        ta, tb = stats(rs_a, task=t), stats(rs_b, task=t)
        if ta and tb:
            line(t, ta, tb)
    for label in ("easy", "hard"):
        ta, tb = stats(rs_a, difficulty=label), stats(rs_b, difficulty=label)
        if ta and tb:
            line(label, ta, tb)

    # Aggregate per (task, i) — majority-correct across seeds — before pairing.
    # Pairing on (task, i, seed) would treat highly-correlated seed replicates as
    # independent and inflate n_discordant by N_seeds, artificially shrinking p.
    def per_example_correct(rs):
        by_ex = {}
        for r in rs:
            by_ex.setdefault((r["task"], r["i"]), []).append(bool(r["correct"]))
        return {k: (sum(v) > len(v) / 2) for k, v in by_ex.items()}

    ka, kb = per_example_correct(rs_a), per_example_correct(rs_b)
    shared = sorted(set(ka) & set(kb))
    if not shared:
        print("\nNo overlapping examples for paired test.")
        return
    b_only = sum(1 for k in shared if ka[k] and not kb[k])
    c_only = sum(1 for k in shared if not ka[k] and kb[k])
    n_disc = b_only + c_only
    p = mcnemar_p(b_only, c_only)
    print(
        f"\nMcNemar exact two-sided  p = {p:.4f}   "
        f"(A-correct/B-wrong={b_only}, A-wrong/B-correct={c_only}, n_discordant={n_disc})"
    )
    if p < 0.05:
        print("→ SIGNIFICANT at α=0.05")
    else:
        print("→ NOT significant at α=0.05 — gap may be noise")
    print()
    # Miss sets use the same per-example majority as McNemar above.
    miss_a = {k for k in shared if not ka[k]}
    miss_b = {k for k in shared if not kb[k]}
    print(f"misses unique to {args.id1}: {len(miss_a - miss_b)}")
    print(f"misses unique to {args.id2}: {len(miss_b - miss_a)}")
    print(f"misses shared:            {len(miss_a & miss_b)}")


def cmd_inspect(args):
    """Show raw model outputs from a run — by default, only the misses.
    Filters: --task <name>, --i <index> (show one example fully), --all (show
    correct items too)."""
    res_p = RUNS_DIR / args.run_id / "results.json"
    if not res_p.exists():
        sys.exit(f"no results at {res_p}")
    res = json.loads(res_p.read_text())
    rs = res["results"]
    meta = json.loads((RUNS_DIR / args.run_id / "meta.json").read_text())

    if args.i is not None:
        matched = [r for r in rs if r["i"] == args.i]
        if not matched:
            sys.exit(f"no example with i={args.i}")
        rows = matched
    else:
        rows = rs if args.all else [r for r in rs if not r["correct"]]
        if args.task:
            rows = [r for r in rows if r["task"] == args.task]
        if not rows:
            print("no rows match filter")
            return
        print(f"# {meta['run_id']}  ({meta.get('model_key','?')}, {meta.get('format','text')}, n={len(rs)})")
        print(f"showing {len(rows)} row(s){' (misses only)' if not args.all else ''}{f' filter task={args.task}' if args.task else ''}")
        print()
    for r in rows:
        _print_record(r, show_prompt=args.prompt)


def _print_record(r, show_prompt=False, max_raw=400):
    """Compact human-readable print of one result row.

    Raw output is truncated at `max_raw` chars; the only caller-side controllable
    knob (cmd_inspect) doesn't currently set this. Reasoning models produce
    multi-KB raw outputs that nobody wants spammed verbatim — 400 chars is
    enough to see the chain-of-thought signature.
    """
    mark = "OK" if r["correct"] else "XX"
    fmt = "" if r.get("format_ok", True) else " !fmt"
    diff = r.get("difficulty", "?")
    print(f"[i={r['i']:>3} {r['task']:9s}/{diff:4s} seed={r.get('seed','?')}] {mark}{fmt}  label={r.get('label')}  pred={r.get('pred')}  ({r.get('time_s')}s)")
    print(f"  TEXT: {r['text']}")
    if show_prompt:
        p = r.get("prompt_sent")
        if p:
            print("  PROMPT: " + p.replace("\n", "\n          "))
        else:
            print("  PROMPT: (not recorded — this run pre-dates prompt_sent capture)")
    raw = r.get("raw", "")
    if len(raw) > max_raw:
        raw = raw[:max_raw] + f"\n  ... [truncated, {len(r['raw'])} chars total]"
    print("  RAW : " + raw.replace("\n", "\n        "))
    if "validators" in r:
        print("  VALIDATORS:")
        for v in r["validators"]:
            print(f"    {'✓' if v['pass'] else '✗'} {v['type']}")
    print()


def cmd_serve(args):
    try:
        m = get_model(args.model)
    except KeyError as e:
        sys.exit(str(e))
    if m["backend"] == "openai":
        sys.exit(
            f"'{args.model}' uses backend=openai (endpoint={m.get('endpoint','?')}); "
            "the server is external and `bench serve` doesn't manage it. "
            "Start it yourself (e.g. `afm` for Apple FM, `ollama serve`, `mlx_lm.server`)."
        )
    py = VENVS[m["backend"]]
    server_mod = "mlx_lm.server" if m["backend"] == "mlx-lm" else "mlx_vlm.server"
    cmd = [str(py), "-m", server_mod, "--model", m["model_id"],
           "--host", args.host, "--port", str(args.port)]
    print(f"starting {server_mod} for {m['model_id']} on http://{args.host}:{args.port}")
    print(f"  (Ctrl-C to stop. OpenAI-compatible: POST /v1/chat/completions)")
    os.execvp(cmd[0], cmd)


def cmd_rescore(args):
    """Re-apply the canonical scoring logic to saved results.json files.

    Use this after improving parse_answer / validate_ifeval — historical raw
    outputs get re-scored without re-running the models. The leaderboard
    export will then reflect the new scoring on the next `bench export`.
    """
    from mlx_llm_bench.rescore import load_dataset_with_validators, rescore_run
    data = load_dataset_with_validators(DATA_FILE)
    data_by_i = {i: ex for i, ex in enumerate(data)}
    if not RUNS_DIR.exists():
        print("no runs dir")
        return
    total_runs = 0
    total_changed = 0
    for rdir in sorted(RUNS_DIR.iterdir()):
        meta_p = rdir / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("status") != "ok":
            continue
        if args.sha and meta.get("dataset_sha") != args.sha:
            continue
        n, ch = rescore_run(rdir, data_by_i)
        if n > 0:
            # Even when no `correct` flipped, `pred` / `format_ok` / validators
            # may have changed (e.g. cleanup of special tokens). Regenerate the
            # markdown summary so `bench show` doesn't display stale text.
            _write_summary(rdir)
            print(f"  {meta.get('model_key','?'):28s} fmt={meta.get('format','?'):4s} sha={meta.get('dataset_sha','?')}  changed: {ch}/{n}")
            total_runs += 1
            total_changed += ch
    print(f"\nRescored {total_runs} run(s), {total_changed} total cell changes.")


def cmd_export(args):
    runs = _all_runs()
    ds_sha = dataset_sha()
    allow_stale = getattr(args, "allow_stale", False)

    fresh = {}
    stale_skipped = []
    for r in runs:
        if r.get("status") != "ok" or not r.get("_has_results"):
            continue
        run_sha = r.get("dataset_sha")
        if run_sha and run_sha != ds_sha and not allow_stale:
            stale_skipped.append(r)
            continue
        key = (r["model_key"], r.get("format", "json"))
        fresh[key] = r
    latest = fresh

    if stale_skipped:
        print(f"skipped {len(stale_skipped)} stale run(s) (dataset_sha != current {ds_sha}). Pass --allow-stale to include them:")
        for r in stale_skipped:
            print(f"    {r.get('model_key','?')} (fmt={r.get('format','?')}) → ran on {r.get('dataset_sha','?')}")
    if not latest:
        print("no fresh completed runs to export. Rerun `bench run all --cached` against the current dataset, or pass --allow-stale.")
        return

    models = load_models()
    hw = detect_hardware()

    entries = []
    for (key, fmt), r in latest.items():
        res = json.loads((r["_dir"] / "results.json").read_text())
        b = breakdown(res["results"])
        overall, per_task = b["overall"], b["per_task"]
        easy_s, hard_s = b["easy"], b["hard"]
        seed_acc = seed_stats(res["results"])
        seed_std = None
        if seed_acc:
            vals = list(seed_acc.values())
            mean = sum(vals) / len(vals)
            seed_std = round((sum((a - mean) ** 2 for a in vals) / len(vals)) ** 0.5, 2)

        def opt(d, k):
            return d[k] if d else None

        entries.append({
            "model_key": key,
            "model_id": r["model_id"],
            "backend": r["backend"],
            "format": fmt,
            "size_gb": models.get(key, {}).get("size_gb"),
            "overall_acc": overall["acc"],
            "overall_ci": overall["ci"],
            "strict_acc": overall["strict_acc"],
            "strict_ci": overall["strict_ci"],
            "format_ok_rate": overall["format_ok_rate"],
            "sentiment_acc": opt(per_task["sentiment"], "acc"),
            "topic_acc": opt(per_task["topic"], "acc"),
            "spam_acc": opt(per_task["spam"], "acc"),
            "easy_acc": opt(easy_s, "acc"),
            "hard_acc": opt(hard_s, "acc"),
            "hard_ci": opt(hard_s, "ci"),
            "avg_inference_s": overall["avg_s"],
            "load_s": round(res["load_s"], 1),
            "seeds": r.get("seeds", [1]),
            "seed_std_acc": seed_std,
            "dataset_sha_at_run": r.get("dataset_sha", "?"),
            "run_id": r["run_id"],
            "run_date": r.get("started_at", "")[:10],
        })

    entries.sort(key=lambda e: (-e["overall_acc"], e["model_key"]))

    payload = {
        "schema_version": 2,
        "dataset_sha": ds_sha,
        "dataset_file": DATA_FILE.name,
        "dataset_count": len(json.loads(DATA_FILE.read_text())),
        "hardware": hw,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": {
            "temperature": 0.0,
            "max_tokens": 250,
            "think_strip": True,
            "ci_method": "wilson_95",
            "significance_test": "mcnemar_exact_two_sided",
        },
        "results": entries,
    }

    if allow_stale:
        stale = [e for e in entries if e["dataset_sha_at_run"] and e["dataset_sha_at_run"] != ds_sha]
        if stale:
            print(f"⚠️  including {len(stale)} stale row(s) (--allow-stale was set):")
            for e in stale:
                print(f"    {e['model_key']} (fmt={e['format']}) → ran on {e['dataset_sha_at_run']}")

    json_path = ROOT / "leaderboard.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = ROOT / "leaderboard.csv"
    if entries:
        flat = []
        for e in entries:
            f = dict(e)
            f["overall_ci_lo"] = e["overall_ci"][0] if e.get("overall_ci") else None
            f["overall_ci_hi"] = e["overall_ci"][1] if e.get("overall_ci") else None
            f["hard_ci_lo"] = e["hard_ci"][0] if e.get("hard_ci") else None
            f["hard_ci_hi"] = e["hard_ci"][1] if e.get("hard_ci") else None
            f.pop("overall_ci", None)
            f.pop("hard_ci", None)
            f["seeds"] = ",".join(str(s) for s in e["seeds"])
            flat.append(f)
        fields = list(flat[0].keys())
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    ARCHIVE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"leaderboard-{stamp}-{ds_sha}"
    (ARCHIVE_DIR / f"{base}.json").write_text(json.dumps(payload, indent=2))
    if csv_path.exists():
        (ARCHIVE_DIR / f"{base}.csv").write_text(csv_path.read_text())

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"archived to {ARCHIVE_DIR.name}/{base}.{{json,csv}}")
    print(f"\ndataset sha:  {ds_sha}")
    print(f"hardware:     {hw.get('model', '?')} · {hw.get('chip', '?')} · {hw.get('memory', '?')}")
    print(f"results rows: {len(entries)}")


# ---------- entry ----------

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
    sp.add_argument("--seeds", type=int, nargs="+", default=[1],
                    help="one or more seeds; multi-seed runs aggregate into mean ± std")
    sp.add_argument("--format", choices=["text", "json"], default="json",
                    help="response format: json (default; {\"label\":\"x\"}) or text (free word)")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("history", help="list past runs")
    sp.add_argument("--model", default=None)
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("show", help="show a run's summary")
    sp.add_argument("run_id")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("inspect", help="show raw model outputs (misses by default)")
    sp.add_argument("run_id")
    sp.add_argument("--task", default=None, help="filter to one task (sentiment/topic/spam/ifeval)")
    sp.add_argument("--i", type=int, default=None, help="show one example by dataset index")
    sp.add_argument("--all", action="store_true", help="show correct items too, not just misses")
    sp.add_argument("--prompt", action="store_true", help="also show the chat-template-wrapped prompt as the model received it")
    sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("compare", help="compare two runs (paired McNemar)")
    sp.add_argument("id1")
    sp.add_argument("id2")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("export", help="export leaderboard.{json,csv} for committing")
    sp.add_argument("--allow-stale", action="store_true",
                    help="include runs whose dataset_sha doesn't match the current dataset (skipped by default)")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("rescore", help="re-apply scoring to historical results.json files in place")
    sp.add_argument("--sha", default=None, help="restrict to one dataset_sha")
    sp.set_defaults(fn=cmd_rescore)

    sp = sub.add_parser("serve", help="start OpenAI-compatible server for a model")
    sp.add_argument("model")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--host", default="127.0.0.1")
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
