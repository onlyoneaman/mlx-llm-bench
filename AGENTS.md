# AGENTS.md

Project-specific instructions for AI coding agents working on mlx-llm-bench. Read this first when invoked here.

This is a **public repository** (github.com/onlyoneaman/mlx-llm-bench + huggingface.co/datasets/onlyoneaman/mlx-llm-bench). Anything you commit is visible to everyone — never include personal paths, tokens, hostnames in source.

---

## Project in one paragraph

A local LLM classification benchmark for Apple Silicon Macs with limited unified memory (≈16 GB). Three classification tasks (sentiment, topic, spam) with a deliberate easy/hard split that separates models. Runs locally via the MLX framework (mlx-lm for text-only, mlx-vlm for multimodal). Results are stored per-run in `runs/` (gitignored audit trail), aggregated into a canonical `leaderboard.json` + `leaderboard.csv` that are committed.

## Quick orientation (read these first)

1. `README.md` — user-facing overview and quick start.
2. `models.json` — the model registry. What's benchmarked.
3. `data.json` — the labeled dataset. What we benchmark against.
4. `leaderboard.json` — the canonical results snapshot.
5. `src/mlx_llm_bench/cli.py` — all subcommands.
6. `src/mlx_llm_bench/runner.py` — the inner loop (loads model, runs prompts).

After reading these you should know what changes for any request.

## Repo layout

```
mlx-llm-bench/
├── bench                  shell entrypoint -> mlx-lm venv python -> cli.py
├── data.json              68 labeled examples (sentiment + topic + spam)
├── models.json            9 models with HF repo, backend, size, notes
├── leaderboard.json       canonical snapshot (committed)
├── leaderboard.csv        flat snapshot (committed)
├── runs/                  per-run audit trail (gitignored, local only)
├── archive/               historical leaderboard snapshots (gitignored)
├── src/mlx_llm_bench/
│   ├── cli.py             argparse + subcommand handlers + run lifecycle
│   ├── runner.py          loads model, runs prompts, parses labels (per-venv)
│   └── utils.py           pure helpers: registry, cache, hardware, SHA, stats
├── pyproject.toml
├── README.md              public-facing
├── AGENTS.md              this file
└── LICENSE                MIT
```

## Environment assumptions

- macOS on Apple Silicon. mlx-lm and mlx-vlm are installed in **two separate pipx venvs**:
  - `~/.local/pipx/venvs/mlx-lm/bin/python`
  - `~/.local/pipx/venvs/mlx-vlm/bin/python`
- `bench` shell script dispatches via the mlx-lm venv. `cli.py` then subprocesses to the correct venv per model based on `models.json[<key>]["backend"]`.
- `huggingface_hub` is in both venvs. `hf` / `huggingface-cli` is its own pipx app.

## Common tasks — how to act on user requests

### "Add a new model" / "Try model X"

Three backend types supported:

| Backend | When | Loading |
|---|---|---|
| `mlx-lm` | Text-only HF models (Qwen, Llama, Phi, SmolLM, Gemma 3, …) | `snapshot_download` + `mlx_lm.load` |
| `mlx-vlm` | Multimodal HF models OR models broken on mlx-lm (Gemma 4 family — see Gotchas) | `snapshot_download` + `mlx_vlm.load` |
| `openai` | Any OpenAI-compatible HTTP server (Apple FM via `afm`, Ollama, LM Studio, `mlx_lm.server`, …) | None on our side — server is responsible |

**HF-backed models (mlx-lm / mlx-vlm):**
1. Find its 4-bit MLX repo on huggingface.co/mlx-community (or lmstudio-community if not on mlx-community). Verify it exists.
2. Choose a stable short key (lowercase-hyphenated, e.g. `phi4-mini-reasoning`).
3. Edit `models.json`:
   ```json
   "my-new-model": {
     "model_id": "mlx-community/<exact-repo-name>",
     "backend": "mlx-lm",
     "size_gb": <disk size in GB at 4-bit>,
     "notes": "<one-line: what it's good at, with a primary-source benchmark if you have one>"
   }
   ```
4. `./bench pull my-new-model`
5. `./bench run my-new-model`
6. `./bench export`
7. `git add models.json leaderboard.json leaderboard.csv` and commit.

**`openai`-backed models** (run against a local HTTP server):
1. Make sure the server is reachable (start it manually first — `bench` does not start servers).
2. Edit `models.json`:
   ```json
   "my-server-model": {
     "model_id": "<your-internal-id>",
     "backend": "openai",
     "endpoint": "http://localhost:PORT/v1",
     "remote_model": "<model name the server expects in the 'model' field>",
     "size_gb": 0.0,
     "notes": "<setup steps, prereqs, throttling caveats>"
   }
   ```
3. `./bench list` shows status `live` (reachable) or `offline` (no TCP listener).
4. `./bench run my-server-model` — no download; runner sends HTTP POSTs.
5. `./bench export` and commit as usual.

Notes for `openai` backend:
- No model weights are downloaded; `bench pull` for an openai model just prints config.
- `bench run all --cached` filters openai models by endpoint reachability (skips offline ones).
- The `seed` field is sent in the OpenAI request body; respect depends on the server.

### Annotation rubric (applies when adding/editing examples)

When you add a new example or evaluate an existing label, apply these tiebreakers:

- **Topic**: corporate transactions, earnings, market cap, M&A, layoffs, recalls = `business`, even when the company is a tech company. Product launches, technical announcements, software releases, research milestones = `tech`. Geopolitics, conflicts, disasters, international policy = `world`. Anything where a team, league, or athlete is the subject = `sports`.
- **Sentiment**: litotes ("not bad", "can't say I hated it") reads as the *opposite of the negated adjective* — generally positive. Faint praise comparatives ("better than expected, which isn't saying much") read by the dominant signal — if the qualifier negates the praise, it's negative.
- **Spam**: requires ≥2 BEC markers from {changed bank details, urgency to bypass normal approval, requests for gift cards, claimed authority, secrecy, mismatched sender pattern} to be labeled `spam`. A single suspicious framing alone stays `ham`.

Past audits caught these inconsistencies: lines 33/59 had identical-structure acquisition stories labeled opposite ways (now both `business`); lines 52/56 had near-identical litotes pairs labeled opposite ways (now line 56 is non-litotes). If you add a new edge case, search the existing data first for any similar phrasing to avoid recreating the same trap.

### "Add a new test example" / "Test with harder X cases"

1. Edit `data.json`. Each example is `{"task", "text", "label", "difficulty"}`. The `difficulty` field is `"easy"` or `"hard"` — **set it explicitly**; the harness does not infer from position.
2. Keep the file format stable: one example per line, blank line between task+difficulty blocks. The convention is easy block first, hard block at the end of each task, but ordering does not affect scoring — `difficulty` does.
3. **Changing `data.json` changes its SHA** — old leaderboard entries are no longer directly comparable. After editing:
   - `./bench run all --cached` to refresh every locally-cached model
   - `./bench export` to write a new `leaderboard.json` with the new `dataset_sha`
   - Commit `data.json` + `leaderboard.json` + `leaderboard.csv` together.
4. Cross-task balance matters less than within-task class balance. Keep labels roughly balanced inside each task.

### "Run the benchmark"

- One model: `./bench run <key>`
- All cached: `./bench run all --cached`
- JSON-format variant: `./bench run <key> --format json` — model must reply `{"label": "..."}`. Useful for diagnosing format-following vs label-accuracy.
- Multi-seed: `./bench run <key> --seeds 1 2 3` — runs N times, results tagged per-seed, summary shows mean ± std. MLX is not always bitwise deterministic at temp=0 across batch sizes / KV-cache layouts, so seeds are an honest noise floor.
- Never run all without `--cached` unless the user explicitly asks — uncached invocation downloads every model in the registry (tens of GB).

### "Compare two models" / "Compare these runs"

- `./bench history` to find run IDs.
- `./bench compare <id1> <id2>` for accuracy delta + paired McNemar exact two-sided p-value + miss overlap. The McNemar result is the headline — at n=68, Wilson CIs are wide enough that bare accuracy deltas overclaim.
- `./bench show <id>` for one run's full markdown summary, including per-task and easy/hard breakdowns with 95% Wilson CIs.

### "Publish updated results"

```bash
./bench export
git add leaderboard.json leaderboard.csv [data.json] [models.json]
git commit -m "..."
git push
# Then HF dataset (optional but matches GitHub):
hf upload onlyoneaman/mlx-llm-bench leaderboard.json leaderboard.csv data.json models.json --repo-type dataset
```

Never `git add -A` blindly — it would catch `runs/` and `archive/` if .gitignore is misconfigured. Always stage explicit files.

## Conventions

### Run IDs
`YYYYMMDD-HHMMSS_<model-key>` — sortable, parseable. Stored as `runs/<run_id>/{meta.json,results.json,summary.md}`.

### Dataset SHA
First 12 hex chars of `sha256(data.json)`. Embedded in `leaderboard.json` and in `archive/leaderboard-<ts>-<sha>.json` filenames. If two snapshots have different SHAs, do not compare their numbers — the dataset diverged.

### Hardware fingerprint
Auto-detected from `system_profiler SPHardwareDataType` on macOS. Lives at `hardware:` in `leaderboard.json`. Future cross-platform results should keep this schema even if the values look different.

### Prompts
Defined in `runner.py` (`PROMPT_TEMPLATES`) per format. All three tasks share the same shape: instruction + format directive + input + `Answer:`. `temp=0`, `max_tokens=250` (safety upper bound — non-thinking models EOS at 1–3 tokens so no slowdown; reasoning-trained models need room to finish their `<think>` block before emitting the answer).

Two formats:
- `text` (default): "Reply with exactly one word." Parser scans for first valid label.
- `json`: "Respond with `{\"label\": \"...\"}`." Parser regexes the JSON object first; if it can extract a valid label from a `{"label": "x"}` block, `format_ok=True`. Otherwise falls back to text-parse with `format_ok=False`.

### Scoring
1. Strip any `<think>...</think>` or `<thought>...</thought>` blocks from the raw output (reasoning-model CoT).
2. For `format=text`: extract the first word in the remaining response that's in `VALID[task]` — that's the prediction. `format_ok = (any valid label found)`.
3. For `format=json`: try the `{"label": "..."}` regex first; if it yields a valid label, `format_ok=True`. Else scan for a valid label anywhere (still record it as `pred`) but mark `format_ok=False`.
4. Compare prediction exactly against `label` for `correct`.
5. Stats are computed with **95% Wilson score intervals**. Significance between two runs uses **exact two-sided McNemar** on paired discordant pairs.

## Gotchas (recurring failure modes)

### Reasoning / thinking models
Three layers of defense, in this order:
1. **`enable_thinking=False`** passed to `apply_chat_template` for both backends — clean suppression for models that honor it (Qwen3, Gemma 4, SmolLM3).
2. **`max_tokens=250`** default — gives unsuppressed reasoning room to finish; non-thinking models still EOS at 1–3 tokens so no perf hit.
3. **`<think>` / `<thought>` stripping** in `parse_answer` — catches any leaked CoT from models that ignore `enable_thinking` (Phi-4-mini-reasoning, DeepSeek-R1 distills).
Despite all three, dedicated reasoning models still rank badly on this benchmark (Phi-4-mini-reasoning 58.8%, DeepSeek-R1-Distill 42.6%) because their training pushes them toward CoT even for single-token answers. **Don't add pure reasoning models to the registry as if they were general chat models** — frame them as "experimental / not recommended for classification."

### mlx-lm 0.31.3 Gemma 4 E4B regression
Issue [ml-explore/mlx-lm#1242](https://github.com/ml-explore/mlx-lm/issues/1242) — loading `mlx-community/gemma-4-e4b-it-4bit` fails with `Received 126 parameters not in model`. Workaround: use the `mlx-vlm` backend (it loads the model correctly and is the multimodal path anyway). Already configured in `models.json`. If you add other Gemma 4 variants, default to `mlx-vlm` backend until the upstream bug is fixed.

### `bench run all` without `--cached`
Would try to download every model in `models.json` first (tens of GB). Always add `--cached` unless explicitly asked otherwise.

### MoE models that don't fit
Models like Qwen3.6-35B-A3B advertise as MoE but require the full param count (~22GB at Q4) resident in RAM. **They will not fit 16GB.** Do not add them to `models.json` for this hardware target. The size budget is realistically 4B–8B dense or smaller.

### License diversity
Gemma 1/2/3/3n use Google's Gemma Terms (not OSI-open). Gemma 4 is Apache 2.0. Qwen/Llama/Mistral/Phi are mostly Apache 2.0 or similar. The benchmark code is MIT and the dataset is MIT; we never redistribute weights — `bench pull` fetches them directly from each user's HuggingFace account.

## What never to do

- Don't commit `runs/`, `archive/`, or `RESULTS.md` (already gitignored).
- Don't hardcode `/Users/onlyoneaman/...` paths anywhere. Use `Path.home()` or project-relative.
- Don't add models that don't fit 16 GB unified memory unless the user explicitly wants a "doesn't fit" baseline.
- Don't `git push --force` to `main`.
- Don't bump `schema_version` in `leaderboard.json` unless you're actually changing the schema — downstream consumers will assume incompatibility.

## Pre-publish checklist (always run before committing or pushing)

Documentation drift is the most common bug in this repo's history. Before any commit that touches code, data, or results:

1. **README leaderboard table matches `leaderboard.json`.** If you re-ran benchmarks or edited `data.json`, the table in README is almost certainly stale. Regenerate by hand or skip the table and link to `leaderboard.csv`.
2. **AGENTS.md describes the actual code.** If you changed defaults (e.g. `max_tokens` in `runner.py`), update the matching reference in the Conventions section here. The published methodology must match what `./bench run` actually does.
3. **HuggingFace README is in sync** with the GitHub README. After `git push`, run `hf upload onlyoneaman/mlx-llm-bench README.md leaderboard.json leaderboard.csv data.json models.json --repo-type dataset`.
4. **`dataset_sha` in `leaderboard.json` matches `sha256(data.json)`.** If `bench export` didn't run after a data edit, the snapshot is for an older dataset.
5. **No stale `notes` in `models.json`** referencing benchmarks the model no longer leads or behaviors that have been disproven on this dataset.

If any check fails, fix the docs before pushing. Stale docs erode trust faster than incomplete features.

## When the user asks for something not covered here

Prefer the conservative read of their request. If they say "add a model that's fast" → pick a small (≤4B) candidate from existing benchmarks rather than guessing. If they say "make this harder" → look at the misses in recent `runs/*/results.json` to identify what current models already get wrong and add more of that class.
