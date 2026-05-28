# mlx-llm-bench

Local LLM classification benchmark for Apple Silicon Macs with limited unified memory.

Built around the **16GB Mac mini class** — focuses on candidates that actually fit and stay fast on consumer hardware. Runs entirely offline once weights are cached.

📊 **Dataset & leaderboard on HuggingFace:** [huggingface.co/datasets/onlyoneaman/mlx-llm-bench](https://huggingface.co/datasets/onlyoneaman/mlx-llm-bench)

## What it measures

Three classification tasks with balanced easy + hard examples (68 total):

- **Sentiment** — positive / negative (15 easy + 8 hard, incl. sarcasm, litotes, faint praise)
- **Topic** — world / sports / business / tech (16 easy + 8 hard, incl. tech-keyword pressure)
- **Spam** — spam / ham (14 easy + 7 hard, incl. sophisticated BEC scams)

Each model gets the same prompts at `temp=0`, response parsed to a single label, scored exact-match.

The easy / hard split is the key signal — strong models often max out easy and diverge dramatically on hard.

## Quick start

Requires Python 3.10+, pipx, and the MLX runtimes:

```bash
pipx install mlx-lm    # text models
pipx install mlx-vlm   # multimodal models (Gemma 4, etc.)

git clone https://github.com/onlyoneaman/mlx-llm-bench
cd mlx-llm-bench

./bench list                       # see registered models
./bench pull gemma3-4b-qat         # download weights from HuggingFace
./bench run gemma3-4b-qat          # run the benchmark
./bench history                    # list past runs
./bench export                     # write leaderboard.{json,csv}
```

## Subcommands

| Command | Purpose |
|---|---|
| `bench list` | Show registered models and cached state |
| `bench pull <model>` | Pre-download a model's weights |
| `bench run <model>` | Run benchmark; saves to `runs/<id>/` |
| `bench run all --cached` | Run every locally-cached model |
| `bench history` | List past runs with accuracy |
| `bench show <run-id>` | Print one run's full summary |
| `bench compare <id1> <id2>` | Side-by-side accuracy + miss diff |
| `bench report` | Generate `RESULTS.md` (latest per model) |
| `bench export` | Write `leaderboard.{json,csv}` for committing |
| `bench serve <model>` | Start OpenAI-compatible HTTP server |

## Current leaderboard (Mac mini M4 16GB · n=100 · JSON format)

Canonical source: [`leaderboard.csv`](./leaderboard.csv) / [`leaderboard.json`](./leaderboard.json). Snapshot below regenerated from `bench export` against `dataset_sha 2f0dc8844537`.

| Rank | Model | Acc (95% CI) | Easy | Hard | fmt_ok | Avg time | Size |
|---|---|---|---|---|---|---|---|
| 1 | **gemma3-12b-qat** | **93.0%** [86–97] | 98% | 83% | 100% | 1.38s | 8.0GB |
| 2 | **llama-3.2-3b** | **91.0%** [84–95] | 97% | 80% | 100% | **0.42s** | **1.8GB** |
| 3 | ministral-3-8b | 90.0% [83–94] | 98% | 74% | 100% | 3.55s | 5.6GB |
| 4 | gemma4-e4b | 89.0% [81–94] | 98% | 71% | 100% | 0.58s | 5.2GB |
| 4 | llama-3.1-8b | 89.0% [81–94] | 97% | 74% | 100% | 0.93s | 4.2GB |
| 4 | mistral-nemo-minitron-8b | 89.0% [81–94] | 97% | 74% | 100% | 12.18s | 4.4GB |
| 7 | nemotron-nano-9b | 88.0% [80–93] | 94% | 77% | 95% | 9.82s | 5.0GB |
| 7 | qwen3-8b | 88.0% [80–93] | 97% | 71% | 100% | 0.83s | 5.0GB |
| 9 | phi4-mini-instruct | 87.0% [79–92] | 95% | 71% | 100% | 0.53s | 2.2GB |
| 10 | qwen2.5-coder-7b | 86.0% [78–92] | 97% | 66% | 100% | 0.89s | 4.7GB |
| 11 | gemma3-4b-qat | 85.0% [77–91] | 94% | 69% | 99% | 0.54s | 2.6GB |
| 12 | hermes-3-llama-3.2-3b | 80.0% [71–87] | 92% | 57% | **28%** ⚠️ | 0.39s | 1.7GB |
| 13 | smollm3-3b | 77.0% [68–84] | 88% | 57% | 97% | 0.58s | 1.8GB |
| 14 | deepseek-r1-distill-7b | 64.0% [54–73] | 69% | 54% | **38%** | 10.11s | 4.5GB |
| 15 | phi4-mini-reasoning | 52.0% [42–62] | 54% | 49% | **27%** | 6.08s | 2.2GB |

## Which model should I run?

For a **16 GB Mac mini class** machine, picking from this bench's evidence:

- 🎯 **Default daily driver: `llama-3.2-3b`.** 91% accuracy at 0.42 s/example in 1.8 GB. Pareto-dominates everything except gemma3-12b. Validated against Llama 3.1 8B (89%, 0.93s) — newer & smaller wins on both axes.
- 🧠 **Maximum accuracy: `gemma3-12b-qat`.** 93% but 1.4 s/example and 8 GB. Use when accuracy matters more than speed, and close other apps first.
- 👁️ **Multimodal: `gemma4-e4b`.** Vision + native audio. 89% on text classification. Only fully-multimodal model in the registry.
- 🔁 **Strict instruction-following needed (clean JSON output): `llama-3.2-3b` or `phi4-mini-instruct`.** Both hit 100% format compliance. Avoid finetunes for this — see hermes-3 finding below.
- ❌ **Don't use for classification**:
  - `phi4-mini-reasoning`, `deepseek-r1-distill-7b` — reasoning-tuned, emit `<think>` blocks, 27–38% fmt_ok, 10–24× slower
  - `hermes-3-llama-3.2-3b` — finetune **broke** the base Llama 3.2 3B's structured-output ability. 100% → 28% fmt_ok after Hermes-3 RLHF. Cautionary tale about alignment finetunes for structured tasks
  - `mistral-nemo-minitron-8b`, `nemotron-nano-9b` — NVIDIA distillations are wildly slow (10–12s/ex) without an accuracy gain

> Caveat on significance: at n=100, 95% Wilson CIs are ±8 pp around accuracies in the 85-95% range. Sub-3-point gaps are likely noise. Use `./bench compare <id1> <id2>` to get the paired McNemar p-value for a real significance test.

## Annotation rubric

For ambiguous "hard" examples, these tiebreakers were applied — published so anyone reading the dataset (or any LLM evaluated on it) can see how reasonable disagreements were resolved.

- **Topic**: stories featuring a corporate transaction, earnings, market cap, or M&A are labeled **business** even when the subject is a tech company. ("Apple acquired AI startup for $200M" → business; "Apple's M5 chip launch" → tech.)
- **Sentiment**: litotes ("not bad", "can't say I hated it") read as the *opposite* of the negated adjective and are labeled accordingly. Faint praise / comparative-against-worst patterns read by dominant signal — if the speaker is establishing the thing is still bad, the label is negative.
- **Spam**: messages must carry multiple BEC markers (changed bank details, urgency to bypass approval, requests for gift cards, claimed authority, secrecy) to be labeled spam. A single suspicious framing alone stays ham.

## How comparability works

- `leaderboard.json` carries `dataset_sha` — a hash of `data.json`. Different SHAs are not comparable.
- `hardware` (chip, memory) is auto-detected and embedded. Cross-hardware comparisons should consider memory bandwidth.
- `runs/` keeps full per-prediction audit trail locally; `archive/` keeps timestamped leaderboard snapshots. Both are gitignored.

## Layout

```
mlx-llm-bench/
├── data.json              labeled examples (editable)
├── models.json            model registry (editable)
├── bench                  shell entrypoint
├── leaderboard.json       canonical snapshot (committed)
├── leaderboard.csv        flat snapshot (committed)
├── runs/                  per-run audit trail (gitignored)
├── archive/               historical leaderboards (gitignored)
└── src/mlx_llm_bench/     CLI + runner
```

## License

MIT — see `LICENSE`. The benchmark and dataset are MIT-licensed; individual model weights retain their own licenses (e.g. Gemma 3 uses Google's Gemma Terms; Gemma 4, Qwen, and Llama 3.2 are Apache 2.0 or similar permissive).

## Author

Aman — [amankumar.ai](https://amankumar.ai) · [@onlyoneaman](https://x.com/onlyoneaman)
