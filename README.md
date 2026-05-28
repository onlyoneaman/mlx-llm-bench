# mlx-llm-bench

Local LLM classification benchmark for Apple Silicon Macs with limited unified memory.

Built around the **16GB Mac mini class** — focuses on candidates that actually fit and stay fast on consumer hardware. Runs entirely offline once weights are cached.

📊 **Dataset & leaderboard on HuggingFace:** [huggingface.co/datasets/onlyoneaman/mlx-llm-bench](https://huggingface.co/datasets/onlyoneaman/mlx-llm-bench)

## What it measures

Four tasks with balanced easy + hard examples (n=125 total):

- **Sentiment** — positive / negative (21 easy + 12 hard, incl. sarcasm, litotes, faint praise)
- **Topic** — world / sports / business / tech (24 easy + 12 hard, incl. tech-keyword pressure)
- **Spam** — spam / ham (20 easy + 11 hard, incl. sophisticated BEC scams)
- **IFEval** — instruction-following (13 easy + 12 hard, incl. exact word count, letter exclusion, format constraints)

Classification uses JSON-constrained output (model replies `{"label": "x"}`). IFEval uses raw prompts with python-regex validators per item — pass iff ALL of an item's validators pass.

Each model gets the same prompts at `temp=0`. `format_ok` is tracked separately from `correct` so format-following ability is a first-class metric.

The easy / hard split is the key signal — strong models often max out easy and diverge on hard.

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
| `bench export` | Write `leaderboard.{json,csv}` for committing |
| `bench rescore [--sha S]` | Re-apply scoring to saved results in place (no re-runs needed) |
| `bench inspect <run-id> [--task T] [--i N]` | Show raw model outputs for inspection (misses by default) |
| `bench serve <model>` | Start OpenAI-compatible HTTP server |

## Current leaderboard (Mac mini M4 16GB · n=125 · JSON format)

Canonical source: [`leaderboard.csv`](./leaderboard.csv) / [`leaderboard.json`](./leaderboard.json). Snapshot below regenerated from `bench export` against `dataset_sha 2c34076d55d3` (content-based — invariant to JSON formatting). **n=125 = 100 classification examples (sentiment + topic + spam) + 25 IFEval instruction-following examples.**

| Rank | Model | Acc (95% CI) | Easy | Hard | fmt_ok | Avg time | Size |
|---|---|---|---|---|---|---|---|
| 1 | **gemma3-12b-qat** | **92.8%** [87–96] | 97% | 85% | 100% | 2.24s | 8.0GB |
| 2 | **llama-3.2-3b** | **92.0%** [86–96] | 97% | 83% | 100% | **0.74s** | **1.8GB** |
| 3 | gemma4-e4b | 90.4% [84–94] | 99% | 77% | 100% | 0.82s | 5.2GB |
| 3 | ministral-3-8b | 90.4% [84–94] | 97% | 79% | 100% | 4.21s | 5.6GB |
| 5 | llama-3.1-8b | 89.6% [83–94] | 97% | 77% | 100% | 1.63s | 4.2GB |
| 5 | qwen3-8b | 89.6% [83–94] | 97% | 77% | 100% | 1.25s | 5.0GB |
| 7 | gemma3-4b-qat | 85.6% [78–91] | 95% | 70% | 99% | 0.89s | 2.6GB |
| 8 | phi4-mini-instruct | 84.8% [78–90] | 94% | 70% | 100% | 0.96s | 2.2GB |
| 8 | qwen2.5-coder-7b | 84.8% [78–90] | 96% | 66% | 100% | 1.26s | 4.7GB |
| 10 | mistral-nemo-minitron-8b | 80.0% [72–86] | 90% | 64% | 100% | 12.18s | 4.4GB |
| 11 | smollm3-3b | 79.2% [71–85] | 88% | 64% | 98% | 0.86s | 1.8GB |
| 12 | nemotron-nano-9b | 78.4% [70–85] | 86% | 66% | 96% | 10.13s | 5.0GB |
| 13 | hermes-3-llama-3.2-3b | 74.4% [66–81] | 87% | 53% | **42%** ⚠️ | 0.75s | 1.7GB |
| 14 | deepseek-r1-distill-7b | 59.2% [50–67] | 64% | 51% | **50%** | 10.06s | 4.5GB |
| 15 | phi4-mini-reasoning | 48.0% [39–57] | 53% | 40% | **42%** | 6.07s | 2.2GB |

## Which model should I run?

For a **16 GB Mac mini class** machine, picking from this bench's evidence:

All exact numbers are in `leaderboard.csv`; this section is qualitative so it doesn't drift every time the bench reruns.

- 🎯 **Default daily driver: `llama-3.2-3b`.** Smallest model that lands near the top tier on both lenient and strict accuracy, at the fastest per-example time in the top tier. Validated against Llama 3.1 8B — newer & smaller wins on both axes.
- 🧠 **Maximum accuracy: `gemma3-12b-qat`.** Top of the leaderboard but slowest non-reasoning model and tight on 16 GB. Close other apps before running.
- 👁️ **Multimodal: `gemma4-e4b`.** Vision + native audio. Only fully-multimodal model in the registry.
- 🔁 **Strict instruction-following: `llama-3.2-3b` or `phi4-mini-instruct`.** Both hit ~100% format compliance — `strict_acc ≈ acc` on the leaderboard. Avoid alignment finetunes for this; see Hermes-3 finding below.
- ❌ **Don't use for classification**:
  - `phi4-mini-reasoning`, `deepseek-r1-distill-7b` — reasoning-tuned models emit `<think>` blocks instead of answers. Look for the gap between `acc` and `strict_acc` in `leaderboard.csv`.
  - `hermes-3-llama-3.2-3b` — Hermes-3 RLHF broke the base Llama 3.2 3B's structured-output ability. Compare `strict_acc` vs `acc` to see how much credit is from free-form fallback parsing.
  - `mistral-nemo-minitron-8b`, `nemotron-nano-9b` — NVIDIA distillations are 10–20× slower than Llama 3.2 3B with no accuracy gain.

> Caveat on significance: at n=125, 95% Wilson CIs are ±7 pp around accuracies in the 85-95% range. Sub-3-point gaps are likely noise. Use `./bench compare <id1> <id2>` to get the paired McNemar p-value for a real significance test. Headline `acc` is *lenient* — it counts free-form responses that recovered a valid label even when JSON format compliance failed. `fmt_ok` shows the fraction of responses that followed the requested format; Hermes-3 at 42% fmt_ok and 74% acc means a third of its credit comes from free-form fallback parsing.

## Annotation rubric

For ambiguous "hard" examples, these tiebreakers were applied — published so anyone reading the dataset (or any LLM evaluated on it) can see how reasonable disagreements were resolved.

- **Topic**: stories featuring a corporate transaction, earnings, market cap, or M&A are labeled **business** even when the subject is a tech company. ("Apple acquired AI startup for $200M" → business; "Apple's M5 chip launch" → tech.)
- **Sentiment**: litotes ("not bad", "can't say I hated it") read as the *opposite* of the negated adjective and are labeled accordingly. Faint praise / comparative-against-worst patterns read by dominant signal — if the speaker is establishing the thing is still bad, the label is negative.
- **Spam**: messages must carry multiple BEC markers (changed bank details, urgency to bypass approval, requests for gift cards, claimed authority, secrecy) to be labeled spam. A single suspicious framing alone stays ham.

## How comparability works

- `leaderboard.json` carries `dataset_sha` — a content-based hash of `data.json` plus the IFEval validators sidecar (`ifeval_validators.json`). Invariant to JSON formatting; only label/text/validator changes shift it. Different SHAs are not comparable.
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
