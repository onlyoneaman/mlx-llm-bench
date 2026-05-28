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

## Current leaderboard (Mac mini M4 16GB)

Canonical source: [`leaderboard.csv`](./leaderboard.csv) / [`leaderboard.json`](./leaderboard.json). Snapshot below is regenerated from `bench export` against `dataset_sha cb96152deb7f`.

| Model | Overall | Easy | Hard | Avg time | Size |
|---|---|---|---|---|---|
| **gemma3-12b-qat** | **92.6%** | 98% | 83% | 0.83s | 8.0GB |
| gemma4-e4b | 89.7% | 98% | 74% | 0.28s | 5.2GB |
| gemma3-4b-qat | 89.7% | 98% | 74% | 0.35s | 2.6GB |
| qwen3-8b | 86.8% | 98% | 65% | 0.47s | 5.0GB |
| qwen2.5-coder-7b | 82.4% | 98% | 52% | 0.62s | 4.7GB |
| llama-3.2-3b | 77.9% | 91% | 52% | 0.29s | 1.8GB |
| smollm3-3b | 66.2% | 78% | 44% | 1.45s | 1.8GB |
| phi4-mini-reasoning | 58.8% | 67% | 44% | 5.94s | 2.2GB |
| deepseek-r1-distill-7b | 42.6% | 42% | 44% | 10.01s | 4.5GB |

**Three Pareto winners** for the 16 GB target:
- **Most accurate**: `gemma3-12b-qat` (92.6%, tight on 16 GB)
- **Fastest at top tier**: `gemma4-e4b` (0.28s/ex, multimodal)
- **Smallest at top tier**: `gemma3-4b-qat` (2.6 GB on disk)

> Caveat on statistical significance: at n=68, the 95% Wilson interval on a single accuracy is roughly ±11 pp. Treat sub-3-point gaps as noise until the dataset grows or paired tests are added.

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
