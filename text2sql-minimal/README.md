# text2sql-minimal

Small models for text-to-SQL: how close does a fine-tuned 1.7B model get to a frontier LLM on execution accuracy, and at what cost?

## Thesis

<!-- Rewrite in your own words. Keep it to one paragraph. -->
Harness Bench showed that agent cost is driven by context re-sending, not model choice. TAXUE's agenda is to strip everything that doesn't improve the answer — fewer tokens, smaller models, faster serving, cheaper judges — and prove quality holds on public benchmarks. This repo is the "smaller models" chapter: SFT, then GRPO, on Qwen3-1.7B for text-to-SQL, evaluated by executing the SQL.

## Results

Execution accuracy on Spider-dev (1,034 questions). Greedy decoding, schema given as `CREATE TABLE` DDL, thinking disabled. 95% CIs are percentile bootstrap over questions.

| Model | Params | Train | Exec acc | 95% CI | Exec err rate | Tokens / query (in / out) | Latency / query | Cost / 1K queries |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | base | | | | | | GPU-hrs: |
| Qwen3-1.7B | 1.7B | base | | | | | | |
| Qwen3-1.7B + LoRA SFT | 1.7B | Spider train | | | | | | |
| Qwen3-1.7B + SFT + GRPO | 1.7B | Spider train | | | | | | |
| *frontier model* | — | zero-shot | | | | | | $ |

Cost basis: API models use the provider's list price on the run date (recorded in `metrics.json`). Local models use GPU rental rate × wall time; rate and date recorded alongside.

### What the numbers mean
<!-- 3–5 sentences, written after the table is filled. Lead with the delta that matters. -->

## Metric

**Execution accuracy (single-DB exec match):** a prediction is correct iff executing it on the question's SQLite database returns the same multiset of rows as the gold query. Row order is enforced only when the gold query contains `ORDER BY`. Predictions that error are wrong. Questions whose gold query fails to execute are excluded from the denominator (count reported as `n - n_valid`).

This is the common "exec acc" definition, **not** the Spider test-suite metric (Zhong et al. 2020), which runs each query against distilled database variants to catch false positives. Test-suite numbers are typically a few points lower. Reported numbers here are comparable with each other, and approximately with published exec-acc figures; they are not directly comparable with test-suite leaderboard entries.

## Setup

```bash
bash setup.sh                      # installs deps, downloads Spider DBs (206 MB) to data/spider_data
python src/eval_local.py --model Qwen/Qwen3-1.7B --out results/base_qwen3-1.7b
python src/eval_api.py --model <model> --out results/api_<model> --price-in <usd/1M> --price-out <usd/1M>
```

`notebooks/01_eval_base.ipynb` runs the same thing on Colab. Databases come from [HAL-9001/spider-databases](https://huggingface.co/datasets/HAL-9001/spider-databases) (a re-host of the Yale distribution; SHA256 pinned there).

Outputs: `results/<name>.metrics.json` (summary) and `results/<name>.predictions.jsonl` (every question, raw output, extracted SQL, correctness, error, tokens, latency).

## Design notes

<!-- Each entry: the decision, the alternative, why. Written for an interviewer who asks "why did you…". -->

- **Schema as raw DDL from `sqlite_master`** rather than a hand-formatted table list. Alternative: `tables.json` with column types and foreign keys. DDL is what the model saw most of in pretraining and needs zero parsing code; revisit if exec-error analysis shows join/FK mistakes dominate.
- **Thinking disabled** for Qwen3. SQL is short and the task is closer to translation than reasoning; thinking multiplies output tokens and latency. Ablation worth running if GRPO underperforms: thinking on vs. off.
- **Greedy decoding, `max_new_tokens=256`.** Deterministic, cheap, and SQL rarely exceeds 100 tokens. Sampling would only matter for self-consistency, which is out of scope.
- **Single-DB exec match, not test-suite.** See Metric. Chosen for simplicity and because GRPO's reward will use the same executor — the metric and the reward share one code path.
- **Spider-dev, not BIRD, for phase 1.** Spider DBs are 206 MB and the eval is fast; BIRD train DBs are tens of GB. Phase 1 is the SFT baseline; BIRD arrives with GRPO.

## Known limits

- Spider is old (2018) and likely in Qwen3's pretraining data. Base-model numbers may be inflated by contamination; the SFT/GRPO *deltas* are the honest part of this table.
- Exec match on the original DB can accept semantically wrong queries that happen to return the same rows on that data.
- Bootstrap CIs are over questions only; they don't capture seed variance in training.

## Can't-explain-yet

<!-- Every setting you used but could not explain to a senior MLE goes here, with the date. Remove it when you can. Empty list = not pushing hard enough. -->

- 

## Roadmap

1. Base numbers: Qwen3-0.6B, Qwen3-1.7B, one frontier model, with cost. ← **phase 1**
2. LoRA SFT on Spider train (7,000). Ablation: 0.6B vs 1.7B.
3. GRPO with execution reward. Ablation: SFT-only vs SFT→GRPO.
4. Transfer check on BIRD-dev.
5. Quantize + serve (AWQ/INT4, vLLM/SGLang): accuracy delta vs. throughput.

## Citation

Spider: Yu et al., 2018. Test-suite evaluation: Zhong et al., 2020.
