"""
Evaluate an OpenAI-compatible API model on Spider-dev execution accuracy,
and record cost per 1K queries from the provider's token prices.

Works with OpenAI, OpenRouter, Together, etc. — anything with /v1/chat/completions.

Example:
  export OPENAI_API_KEY=...            # or OPENROUTER_API_KEY with --base-url
  python src/eval_api.py --model gpt-4.1-mini --out results/api_gpt-4.1-mini \
      --price-in 0.40 --price-out 1.60     # USD per 1M tokens, from the provider's pricing page

Prices are inputs you look up on the day; they are written into metrics.json
so the README cost table is reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spider_eval import bootstrap_ci, evaluate, load_split  # noqa: E402


def make_generate(model: str, base_url: str | None, api_key: str, concurrency: int, max_tokens: int):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def one(msgs):
        t0 = time.time()
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0, max_tokens=max_tokens
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    return {"text": f"API_ERROR {e}", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": time.time() - t0}
                time.sleep(2 ** attempt)
        u = r.usage
        return {
            "text": r.choices[0].message.content or "",
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "latency_s": time.time() - t0,
        }

    pool = ThreadPoolExecutor(max_workers=concurrency)

    def generate(batch_msgs):
        return list(pool.map(one, batch_msgs))

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--spider", default="data/spider_data")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--price-in", type=float, required=True, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, required=True, help="USD per 1M output tokens")
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("No API key: set OPENAI_API_KEY (or OPENROUTER_API_KEY) or pass --api-key")

    spider_root = Path(args.spider)
    examples = load_split(spider_root, args.split)
    if args.limit:
        examples = examples[: args.limit]
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    generate = make_generate(args.model, args.base_url, args.api_key, args.concurrency, args.max_tokens)
    metrics = evaluate(
        examples, spider_root, generate, batch_size=args.concurrency,
        out_jsonl=out_prefix.with_suffix(".predictions.jsonl"),
    )
    flags = [json.loads(l)["correct"] for l in out_prefix.with_suffix(".predictions.jsonl").open()]
    lo, hi = bootstrap_ci(flags)

    n = metrics["n"]
    cost_total = (metrics["prompt_tokens_total"] * args.price_in + metrics["completion_tokens_total"] * args.price_out) / 1e6
    metrics.update(
        {
            "model": args.model,
            "base_url": args.base_url,
            "split": args.split,
            "exec_acc_ci95": [lo, hi],
            "price_in_per_1m": args.price_in,
            "price_out_per_1m": args.price_out,
            "cost_total_usd": cost_total,
            "cost_per_1k_queries_usd": cost_total / n * 1000 if n else 0.0,
        }
    )
    out_prefix.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
