"""
Evaluate a local HF model (base or LoRA-merged) on Spider-dev execution accuracy.

Example (Colab, A100/L4):
  python src/eval_local.py --model Qwen/Qwen3-1.7B --spider data/spider_data \
      --out results/base_qwen3-1.7b --batch-size 16

Notes
- Qwen3 chat template supports enable_thinking; we disable it (SQL doesn't need
  a reasoning trace and it would inflate tokens/latency). Other models ignore it.
- Greedy decoding, max_new_tokens=256. SQL is short; keep it short.
- --adapter loads a PEFT LoRA adapter on top of --model (used after SFT).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from spider_eval import bootstrap_ci, evaluate, load_split  # noqa: E402


def make_generate(model_name: str, adapter: str | None, max_new_tokens: int, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()

    def _apply(msgs):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return tok.apply_chat_template(msgs, **kwargs)

    @torch.no_grad()
    def generate(batch_msgs):
        prompts = [_apply(m) for m in batch_msgs]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.time()
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        dt = time.time() - t0
        in_len = enc["input_ids"].shape[1]
        results = []
        for i in range(len(prompts)):
            gen_ids = out[i, in_len:]
            # strip padding/eos
            gen_ids = gen_ids[gen_ids != tok.pad_token_id]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(
                {
                    "text": text,
                    "prompt_tokens": int(enc["attention_mask"][i].sum().item()),
                    "completion_tokens": int(len(gen_ids)),
                    "latency_s": dt / len(prompts),  # amortized per example
                }
            )
        return results

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (optional)")
    ap.add_argument("--spider", default="data/spider_data")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/base_qwen3-1.7b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    spider_root = Path(args.spider)
    examples = load_split(spider_root, args.split)
    if args.limit:
        examples = examples[: args.limit]

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    generate = make_generate(args.model, args.adapter, args.max_new_tokens, args.dtype)
    metrics = evaluate(
        examples,
        spider_root,
        generate,
        batch_size=args.batch_size,
        out_jsonl=out_prefix.with_suffix(".predictions.jsonl"),
    )

    flags = [json.loads(l)["correct"] for l in out_prefix.with_suffix(".predictions.jsonl").open()]
    lo, hi = bootstrap_ci(flags)
    metrics.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "split": args.split,
            "exec_acc_ci95": [lo, hi],
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        }
    )
    out_prefix.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
