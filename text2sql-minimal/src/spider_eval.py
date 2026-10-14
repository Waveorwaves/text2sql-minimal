"""
Execution-accuracy evaluation harness for Spider-style text-to-SQL.

Shared by eval_local.py (HF transformers) and eval_api.py (OpenAI-compatible API).

Metric: execution accuracy (single-DB exec match).
  pred is correct iff running it on the example's SQLite DB returns the same
  multiset of rows as the gold query. Row order is enforced only when the gold
  query contains ORDER BY. This is the common "exec acc" definition; it is NOT
  the Spider test-suite metric (which uses distilled DB variants) — see README.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable
from itertools import permutations
# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class Example:
    idx: int
    db_id: str
    question: str
    gold_sql: str


def load_split(spider_root: Path, split: str = "dev") -> list[Example]:
    fname = {"dev": "dev.json", "train": "train_spider.json"}[split]
    rows = json.loads((spider_root / fname).read_text())
    return [
        Example(idx=i, db_id=r["db_id"], question=r["question"], gold_sql=r["query"])
        for i, r in enumerate(rows)
    ]


def db_path(spider_root: Path, db_id: str) -> Path:
    return spider_root / "database" / db_id / f"{db_id}.sqlite"


# --------------------------------------------------------------------------- #
# Schema -> prompt
# --------------------------------------------------------------------------- #

_schema_cache: dict[str, str] = {}


def load_schema_ddl(path: Path) -> str:
    """CREATE TABLE statements straight from sqlite_master. Cached per DB."""
    key = str(path)
    if key in _schema_cache:
        return _schema_cache[key]
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    ddl = "\n\n".join(re.sub(r"\s+", " ", r[0]).strip() for r in rows)
    _schema_cache[key] = ddl
    return ddl


SYSTEM_PROMPT = (
    "You are a text-to-SQL system. Given a SQLite schema and a question, "
    "write one SQLite query that answers the question. "
    "Output only the SQL inside a ```sql code block. No explanation."
)


def build_messages(schema_ddl: str, question: str) -> list[dict]:
    user = f"### Schema\n{schema_ddl}\n\n### Question\n{question}\n\n### SQL"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Model output -> SQL
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"(?is)\b(select|with)\b.*")


def extract_sql(text: str) -> str:
    text = _THINK_RE.sub("", text)
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    else:
        m = _SELECT_RE.search(text)
        text = m.group(0) if m else text
    text = text.strip().rstrip(";").strip()
    # keep only the first statement
    text = text.split(";")[0].strip()
    return text


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def execute_sql(path: Path, sql: str, timeout_s: float = 30.0):
    """Returns (rows, error). rows is a list of tuples or None on error."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    try:
        timer.start()
        cur = con.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    finally:
        timer.cancel()
        con.close()


def _norm_cell(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def rows_match(gold_rows, pred_rows, order_matters: bool, max_perm_cols: int = 6) -> bool:
    g = [tuple(_norm_cell(c) for c in r) for r in gold_rows]
    p = [tuple(_norm_cell(c) for c in r) for r in pred_rows]
    if len(g) != len(p):
        return False
    if not g:
        return True
    ncol = len(g[0])
    if ncol != len(p[0]):
        return False
    eq = (lambda a, b: a == b) if order_matters else (lambda a, b: Counter(a) == Counter(b))
    if eq(g, p):
        return True
    if ncol > max_perm_cols:
        return False
    for perm in permutations(range(ncol)):
        if perm != tuple(range(ncol)) and eq(g, [tuple(r[i] for i in perm) for r in p]):
            return True
    return False


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #


@dataclass
class Prediction:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    raw_output: str
    pred_sql: str
    correct: bool
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


GenerateFn = Callable[[list[list[dict]]], list[dict]]
"""Takes a batch of chat-message lists; returns a list of dicts with keys
   text, prompt_tokens, completion_tokens, latency_s (per example)."""


def evaluate(
    examples: Iterable[Example],
    spider_root: Path,
    generate: GenerateFn,
    batch_size: int = 8,
    out_jsonl: Path | None = None,
    log_every: int = 50,
) -> dict:
    examples = list(examples)
    preds: list[Prediction] = []
    t0 = time.time()
    fout = out_jsonl.open("w") if out_jsonl else None

    for b in range(0, len(examples), batch_size):
        batch = examples[b : b + batch_size]
        msgs = [
            build_messages(load_schema_ddl(db_path(spider_root, ex.db_id)), ex.question)
            for ex in batch
        ]
        outs = generate(msgs)
        for ex, out in zip(batch, outs):
            pred_sql = extract_sql(out["text"])
            path = db_path(spider_root, ex.db_id)
            gold_rows, gold_err = execute_sql(path, ex.gold_sql)
            if gold_err:
                # gold must execute; if not, the example is skipped from the denominator
                correct, err = False, f"GOLD_ERROR {gold_err}"
            else:
                pred_rows, err = execute_sql(path, pred_sql)
                order_matters = "order by" in ex.gold_sql.lower()
                correct = err is None and rows_match(gold_rows, pred_rows, order_matters)
            p = Prediction(
                idx=ex.idx,
                db_id=ex.db_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
                raw_output=out["text"],
                pred_sql=pred_sql,
                correct=bool(correct),
                error=err,
                prompt_tokens=int(out.get("prompt_tokens", 0)),
                completion_tokens=int(out.get("completion_tokens", 0)),
                latency_s=float(out.get("latency_s", 0.0)),
            )
            preds.append(p)
            if fout:
                fout.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        done = min(b + batch_size, len(examples))
        if done % log_every < batch_size or done == len(examples):
            acc = sum(p.correct for p in preds) / len(preds)
            print(f"[{done}/{len(examples)}] exec_acc={acc:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

    if fout:
        fout.close()
    return summarize(preds, wall_s=time.time() - t0)


def summarize(preds: list[Prediction], wall_s: float) -> dict:
    n = len(preds)
    valid = [p for p in preds if not (p.error or "").startswith("GOLD_ERROR")]
    n_valid = len(valid)
    correct = sum(p.correct for p in valid)
    exec_errors = sum(1 for p in valid if p.error)
    pt = sum(p.prompt_tokens for p in preds)
    ct = sum(p.completion_tokens for p in preds)
    return {
        "n": n,
        "n_valid": n_valid,
        "exec_acc": correct / n_valid if n_valid else 0.0,
        "n_correct": correct,
        "n_exec_error": exec_errors,
        "exec_error_rate": exec_errors / n_valid if n_valid else 0.0,
        "prompt_tokens_total": pt,
        "completion_tokens_total": ct,
        "prompt_tokens_per_ex": pt / n if n else 0,
        "completion_tokens_per_ex": ct / n if n else 0,
        "mean_latency_s": sum(p.latency_s for p in preds) / n if n else 0,
        "wall_s": wall_s,
    }


def bootstrap_ci(correct_flags: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI on accuracy. Use for reporting deltas honestly."""
    import random

    rng = random.Random(seed)
    n = len(correct_flags)
    accs = []
    for _ in range(n_boot):
        s = sum(correct_flags[rng.randrange(n)] for _ in range(n))
        accs.append(s / n)
    accs.sort()
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]
