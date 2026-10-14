"""Re-score a predictions.jsonl with the current rows_match. No GPU needed."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from spider_eval import execute_sql, rows_match, db_path, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--spider", default="data/spider_data")
a = ap.parse_args()
root = Path(a.spider); rows = [json.loads(l) for l in open(a.pred)]
flags, n_err, n_valid = [], 0, 0
for r in rows:
    path = db_path(root, r["db_id"])
    g, ge = execute_sql(path, r["gold_sql"])
    if ge: continue
    n_valid += 1
    p, pe = execute_sql(path, r["pred_sql"])
    ok = pe is None and rows_match(g, p, "order by" in r["gold_sql"].lower())
    n_err += pe is not None; flags.append(ok)
acc = sum(flags) / n_valid
print(json.dumps({"n_valid": n_valid, "exec_acc": acc, "exec_acc_ci95": bootstrap_ci(flags),
                  "n_exec_error": n_err, "pred_file": a.pred}, indent=2))