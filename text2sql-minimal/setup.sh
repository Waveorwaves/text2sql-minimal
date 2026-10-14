#!/usr/bin/env bash
# One-shot environment setup. Works on Colab and on a rented pod. Re-run freely.
set -euo pipefail
pip install -q -U "transformers>=4.51" accelerate peft huggingface_hub openai
python - <<'PY'
from huggingface_hub import hf_hub_download
import zipfile, pathlib
p = hf_hub_download("HAL-9001/spider-databases", "spider_data.zip", repo_type="dataset", local_dir="data")
root = pathlib.Path("data/spider_data")
if not root.exists():
    with zipfile.ZipFile(p) as z: z.extractall("data")
print("spider at", root, "| dbs:", len(list((root/"database").iterdir())))
PY
