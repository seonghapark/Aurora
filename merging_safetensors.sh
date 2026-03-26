#!/usr/bin/env bash
set -euo pipefail

# Download sharded safetensors from Hugging Face (resume-safe) and merge into one file.
#
# Usage:
#   ./merging_safetensors.sh <repo_id> <model_dir> <out_dir>
#
# Example:
#   HF_HUB_DISABLE_XET=1 \
#   ./merging_safetensors.sh \
#     MaziyarPanahi/calme-3.2-instruct-78b \
#     /tmp/calme-3.2-instruct-78b \
#     /tmp/calme-3.2-instruct-78b-merged
#
# Optional environment variables:
#   PYTHON_BIN                  Python executable (default: python3)
#   HF_TOKEN                    Hugging Face token for gated/private repos
#   HF_HUB_DISABLE_XET          Set 1 to disable Xet backend (default in this script: 1)
#   FORCE_REDOWNLOAD            1 to force redownload of all files (default: 0)
#   INCLUDE_TOKENIZER           1 to download tokenizer files too (default: 1)
#   DELETE_SHARDS_BEFORE_SAVE   1 to delete shard files after loading, before saving merged file (default: 0)
#   MAX_SHARD_SIZE              Max shard size for output save_pretrained (default: 1000GB)
#   TRUST_REMOTE_CODE           1 or 0 for transformers trust_remote_code (default: 1)

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repo_id> <model_dir> <out_dir>" >&2
  exit 1
fi

REPO_ID="$1"
MODEL_DIR="$2"
OUT_DIR="$3"

export REPO_ID_ARG="${REPO_ID}"
export MODEL_DIR_ARG="${MODEL_DIR}"
export OUT_DIR_ARG="${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"
INCLUDE_TOKENIZER="${INCLUDE_TOKENIZER:-1}"
DELETE_SHARDS_BEFORE_SAVE="${DELETE_SHARDS_BEFORE_SAVE:-0}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-1000GB}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# Disable Xet by default to avoid CAS/Xet quota issues on some systems.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${MODEL_DIR}"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("[ERROR] Missing dependency: huggingface_hub", file=sys.stderr)
    print("Install with: pip install huggingface_hub", file=sys.stderr)
    raise

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    print("[ERROR] Missing dependency: transformers", file=sys.stderr)
    print("Install with: pip install transformers", file=sys.stderr)
    raise

repo_id = os.environ.get("REPO_ID_ARG")
model_dir = Path(os.environ.get("MODEL_DIR_ARG", "")).resolve()
out_dir = Path(os.environ.get("OUT_DIR_ARG", "")).resolve()
force_redownload = os.environ.get("FORCE_REDOWNLOAD", "0") == "1"
include_tokenizer = os.environ.get("INCLUDE_TOKENIZER", "1") == "1"
delete_shards_before_save = os.environ.get("DELETE_SHARDS_BEFORE_SAVE", "0") == "1"
max_shard_size = os.environ.get("MAX_SHARD_SIZE", "1000GB")
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
hf_token = os.environ.get("HF_TOKEN")

if not repo_id:
    raise ValueError("REPO_ID_ARG is required")
if not model_dir:
    raise ValueError("MODEL_DIR_ARG is required")
if not out_dir:
    raise ValueError("OUT_DIR_ARG is required")

model_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

def download_if_needed(filename: str) -> Path:
    target = model_dir / filename
    if force_redownload:
        need = True
    else:
        # LFS pointer files are tiny. Real safetensors shards are very large.
        if target.exists() and target.stat().st_size > 1024 * 1024:
            need = False
        else:
            need = True

    if need:
        print(f"[DOWNLOAD] {filename}", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            force_download=force_redownload,
            token=hf_token,
        )
    else:
        print(f"[SKIP] {filename} already present", flush=True)

    return target

# 1) Download metadata first
config_path = download_if_needed("config.json")
index_path = download_if_needed("model.safetensors.index.json")

# Optional generation config
try:
    download_if_needed("generation_config.json")
except Exception:
    pass

# 2) Parse index and download all shard files
with index_path.open("r", encoding="utf-8") as f:
    index_data = json.load(f)

weight_map = index_data.get("weight_map", {})
if not weight_map:
    raise RuntimeError("model.safetensors.index.json has no weight_map")

shard_files = sorted(set(weight_map.values()))
print(f"[INFO] Found {len(shard_files)} shard files in index", flush=True)

for shard in shard_files:
    download_if_needed(shard)

# 3) Optionally fetch tokenizer assets
tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]
if include_tokenizer:
    for tf in tokenizer_files:
        try:
            download_if_needed(tf)
        except Exception:
            # Not every tokenizer has every file type.
            print(f"[WARN] Tokenizer file not available: {tf}", flush=True)

# 4) Load full model from shards
print("[INFO] Loading model from shards...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=trust_remote_code,
)

# 5) Optional cleanup of shard files before save to reduce peak disk pressure
if delete_shards_before_save:
    print("[INFO] Deleting shard files before merged save...", flush=True)
    for shard in shard_files:
        shard_path = model_dir / shard
        if shard_path.exists():
            shard_path.unlink()
    if index_path.exists():
        index_path.unlink()

# 6) Save as a single safetensors file
print("[INFO] Saving merged model...", flush=True)
model.save_pretrained(
    str(out_dir),
    safe_serialization=True,
    max_shard_size=max_shard_size,
)

# 7) Copy tokenizer assets to output dir if present
if include_tokenizer:
    for tf in tokenizer_files + ["generation_config.json"]:
        src = model_dir / tf
        dst = out_dir / tf
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

merged_file = out_dir / "model.safetensors"
if not merged_file.exists():
    raise RuntimeError(f"Merged file not found: {merged_file}")

size_gb = merged_file.stat().st_size / (1024 ** 3)
print(f"[SUCCESS] Merged file: {merged_file}", flush=True)
print(f"[SUCCESS] Merged size: {size_gb:.2f} GB", flush=True)
PY
