import os
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

repo_id = "MaziyarPanahi/calme-3.2-instruct-78b"
model_dir = os.environ.get("MODEL_DIR", "/home/seonghapark/llm_evaluation/calme-3.2-instruct-78b")
out_dir = os.environ.get("OUT_DIR", "/home/seonghapark/llm_evaluation/calme-3.2-instruct-78b-merged")
delete_shards_before_save = os.environ.get("DELETE_SHARDS_BEFORE_SAVE", "0") == "1"

required_files = [
    "config.json",
    "model.safetensors.index.json",
] + [f"model-{i:05d}-of-00067.safetensors" for i in range(1, 68)]

# Download any missing or pointer-only shard files
for fname in required_files:
    fpath = os.path.join(model_dir, fname)
    need_download = True
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        # LFS pointer files are tiny; real shards are multi-GB
        if size > 1024 * 1024:
            need_download = False
    if need_download:
        print(f"Downloading {fname} ...", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

print("All shards present. Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True,
)

if delete_shards_before_save:
    print("Deleting shard files to reduce peak disk usage before save...", flush=True)
    for fname in required_files:
        if fname.startswith("model-") and fname.endswith(".safetensors"):
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        os.remove(index_path)

os.makedirs(out_dir, exist_ok=True)
print("Saving as single safetensors file...", flush=True)
model.save_pretrained(
    out_dir,
    safe_serialization=True,
    max_shard_size="1000GB",
)
print(f"Done. Merged model saved in: {out_dir}", flush=True)
