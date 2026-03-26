from huggingface_hub import hf_hub_download

repo_id = "MaziyarPanahi/calme-3.2-instruct-78b"
local_dir = "/tmp/calme-3.2-instruct-78b-merged"
files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]

for f in files:
    p = hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
    print(f"downloaded: {f} -> {p}")
