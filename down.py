from huggingface_hub import snapshot_download

# Replace with your model repo ID (e.g., "bert-base-uncased", "runwayml/stable-diffusion-v1-5")
model_id = "meta-llama/Meta-Llama-3-8B"

snapshot_download(
    repo_id=model_id,
    local_dir="./Meta-Llama-3-8B",
    local_dir_use_symlinks=False,
    cache_dir="./hf_cache"  # force caching to local writable folder
)