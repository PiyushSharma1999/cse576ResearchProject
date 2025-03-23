from huggingface_hub import snapshot_download

# Download the model to a specific directory
local_dir = snapshot_download(
    repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    cache_dir="./CapybaraHermes-2.5-Mistral-7B-GPTQ"  # Specify the directory here
)

print(f"Model downloaded to: {local_dir}")