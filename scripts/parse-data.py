from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="AI4EPS/quakeflow_das",
    filename="ridgecrest/20190705000000.h5",  # testing, this one real
    repo_type="dataset"
)

print("Downloaded to:", file_path)
