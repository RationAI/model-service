import os

from huggingface_hub import login, snapshot_download


HF_TOKEN = os.environ.get("HF_TOKEN")
CACHE_DIR = "/mnt/huggingface_cache"
MODEL_ID = "paige-ai/Virchow2"

os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Starting download for {MODEL_ID} to {CACHE_DIR}")

if HF_TOKEN:
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
else:
    print("No HF_TOKEN provided! Download might fail for gated models.")

print("Downloading model snapshot...")
try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        resume_download=True,
        local_files_only=False,
    )
    print(f"Model downloaded to: {path}")

    print("Verifying model files exist...")
    import timm

    try:
        model = timm.create_model(
            f"hf-hub:{MODEL_ID}",
            pretrained=True,
            num_classes=0,
        )
        print(f"Model successfully loaded! Type: {type(model).__name__}")
        del model  # Free memory
    except Exception as e:
        print(f"Verification warning: {e}")

except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

print("DONE. Model is cached and ready for offline use.")
