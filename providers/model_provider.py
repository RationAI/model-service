def mlflow(artifact_uri: str) -> str:
    import mlflow.artifacts

    return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)


def huggingface(repo_id: str, filename: str | None = None) -> str:
    import os

    from huggingface_hub import hf_hub_download, snapshot_download

    # If HF_TOKEN is not set, we assume we're running in an offline environment and only allow loading from the local cache.
    offline = os.environ.get("HF_TOKEN") is None

    hf_home = os.environ.get("HF_HOME", "/mnt/huggingface_cache")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home

    if filename:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=offline,
        )
    else:
        return snapshot_download(
            repo_id=repo_id,
            local_files_only=offline,
        )
