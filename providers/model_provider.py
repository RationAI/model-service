def mlflow(artifact_uri: str) -> str:
    import os

    import mlflow.artifacts

    # Use shared cache directory if available
    cache_dir = os.environ.get("MLFLOW_CACHE_DIR", "/mnt/mlflow_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Set MLflow download cache location
    os.environ["MLFLOW_ARTIFACT_DOWNLOAD_OUTPUT"] = cache_dir

    return mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path=cache_dir
    )


def huggingface(repo_id: str, filename: str | None = None) -> str:
    import os

    from huggingface_hub import hf_hub_download, snapshot_download

    # Use dedicated HuggingFace cache directory
    cache_dir = os.environ.get("HF_HOME", "/mnt/huggingface_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

    if filename:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    else:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
