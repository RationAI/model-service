# Adding New Models

This guide explains how to integrate your own machine learning models into Model Service.

## Start Here: Reuse Existing Implementations

Before writing a model from scratch, start from the closest existing implementation:

- **Binary classification**: [`models/binary_classifier.py`](https://github.com/RationAI/model-service/blob/main/models/binary_classifier.py)
- **Semantic segmentation**: [`models/semantic_segmentation.py`](https://github.com/RationAI/model-service/blob/main/models/semantic_segmentation.py)
- **Virchow2 embedding/classification**: [`models/virchow2.py`](https://github.com/RationAI/model-service/blob/main/models/virchow2.py)
- **Heatmap pipeline**: [`builders/heatmap_builder.py`](https://github.com/RationAI/model-service/blob/main/builders/heatmap_builder.py)

Matching application definitions are in `helm/rayservice/applications/` and are usually the fastest way to bootstrap a new model route.

## Recommended Workflow (Step by Step)

1. **Choose a base implementation**
   Copy the closest model class and adapt only model-specific logic first.

2. **Export your model to ONNX**
   To achieve the best inference performance, export your model to ONNX format. It is **critical** to define dynamic axes for the batch dimension so the model can accept variable-sized batches from `@serve.batch`. Here is a complete example of loading a model checkpoint via MLflow and exporting it properly:

   ```python
   import mlflow
   import mlflow.artifacts
   import torch

   model = Model()
   path = mlflow.artifacts.download_artifacts(
       ".../checkpoint.ckpt"
   )

   model.load_state_dict(torch.load(path)["state_dict"])
   model.eval()

   torch.onnx.export(
       model,
       torch.randn(1, 3, 512, 512),  # model input (or a tuple for multiple inputs)
       "model.onnx",                 # where to save the model (can be a file or file-like object)
       export_params=True,           # store the trained parameter weights inside the model file
       do_constant_folding=True,     # whether to execute constant folding for optimization
       input_names=["input"],        # the model's input names
       output_names=["output"],      # the model's output names
       dynamic_axes={
           "input": {0: "batch_size"},  # variable length axes required for @serve.batch
           "output": {0: "batch_size"},
       },
   )
   ```

3. **Implement or adapt the Python entrypoint**
   Keep Ray Serve structure (`@serve.deployment`, `@serve.ingress`, optional `reconfigure`) and align request/response format with your target workload.

4. **Add/update application YAML in Helm**
   Add a file in `helm/rayservice/applications/` with your `import_path`, `route_prefix`, and `runtime_env.working_dir`.

5. **Deploy to a dedicated test release**
   Use a dedicated `<release-name>` for isolation during development.

6. **Validate and iterate**
   Test endpoint behavior, check RayService status, inspect worker/head logs, then tune autoscaling/resources.

For detailed deployment procedure, continue with [Deployment Guide](deployment-guide.md). For scaling/resource tuning, use [Configuration Reference](configuration-reference.md). For runtime failure diagnosis, use [Troubleshooting](troubleshooting.md).

## Model Implementation Reference

Instead of writing boilerplate from scratch, you should leverage the patterns already implemented in this repository.

### 1. The Deployment Contract

All models in the service follow the same basic Ray Serve contract:

- A class decorated with `@serve.deployment` (and optionally `@serve.ingress`).
- An `__init__` method for one-time setup (like loading the ONNX session).
- An optional `reconfigure` method for dynamic config updates.
- An HTTP handling method (like `@app_ingress.post("/")`).
- A bound application object at the end of the file (e.g., `app = MyModel.bind()`).

### 2. High-Performance Batching

_Reference: [`models/binary_classifier.py`](https://github.com/RationAI/model-service/blob/main/models/binary_classifier.py)_

For high-throughput workloads (like pathology imaging), the reference models use:

- **Micro-batching**: The `@serve.batch` decorator groups concurrent HTTP requests into a single NumPy/Tensor batch before passing them to the ONNX session.
- **LZ4 Compression**: To minimize network overhead, requests and responses are sent as raw bytes compressed with LZ4. Decompression is offloaded to a separate thread using `asyncio.to_thread`.
- **Header-driven metadata**: Custom HTTP headers (like `x-output-shape`) are used to pass metadata alongside the raw binary payloads.

### 3. MLflow Integration

_Reference: [`providers/model_provider.py`](https://github.com/RationAI/model-service/blob/main/providers/model_provider.py)_

You do not need to hardcode model paths or write custom download logic. The service includes an MLflow provider.

Pass the MLflow artifact URI through your application's `user_config` (in the Helm YAML):

```yaml
user_config:
  model:
    artifact_uri: mlflow-artifacts:/65/abc123.../model.onnx
```

Then in your model's `reconfigure` method, resolve it using the provider:

```python
from providers.model_provider import mlflow

def reconfigure(self, config):
    model_path = mlflow(artifact_uri=config["model"]["artifact_uri"])
    self.session = ort.InferenceSession(model_path)
```

Make sure the cluster has the `MLFLOW_TRACKING_URI` environment variable set in `runtime_env.env_vars`.

### 4. GPU Acceleration

For GPU-accelerated deployments, ensure your YAML requests `num_gpus: 1` (or a fraction thereof) in `ray_actor_options`.

When using ONNX Runtime, initialize the `TensorrtExecutionProvider` or `CUDAExecutionProvider` in your `__init__` or `reconfigure` method. Review the TensorRT optimization details in the [Deployment Guide](deployment-guide.md).

## Helm Application Configuration

Add your model file to `helm/rayservice/applications/my-model.yaml`:

```yaml
- name: my-model
  import_path: models.my_onnx_model:app
  route_prefix: /my-model
  runtime_env:
    working_dir: https://github.com/RationAI/model-service/archive/refs/heads/your-feature-branch.zip
    pip:
      - onnxruntime>=1.23.2
      - numpy
  deployments:
    - name: MyONNXModel
      autoscaling_config:
        min_replicas: 1
        max_replicas: 4
      ray_actor_options:
        num_cpus: 2
        memory: 4294967296 # 4 GiB
        runtime_env:
          pip:
            - onnxruntime>=1.23.2
```

In this repository, model dependencies can be installed under `deployments[*].ray_actor_options.runtime_env.pip` (not only at `runtime_env.pip` at application level). This is useful when different deployments need different dependencies.

Helm automatically renders all files from `helm/rayservice/applications/` into `serveConfigV2` via `helm/rayservice/templates/rayservice.yaml`.

### Best Practice: Test New Models from Your Own Branch

When adding a new model, create and use your own GitHub branch for testing. This avoids affecting deployments that still depend on `main`.

Example:

- Branch name: `feature/my-new-model`
- `working_dir`:

```yaml
runtime_env:
  working_dir: https://github.com/RationAI/model-service/archive/refs/heads/feature/my-new-model.zip
```

After validation, merge the branch and switch `working_dir` back to the target shared branch (for example `main`).

Before running Helm, commit and push your new model code and application YAML to your branch. Ray downloads code from the branch ZIP in `runtime_env.working_dir`, so unpushed local changes will not be deployed.

If Ray keeps using older code after a deploy, append a cache-busting query parameter to `working_dir` (for example `.../main.zip?v=2`) and deploy again.

## Best Practices

1. **Error Handling**: Always wrap inference in try-except blocks
2. **Logging**: Use `print()` or `logging` for debugging (viewable in pod logs)
3. **Resource Limits**: Set appropriate CPU/memory/GPU limits
4. **Model Loading**: Cache models to avoid reloading on each request
5. **Input Validation**: Validate input data format and ranges
6. **Batching**: Use batching for throughput-intensive workloads
7. **Health Checks**: Implement health check endpoints for monitoring

## Related Guides

- [Deployment Guide](deployment-guide.md)
- [Configuration Reference](configuration-reference.md)
- [Architecture Overview](../architecture/overview.md)
