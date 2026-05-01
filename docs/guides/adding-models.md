# Adding New Models

This guide covers the full model preparation path before deployment: export the model to ONNX, store it in MLflow, and then implement the Python entrypoint that Ray Serve will load.

If you are adding a new model from scratch, start here first. Once the Python file is ready, continue with the [Deployment Guide](deployment-guide.md) for Helm configuration, deploy, and rollout monitoring.

## Export the Model to ONNX

Export your model to ONNX before writing the Serve entrypoint. The project expects ONNX so the runtime can load the artifact efficiently and use batching through `@serve.batch`.

If you already have a conversion script, reuse it here. Make sure the exported graph accepts a batch dimension, because the Ray Serve handler will batch multiple requests together.

```python
import torch

model = Model()
model.eval()

torch.onnx.export(
  model,
  torch.randn(1, 3, 512, 512),
  "model.onnx",
  export_params=True,
  do_constant_folding=True,
  input_names=["input"],
  output_names=["output"],
  dynamic_axes={
    "input": {0: "batch_size"},
    "output": {0: "batch_size"},
  },
)
```

If the exported model is too large and ONNX splits it into multiple files, see [Troubleshooting: Large ONNX Model Export](troubleshooting.md#large-onnx-model-export-2gb) for merge and multi-file handling.

## Upload the ONNX Artifact to MLflow

After export, upload the ONNX artifact to MLflow. If the export produced external weight files, upload the whole directory instead of only the `.onnx` graph file.

```python
import mlflow

with mlflow.start_run():
  mlflow.log_artifact("model.onnx", artifact_path="model")

  # If the export is split into multiple files, upload the full directory:
  # mlflow.log_artifacts("onnx_export_dir", artifact_path="model")
```

The deployment guide explains how the runtime reads this artifact back during startup.

## Start With an Existing Implementation

Do not start from an empty file. Copy the closest implementation and adapt only model-specific pieces.

- **Binary classification (baseline)**: [`models/binary_classifier.py`](https://github.com/RationAI/model-service/blob/main/models/binary_classifier.py)
- **Semantic segmentation**: [`models/semantic_segmentation.py`](https://github.com/RationAI/model-service/blob/main/models/semantic_segmentation.py)
- **Virchow2 embedding/classification**: [`models/virchow2.py`](https://github.com/RationAI/model-service/blob/main/models/virchow2.py)
- **Heatmap pipeline**: [`builders/heatmap_builder.py`](https://github.com/RationAI/model-service/blob/main/builders/heatmap_builder.py)

Matching application definitions live in `helm/rayservice/applications/`, but the YAML itself is covered in the [Deployment Guide](deployment-guide.md).

## Python Entrypoint

Create a new file in `models/` (for example `models/my_model.py`). The sections below explain what each method does and what belongs inside it.

### `__init__`: one-time lightweight setup

```python
def __init__(self) -> None:
  import lz4.frame
  self.lz4 = lz4.frame
```

- `import lz4.frame`: imports compression utilities once for this replica.
- `self.lz4 = lz4.frame`: stores a reusable handle so request handlers do not re-import.

What belongs in `__init__`:

- lightweight constants,
- reusable helpers,
- cheap setup only.

What should not be here:

- heavy model download,
- config-dependent initialization.

### `reconfigure`: required runtime initialization

`reconfigure` is called after startup and on `user_config` changes. In this project, treat it as required because model path and runtime settings come from config.

```python
def reconfigure(self, config: Config) -> None:
  import importlib
  import onnxruntime as ort

  self.tile_size = config["tile_size"]

  model_config = dict(config["model"])
  module_path, attr_name = model_config.pop("_target_").split(":")
  provider = getattr(importlib.import_module(module_path), attr_name)

  self.session = ort.InferenceSession(
    provider(**model_config),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
  )

  self.input_name = self.session.get_inputs()[0].name
  self.output_name = self.session.get_outputs()[0].name

  self.predict.set_max_batch_size(config["max_batch_size"])
  self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])
```

For TensorRT, mixed precision, and other runtime tuning options, see [Optimization Guide](optimization-guide.md).

Data and control flow:

- Read static inference shape from config (`tile_size`).
- Resolve model provider from `_target_` and fetch the ONNX artifact (typically from MLflow).
- Build ONNX Runtime session with desired execution providers.
- Cache input and output tensor names for fast inference calls.
- Apply batching limits from config directly to `predict`.

### `predict`: batched ONNX inference

```python
@serve.batch
async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
  batch = np.stack(images, axis=0, dtype=np.uint8)
  outputs = self.session.run([self.output_name], {self.input_name: batch})
  return outputs[0].flatten().tolist()
```

What happens with data:

- Ray Serve collects many incoming requests into `images`.
- `np.stack` converts many single images into one batch tensor.
- ONNX runtime computes one forward pass over the whole batch.
- Output tensor is flattened and returned as a Python list.
- Ray Serve maps each list item back to the original HTTP request.

### `root`: HTTP request parsing and serialization

```python
@fastapi.post("/")
async def root(self, request: Request) -> float:
  data = await asyncio.to_thread(self.lz4.decompress, await request.body())
  image = (
    np.frombuffer(data, dtype=np.uint8)
    .reshape(self.tile_size, self.tile_size, 3)
    .transpose(2, 0, 1)
  )
  result = await self.predict(image)
  return result
```

What each step does:

- `await request.body()`: gets raw HTTP bytes.
- `lz4.decompress`: reconstructs original image bytes.
- `np.frombuffer(...).reshape(...)`: converts bytes to `HWC` uint8 image.
- `.transpose(2, 0, 1)`: converts image to `CHW` layout expected by ONNX model.
- `await self.predict(image)`: sends item to batching queue and waits for matching output.

### Application binding

```python
app = MyModel.bind()
```

This exported symbol is what `import_path: models.my_model:app` points to in Helm.

## Next Step

After the Python entrypoint is ready, continue with the [Deployment Guide](deployment-guide.md). That guide covers the Helm application YAML, deployment, rollout monitoring, and smoke testing in the order you should run them.

## Related Guides

- [Deployment Guide](deployment-guide.md)
- [Configuration Reference](configuration-reference.md)
- [Optimization Guide](optimization-guide.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture Overview](../architecture/overview.md)
