# Adding New Models

This guide explains how to integrate your own machine learning models into Model Service.

## Overview

To add a new model, you need to:

1. Create a model class with Ray Serve decorators
2. Implement the inference logic
3. Configure the model in `kustomize/components/models/models-definitions/`
4. Deploy and test

## Model Implementation

### Basic Structure

Create a Python file in the `models/` directory:

```python
from fastapi import FastAPI, Request
from ray import serve

app_ingress = FastAPI()

@serve.deployment(ray_actor_options={"num_cpus": 2})
@serve.ingress(app_ingress)
class MyModel:
    def __init__(self):
        # Load your model here
        pass

    @app_ingress.post("/")
    async def predict(self, request: Request):
        # Handle inference requests
        data = await request.json()
        # Process data and return prediction
        result = data
        return {"prediction": result}

app = MyModel.bind()
```

The repository's reference model `BinaryClassifier` uses FastAPI ingress + batched inference and expects a **compressed binary payload** (not JSON). For simple JSON models, the examples above are fine; for high-throughput image inference, consider the batching and ingress patterns shown below.

### Key Components

#### 1. Deployment Decorator

The `@serve.deployment` decorator marks your class as a Ray Serve deployment:

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,           # CPUs per replica
        "num_gpus": 0,           # GPUs per replica
        "memory": 2 * 1024**3,   # Memory in bytes
    }
)
class MyModel:
    ...
```

#### 2. Initialization

Load your model in `__init__`. This method corresponds to the replica **startup phase**.

```python
def __init__(self):
    # This runs ONCE when the replica starts.
    # The replica is NOT ready for traffic until this returns.
    import torch

    self.model = torch.load("model.pt")
    self.model.eval()
    print("Model loaded successfully")
```

#### 3. Resource Packing (Fractional CPUs/GPUs)

Ray allows fractional resource requests. This lets you pack multiple small replicas onto a single node.

```python
# Run 4 replicas on a single 1-CPU node (0.25 * 4 = 1.0)
@serve.deployment(ray_actor_options={"num_cpus": 0.25})
```

#### 4. Inference Method

To handle requests, implement routes using the `@app_ingress.post` pattern (for FastAPI) instead of the standard `__call__` method.

```python
@app_ingress.post("/")
async def predict(self, request: Request):
    data = await request.json()
    input_data = self.preprocess(data["input"])

    with torch.no_grad():
        output = self.model(input_data)

    return {"prediction": self.postprocess(output)}
```

## Advanced Features

### Dynamic Configuration

Use `reconfigure()` to update model settings without redeployment:

```python
from typing import TypedDict

class Config(TypedDict):
    threshold: float
    batch_size: int

@serve.deployment
class ConfigurableModel:
    def __init__(self):
        self.model = load_model()

    def reconfigure(self, config: Config):
        self.threshold = config["threshold"]
        self.batch_size = config["batch_size"]
        print(f"Reconfigured: threshold={self.threshold}")
```

Update config via your model definition YAML:

```yaml
user_config:
  threshold: 0.5
  batch_size: 32
```

### Batching Requests

Use `@serve.batch` for efficient batch processing:

```python
app_ingress = FastAPI()

@serve.deployment
@serve.ingress(app_ingress)
class BatchedModel:
    def __init__(self):
        self.model = load_model()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def predict_batch(self, inputs: list[np.ndarray]):
        batch = np.stack(inputs)
        outputs = self.model(batch)
        return outputs.tolist()

    @app_ingress.post("/")
    async def process_request(self, request: Request):
        data = await request.json()
        input_data = np.array(data["input"])
        result = await self.predict_batch(input_data)
        return {"prediction": result}
```

For binary/image workloads, you can also batch raw `bytes` like the `BinaryClassifier` does (see `models/binary_classifier.py`). This avoids JSON overhead and lets you control batch sizing via `user_config` by calling `set_max_batch_size()` and `set_batch_wait_timeout_s()`.

### Using FastAPI

For advanced HTTP features, use FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    input: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

fastapi = FastAPI()

@serve.deployment
@serve.ingress(fastapi)
class FastAPIModel:
    def __init__(self):
        self.model = load_model()

    @fastapi.post("/predict", response_model=PredictionResponse)
    async def predict(self, request: PredictionRequest):
        output = self.model(request.input)
        return PredictionResponse(
            prediction=float(output),
            confidence=0.95
        )

app = FastAPIModel.bind()
```

## Loading Models from MLflow

Use the model provider to load from MLflow:

```python
# models/mlflow_model.py
from fastapi import FastAPI, Request
from ray import serve
from providers.model_provider import mlflow

app_ingress = FastAPI()

@serve.deployment
@serve.ingress(app_ingress)
class MLflowModel:
    def __init__(self):
        # This will be set via user_config
        self.model_path = None

    async def reconfigure(self, config):
        model_uri = config["model"]["artifact_uri"]
        self.model_path = mlflow(artifact_uri=model_uri)

        # Load model
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.model_path)

    @app_ingress.post("/")
    async def predict(self, request: Request):
        # Inference logic
        ...

app = MLflowModel.bind()
```

Configure in YAML:

```yaml
runtime_env:
  env_vars:
    MLFLOW_TRACKING_URI: http://mlflow.rationai-mlflow:5000
user_config:
  model:
    artifact_uri: mlflow-artifacts:/65/abc123.../model.onnx
```

## RayService Configuration

Add your model file to `kustomize/components/models/models-definitions/my-model.yaml`:

```yaml
applications:
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

In this repository, model dependencies can be installed under `deployments[*].ray_actor_options.runtime_env.pip` (not only at `applications[*].runtime_env`). This is useful when different deployments need different dependencies.

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

## GPU Models

For GPU-accelerated models:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app_ingress)
class GPUModel:
    def __init__(self):
        import torch

        self.device = torch.device("cuda")
        self.model = torch.load("model.pt").to(self.device)
        self.model.eval()

    @app_ingress.post("/")
    async def predict(self, request: Request):
        data = await request.json()
        input_tensor = torch.tensor(data["input"]).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return {"prediction": output.cpu().numpy().tolist()}
```

Ensure your deployment specifies `num_gpus` greater than `0`. Model Service natively provides a **GPU workers** configuration. The Kustomize build step handles assigning `nvidia.com/gpu` automatically when the `gpu-workers` component is included in `kustomize/overlays/kustomization.yaml`.

## Deployment

Deploy your model:

```bash
./deploy.sh
```

Monitor deployment:

```bash
kubectl get rayservice -n rationai-jobs-ns
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=100
```

Open the Ray dashboard:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-head-svc 8265:8265
```

Then visit `http://127.0.0.1:8265`.

## Best Practices

1. **Error Handling**: Always wrap inference in try-except blocks
2. **Logging**: Use `print()` or `logging` for debugging (viewable in pod logs)
3. **Resource Limits**: Set appropriate CPU/memory/GPU limits
4. **Model Loading**: Cache models to avoid reloading on each request
5. **Input Validation**: Validate input data format and ranges
6. **Batching**: Use batching for throughput-intensive workloads
7. **Health Checks**: Implement health check endpoints for monitoring

## Next Steps

- [Deployment guide](deployment-guide.md)
- [Configuration reference](configuration-reference.md)
- [Architecture overview](../architecture/overview.md)
