# Deployment Guide

In this guide, you will transition a locally tested model to a production environment in Kubernetes.

You will learn:

- How to structure your Python code for scalable processing using Ray Serve.
- How to configure a Helm release for your model.
- How to identify and avoid common resource allocation pitfalls.

## Prerequisites

Before deploying your model, verify the following prerequisites:

- The KubeRay operator is installed and functioning on your cluster.
- You have access to a target namespace (e.g., `rationai-jobs-ns`).
- Your Python code executes successfully in a standard local environment.
- Your cluster has necessary network access (e.g., MLflow, S3) to download model weights.

---

## Step 1: Prepare the Python Model

Model Service requires an entrypoint properly decorated for Ray Serve. Create a file inside the repository's root `models/` directory (e.g., `models/my_model.py`):

```python
# models/my_model.py
import asyncio
from typing import TypedDict

import numpy as np
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve

fastapi = FastAPI()


class Config(TypedDict):
    tile_size: int
    max_batch_size: int
    batch_wait_timeout_s: float


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class MyModel:
    """Sample binary classifier example for tissue tiles."""

    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        """Called automatically by Ray when Helm values change."""
        self.tile_size = config.get("tile_size", 512)

        # Load your weights here, e.g., via ONNX Runtime or torch
        # self.session = ort.InferenceSession("model.onnx")

        self.predict.set_max_batch_size(config["max_batch_size"])
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        """Batched inference — Ray collects concurrent requests and calls this together."""
        batch = np.stack(images, axis=0, dtype=np.uint8)

        # Perform batched inference here
        # outputs = self.session.run(None, {"input": batch})
        # return outputs[0].flatten().tolist()

        # Mock prediction returning a list of floats
        return (batch.mean(axis=(1, 2, 3)) / 255.0).tolist()

    @fastapi.post("/")
      async def root(self, request: Request) -> list[float]:
        # Receive and decompress LZ4 payload
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        # Decode image to numpy array
        image = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape(self.tile_size, self.tile_size, 3)
            .transpose(2, 0, 1)  # Convert to CHW format
        )

        result = await self.predict(image)
        return result

app = MyModel.bind()
```

### Note: Why use `__init__` and `reconfigure`?

- **`__init__`**: Executes once when the replica initializes. Ideal for static components like loading neural network weights into memory. For performance and hardware compatibility, it is recommended to format your weights as an ONNX model (see [Model Export](https://youtrack.rationai.cloud.e-infra.cz/articles/DEV-A-15/Model-Export)).
- **`reconfigure`**: Executes automatically when the application receives dynamic configuration updates via Helm or the cluster management API.
- **`FastAPI` Route (`@fastapi.post`)**: Binds an external HTTP endpoint to accept requests.

---

## Step 2: Add your Helm YAML definition

Kubernetes requires the resource specification associated with the code. Create a file named `my-model.yaml` within `helm/rayservice/applications/`:

```yaml
- name: my-model
  import_path: models.my_model:app
  route_prefix: /my-model

  runtime_env:
    working_dir: https://github.com/RationAI/model-service/archive/refs/heads/feature/my-new-model.zip

  deployments:
    - name: MyModel
      max_ongoing_requests: 32
      max_queued_requests: 64

      autoscaling_config:
        min_replicas: 0
        max_replicas: 4
        target_ongoing_requests: 16

      ray_actor_options:
        num_cpus: 2
        runtime_env:
          pip:
            - fastapi
```

For development and testing, prefer a dedicated branch in `working_dir` (for example `feature/my-new-model`) so unfinished changes do not affect other users.

**Tip:** Append a query string to the working directory URL (e.g. `?v=1`) between testing updates to invalidate Ray's caching and force it to pick up your latest code.

---

## Step 3: Deploy

The configuration is rendered by Helm. Deploy it with:

```bash
helm upgrade --install rayservice-model helm/rayservice -n rationai-jobs-ns
```

In this command, `rayservice-model` is the Helm release name parameter. Change it to your desired release name (for example `rayservice-model-my-model`) when running isolated test deployments.

This renders the templates and provisions the deployment on the target cluster namespace.

---

## Step 4: Monitor the Deployment

Ray performs several bootstrap steps, including node assignment, code extraction, and package installation. Monitor progress using `kubectl`:

```bash
# Monitor logical Ray Serve applications continuously
kubectl get rayservice rayservice-model -n rationai-jobs-ns -w

# Monitor physical pod scheduling limits
kubectl get pods -n rationai-jobs-ns -l ray.io/cluster=rayservice-model
```

If Ray service remains offline, monitor the Ray head node logs:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head -f
```

---

## Step 5: Test the Endpoint

When the application indicates `RUNNING`, forward the service port locally:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-serve-svc 8000:8000
```

Transmit an LZ4-compressed payload to test the interface:

```python
import lz4.frame
import numpy as np
import requests

# Create a mock 512x512 RGB tile
tile = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
payload = lz4.frame.compress(tile.tobytes())

response = requests.post(
    "http://localhost:8000/my-model/",
    data=payload,
    headers={"Content-Type": "application/octet-stream"}
)
print("Prediction:", response.json())
```

---

## Advanced: TensorRT Optimization

When deploying models using ONNX Runtime, utilizing the **TensorrtExecutionProvider** can drastically improve inference speed. However, proper configuration requires specifying dynamic shapes, workspace sizes, and engine caching.

### Updating Configuration

In your `Config` class, add the necessary TensorRT parameters:

```python
from typing import NotRequired

class Config(TypedDict):
    tile_size: int
    max_batch_size: int
    batch_wait_timeout_s: float
    trt_cache_path: str
    intra_op_num_threads: int
    trt_max_workspace_size: NotRequired[int]
    trt_builder_optimization_level: NotRequired[int]
```

### Initializing the Provider

Inside your `reconfigure` method, configure TensorRT options before instantiating the `InferenceSession`. Note that TensorRT compiles its execution plan (the "engine") specific to local hardware and input variable sizes.

```python
import os
import onnxruntime as ort

def reconfigure(self, config: Config) -> None:
    self.tile_size = config.get("tile_size", 512)
    max_batch = config["max_batch_size"]

    cache_path = config["trt_cache_path"]
    os.makedirs(cache_path, exist_ok=True)

    # 1. Define input shapes (NCHW). For batched inputs, max dimension is max_batch_size.
    min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
    opt_shape = f"input:{max_batch}x3x{self.tile_size}x{self.tile_size}"
    max_shape = f"input:{max_batch}x3x{self.tile_size}x{self.tile_size}"

    # 2. Key TensorRT parameters
    trt_options = {
        "device_id": 0,
        "trt_fp16_enable": True,                     # Faster inference using FP16
        "trt_engine_cache_enable": True,             # Cache to avoid rebuilds
        "trt_engine_cache_path": cache_path,
        "trt_timing_cache_enable": True,
        "trt_max_workspace_size": config.get("trt_max_workspace_size", 8 * 1024**3), # 8GB default
        "trt_builder_optimization_level": config.get("trt_builder_optimization_level", 1),
        "trt_profile_min_shapes": min_shape,
        "trt_profile_opt_shapes": opt_shape,
        "trt_profile_max_shapes": max_shape,
    }

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = config["intra_op_num_threads"]
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    self.session = ort.InferenceSession(
        "model.onnx", # Replace with dynamic loader if needed
        providers=[
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ],
        session_options=sess_options,
    )
```

**Key Points:**

- **Dynamic Shapes (`trt_profile_*`)**: You must provide limits defining exactly what batch sizes dimensions can enter. For images, a standard setup is `1x3xHxW` for the minimum, and `<batch>x3xHxW` for the optimal/maximum.
- **Workspace Size (`trt_max_workspace_size`)**: By default, TensorRT only allocates a very small memory limit resolving kernels. Increasing this (for example to 8GB) helps TensorRT find much faster execution strategies.
- **Cache Path (`trt_engine_cache_path`)**: Building the plan can take several minutes. Saving it inside a persistent volume (e.g. matching a `tensorrt-cache-pvc.yaml` bind) prevents latency spikes upon deployment restarts.

---

## Pitfall: Understanding Pods vs. Replicas

The most critical production configuration error stems from confusing physical Kubernetes Pods with logical Ray Replicas.

### 1. Logical Capacity (Replicas)

Assigning `num_cpus: 4` under `ray_actor_options` requires Ray to allocate logical execution space matching 4 CPUs per Replica object. This process searches inside available Worker Pods.

### 2. Physical Capacity (Pods)

Defining `cpu: 8` under `workerGroupSpecs` limits Kubernetes to construct a hardware-aligned worker context representing 8 CPUs.

### The Relationship

A Worker Pod with **8 CPUs** can effectively host **two copies** of a logical Replica requesting **4 CPUs**.

If a Pod possesses **3 CPUs**, a **4 CPU** Replica continuously fails to provision because there is no single physical environment structurally capable of accommodating the required footprint. The status resolves continuously to "Pending".

> **Requirement:**
> `Physical Pod CPU >= (Logical Replica CPU * Expected Replicas Per Pod) + Overhead`

**Resource Buffer Defaults:**

- **CPU**: Add approximately 0.5 - 2.0 processor cores per node to support Ray backend processes.
- **Memory**: Add approximately 1 - 2 GiB for caching objects plus ~30% size variation bounds to avoid OOM termination.

#### Example Calculation

**Scenario:** Deploy 5 replicas of a model requiring 4 CPUs and 4GB RAM on a single node.

1.  **Logical Needs**: 5 replicas × 4 CPUs = **20 Logical CPUs**.
2.  **Physical Overhead**: We estimate 2 CPUs for Raylet/System.
3.  **Total Physical Request**: 20 + 2 = **22 CPUs**.

_If you request only 20 CPUs in Kubernetes, Ray will detect that some CPU is used by the OS/Raylet and might only offer 19 logical slots, causing the 5th replica to hang._

### Autoscaling Configuration

**Choose appropriate scaling parameters:**

```yaml
autoscaling_config:
  min_replicas: 1
  max_replicas: 10
  target_ongoing_requests: 20

  # Advanced stabilization
  upscale_delay_s: 30
  downscale_delay_s: 600
```

**Key Tuning Recommendations:**

1.  **`target_ongoing_requests`**:
    - **Lower this value** (e.g., 5-10) for latency-sensitive models or if your model is CPU-heavy. This forces the system to scale out sooner.
    - **Increase this value** (e.g., 50-100) for simple models where a single replica can juggle many async requests.

2.  **`upscale_delay_s`**:
    - Keep this low (e.g., `0s` to `30s`) so the system reacts quickly to traffic spikes.

3.  **`downscale_delay_s`**:
    - Keep this high (e.g., `600s`) to avoid "thrashing". It is cheaper to keep an idle replica for 10 minutes than to re-initialize a heavy model (loading weights, etc.) every time traffic dips for a minute.

For the exact formulas and definitions of these settings, see the [Configuration Reference](configuration-reference.md#autoscaling-strategies).

### High Availability

**For production workloads:**

```yaml
# Multiple replicas
autoscaling_config:
  min_replicas: 2 # At least 2 for redundancy

# Multiple workers
workerGroupSpecs:
  - groupName: cpu-workers
    minReplicas: 2
    maxReplicas: 10
```

### Resource Limits

**Always set resource limits:**

```yaml
containers:
  - name: ray-worker
    resources:
      requests: # Guaranteed resources
        cpu: 8
        memory: 16Gi
      limits: # Maximum resources
        cpu: 12
        memory: 20Gi
```

### Network Configuration

**Proxy settings:**

```yaml
env:
  - name: HTTPS_PROXY
    value: "http://proxy.example.com:3128"
```

**Service configuration:**

```yaml
# If you need external access
apiVersion: v1
kind: Service
metadata:
  name: rayservice-external
spec:
  type: LoadBalancer
  selector:
    ray.io/cluster: rayservice-model
  ports:
    - port: 80
      targetPort: 8000
```

## Multi-Model Deployment

Deploy multiple models by adding multiple application definitions to `helm/rayservice/applications/`. Helm renders all application files into one `serveConfigV2` during deployment:

```yaml
# helm/rayservice/applications/model-a.yaml
- name: model-a
  import_path: models.model_a:app
  route_prefix: /model-a
  deployments:
    - name: ModelA
      ray_actor_options:
        num_cpus: 4

# helm/rayservice/applications/model-b.yaml
- name: model-b
  import_path: models.model_b:app
  route_prefix: /model-b
  deployments:
    - name: ModelB
      ray_actor_options:
        num_gpus: 1
```

## Updating Deployments

### Update Model Code

1. Update code in the repository.
2. Commit and push your changes.
3. Redeploy with Helm:

```bash
helm upgrade --install rayservice-model helm/rayservice -n rationai-jobs-ns
```

4. If `runtime_env.working_dir` URL is unchanged and old code is still used, bump a cache-busting query parameter (for example `...?v=2`) and redeploy.

### Update Configuration

```bash
# Edit application configuration
vim helm/rayservice/applications/my-model.yaml

# Apply changes with Helm
helm upgrade --install rayservice-model helm/rayservice -n rationai-jobs-ns
```

KubeRay will reconcile the RayService and attempt a rolling-style update:

- New replicas are created with the new config
- Traffic is routed to healthy replicas
- Old replicas are eventually removed

### Update Model Weights

If using MLflow, update the tracked artifact inside your application's YAML definition:

```yaml
user_config:
  model:
    artifact_uri: mlflow-artifacts:/65/NEW_RUN_ID/model.onnx
```

Apply the update:

```bash
helm upgrade --install rayservice-model helm/rayservice -n rationai-jobs-ns
```

## Rollback

If deployment fails, rollback:

```bash
# RayService is a Custom Resource (CRD), so Kubernetes "rollout" doesn't apply.
# Instead, view KubeRay status and events, then re-apply a known-good spec.

# Inspect current state and recent events
kubectl get rayservice rayservice-model -n rationai-jobs-ns -o yaml
kubectl describe rayservice rayservice-model -n rationai-jobs-ns

# Check Ray Serve controller logs (usually shows the root cause)
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
```

## Troubleshooting

### Deployment Stuck

**Check RayService status:**

```bash
kubectl describe rayservice rayservice-model -n rationai-jobs-ns
```

**Common issues:**

- Image pull errors
- Insufficient resources
- Configuration errors
- Network issues

### Application Not Starting

**Check serve application logs:**

```bash
# View dashboard
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-head-svc 8265:8265

# Check logs
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=100
```

**Common issues:**

- Python import errors
- Model loading failures
- Dependency issues
- Resource limits

### High Latency

**Check metrics:**

```bash
# Ray dashboard: http://localhost:8265
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-head-svc 8265:8265
```

**Possible solutions:**

- Increase replicas
- Enable batching
- Optimize model code
- Increase resources

## Best Practices

1. **Version Control**: Keep all YAML configs in Git
2. **Testing**: Test locally before deploying
3. **Monitoring**: Set up alerts for failures
4. **Resource Limits**: Always set limits to prevent resource hogging
5. **Gradual Rollout**: Update replicas gradually
6. **Documentation**: Document custom configurations
7. **Backup**: Keep backups of working configurations

## Next Steps

- [Configuration reference](configuration-reference.md)
- [Architecture overview](../architecture/overview.md)
- [Adding new models](adding-models.md)
- [Troubleshooting](troubleshooting.md)
