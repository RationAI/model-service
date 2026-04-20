# Deployment Guide

In this guide, you will transition a locally tested model to a production environment in Kubernetes.

You will learn:

- How to structure your Python code for scalable processing using Ray Serve.
- How to configure a Kustomize application definition for your model.
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
from typing import TypedDict
from fastapi import FastAPI, Request
from ray import serve

class Config(TypedDict):
    threshold: float

app_ingress = FastAPI()

@serve.deployment(num_replicas="auto")
@serve.ingress(app_ingress)
class MyModel:
    def __init__(self):
        # Initialize heavy dependencies or weights once per replica
        self.model = self.load_model_once()
        self.threshold = 0.5

    def load_model_once(self):
        # Initialization logic
        return object()

    def reconfigure(self, config: Config):
        # Ray automatically triggers this when you update the YAML configuration
        self.threshold = config.get("threshold", 0.5)

    @app_ingress.post("/")
    async def predict(self, request: Request):
        data = await request.json()
        score = float(data["input"])
        return {"score": score, "label": score >= self.threshold}

app = MyModel.bind()
```

### Note: Why use `__init__` and `reconfigure`?

- **`__init__`**: Executes once when the replica initializes. Ideal for static components like loading neural network weights into memory.
- **`reconfigure`**: Executes automatically when the application receives dynamic configuration updates via Kustomize or the cluster management API.
- **`FastAPI` Route (`@app_ingress.post`)**: Binds an external HTTP endpoint to accept standard JSON formatting directly.

---

## Step 2: Add your Kustomize YAML definition

Kubernetes requires the resource specification associated with the code. Create a file named `my-model.yaml` within `kustomize/components/models/models-definitions/`:

```yaml
applications:
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

---

## Step 3: Deploy

The configuration relies on Kustomize to inject your deployment components. Deploy it by executing the shell script:

```bash
./deploy.sh
```

This compiles your definitions and provisions the deployment on the target cluster namespace.

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

Transmit a JSON payload to test the interface:

```bash
curl -X POST http://localhost:8000/my-model/ -H "Content-Type: application/json" -d '{"input": 0.8}'
```

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

For the exact formulas and definitions of these settings, see the [Configuration Reference](configuration-reference.md#22-autoscaling-strategies).

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

Deploy multiple models by adding multiple application definitions to your Kustomize `models-definitions/` directory. Each file will be merged automatically:

```yaml
# kustomize/components/models/models-definitions/model-a.yaml
applications:
  - name: model-a
    import_path: models.model_a:app
    route_prefix: /model-a
    deployments:
      - name: ModelA
        ray_actor_options:
          num_cpus: 4

# kustomize/components/models/models-definitions/model-b.yaml
applications:
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
3. KubeRay will automatically fetch the new code from the `working_dir` URL on the next deployment.

### Update Configuration

```bash
# Edit configuration in models-definitions/
vim kustomize/components/models/models-definitions/my-model.yaml

# Apply changes using the deploy script
./deploy.sh
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
./deploy.sh
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
