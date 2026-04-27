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

This guide focuses on deployment, so avoid writing a full model implementation here.

Before continuing, ensure your model module already exists and exposes a Serve app object (for example `app = MyModel.bind()`). If you need to implement a new model, follow the [Adding New Models](adding-models.md) guide first.

Use existing implementations as templates:

- `models/binary_classifier.py`
- `models/semantic_segmentation.py`
- `models/virchow2.py`

Your deployment contract should be:

- Ray Serve entrypoint importable through `import_path` (`module.path:app`).
- Request route defined via FastAPI ingress (or equivalent Serve HTTP handler).
- Optional `reconfigure` support for dynamic `user_config` updates.
- Runtime dependencies available through `runtime_env`.

### Note: Why use `__init__` and `reconfigure`?

- **`__init__`**: Executes once when the replica initializes. Ideal for static components like loading neural network weights into memory. For performance and hardware compatibility, it is recommended to format your weights as an ONNX model.
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
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
```

In this command, `<release-name>` is your Helm release name. Use a dedicated test name (for example `rayservice-model-my-model`) for isolated deployments.

This renders the templates and provisions the deployment on the target cluster namespace.

---

## Step 4: Monitor the Deployment

Ray performs several bootstrap steps, including node assignment, code extraction, and package installation. Monitor progress using `kubectl`:

```bash
# Monitor logical Ray Serve applications continuously
kubectl get rayservice <release-name> -n rationai-jobs-ns -w

# Monitor physical pod scheduling limits
kubectl get pods -n rationai-jobs-ns -l ray.io/cluster=<release-name>
```

If Ray service remains offline, monitor the Ray head node logs:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head -f
```

---

## Step 5: Test the Endpoint

When the application indicates `RUNNING`, forward the service port locally:

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-serve-svc 8000:8000
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

When deploying models using ONNX Runtime, utilizing the **TensorrtExecutionProvider** can drastically improve inference speed. The Python implementation for this is already available in the provided templates (e.g., `models/binary_classifier.py`).

From a deployment perspective, you must provide the correct configuration in your application's `user_config` (in `helm/rayservice/applications/`):

1. **Dynamic Shapes (`trt_profile_*`)**: Define limits for batch size dimensions. For images, a standard setup requires `1x3xHxW` for the minimum, and `<batch>x3xHxW` for the optimal/maximum.
2. **Workspace Size (`trt_max_workspace_size`)**: By default, TensorRT only allocates a very small memory limit resolving kernels. Increasing this (for example to 8GB) helps TensorRT find much faster execution strategies.
3. **Engine Caching (`trt_cache_path`)**: Building the execution plan can take several minutes. Saving it inside a persistent volume (e.g., matching a `pvc/tensorrt-cache-pvc.yaml` bind) prevents latency spikes upon deployment restarts.

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

For exact autoscaling formulas and tuning heuristics, see [Autoscaling Strategies](configuration-reference.md#autoscaling-strategies) in the configuration reference.

### Configuration Details (Reference)

To keep this guide focused on deployment flow (and avoid duplicated config docs), detailed settings are maintained in one place:

- [Application Definition](configuration-reference.md#2-application-definition): app-level fields (`import_path`, `route_prefix`, `runtime_env`, `deployments`).
- [Deployment-Level Tuning](configuration-reference.md#3-deployment-level-tuning): `max_ongoing_requests`, `max_queued_requests`, `autoscaling_config`, `ray_actor_options`, `user_config`.
- [Autoscaling Strategies](configuration-reference.md#autoscaling-strategies): `target_ongoing_requests`, `upscale_delay_s`, `downscale_delay_s`.
- [Cluster and Worker Resources](configuration-reference.md#4-cluster-and-worker-resources): worker sizing, CPU/GPU pools, resource requests/limits.

For model-centric examples (including `user_config` patterns), see [Adding New Models](adding-models.md#rayservice-configuration).

> **Recommendation:** Keep the default worker profiles unless you need specific hardware or scheduling behavior. In most deployments, users should tune application/deployment settings first and only customize worker templates when necessary.

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
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
```

4. If `runtime_env.working_dir` URL is unchanged and old code is still used, bump a cache-busting query parameter (for example `...?v=2`) and redeploy.

### Update Configuration

```bash
# Edit application configuration
vim helm/rayservice/applications/my-model.yaml

# Apply changes with Helm
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
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
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
```

## Rollback

If deployment fails, rollback:

```bash
# RayService is a Custom Resource (CRD), so Kubernetes "rollout" doesn't apply.
# Instead, view KubeRay status and events, then re-apply a known-good spec.

# Inspect current state and recent events
kubectl get rayservice <release-name> -n rationai-jobs-ns -o yaml
kubectl describe rayservice <release-name> -n rationai-jobs-ns

# Check Ray Serve controller logs (usually shows the root cause)
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
```

## Troubleshooting

### Deployment Stuck

**Check RayService status:**

```bash
kubectl describe rayservice <release-name> -n rationai-jobs-ns
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
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-head-svc 8265:8265

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
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-head-svc 8265:8265
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

## Related Guides

- [Configuration reference](configuration-reference.md)
- [Architecture overview](../architecture/overview.md)
- [Adding new models](adding-models.md)
- [Troubleshooting](troubleshooting.md)
