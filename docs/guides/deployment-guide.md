# Deployment Guide

Complete guide for deploying models to production with Model Service.

## Prerequisites

Before deploying to production, ensure:

- [x] KubeRay operator installed
- [x] Namespace created (`rationai-notebooks-ns`)
- [x] Model tested locally
- [x] RayService YAML configured
- [x] MLflow accessible (if using MLflow)

## Deployment Workflow

### 1. Prepare Model Code

Ensure your model is in the `models/` directory and properly structured:

```python
# models/my_model.py
from ray import serve
from starlette.requests import Request

@serve.deployment(ray_actor_options={"num_cpus": 2})
class MyModel:
    def __init__(self):
        # Model initialization
        self.model = self.load_model()

    def load_model(self):
        # Load model logic
        pass

    async def __call__(self, request: Request):
        # Inference logic
        data = await request.json()
        result = self.model.predict(data["input"])
        return {"prediction": result}

app = MyModel.bind()
```

### 2. Create RayService Configuration

Create or modify `ray-service.yaml`:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-my-model
  namespace: rationai-notebooks-ns
spec:
  serveConfigV2: |
    applications:
      - name: my-model
        import_path: models.my_model:app
        route_prefix: /my-model
        runtime_env:
          working_dir: https://gitlab.ics.muni.cz/rationai/infrastructure/model-service/-/archive/master/model-service-master.zip
          pip:
            - numpy
            - pandas
          env_vars:
            MODEL_VERSION: "1.0.0"
        deployments:
          - name: MyModel
            autoscaling_config:
              min_replicas: 1
              max_replicas: 5
              target_ongoing_requests: 32
            ray_actor_options:
              num_cpus: 4
              memory: 4294967296  # 4 GiB
              runtime_env:
                pip:
                  - numpy
                  - pandas

  rayClusterConfig:
    rayVersion: 2.52.1
    enableInTreeAutoscaling: true
    autoscalerOptions:
      idleTimeoutSeconds: 300

    headGroupSpec:
      rayStartParams:
        num-cpus: "0"
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.52.1-py312
              resources:
                limits:
                  cpu: 2
                  memory: 4Gi

    workerGroupSpecs:
      - groupName: cpu-workers
        replicas: 1
        minReplicas: 1
        maxReplicas: 10
        template:
          spec:
            containers:
              - name: ray-worker
                image: rayproject/ray:2.52.1-py312
                resources:
                  limits:
                    cpu: 8
                    memory: 16Gi
```

### 3. Deploy to Kubernetes

Apply the configuration:

```bash
kubectl apply -f ray-service.yaml -n [namespace]
```

### 4. Monitor Deployment

Watch the deployment progress:

```bash
# Watch RayService status
kubectl get rayservice rayservice-my-model -n [namespace] -w

# Check pods
kubectl get pods -n [namespace] -l ray.io/cluster=rayservice-my-model

# View head node logs
kubectl logs -n [namespace] -l ray.io/node-type=head -f

# View worker logs
kubectl logs -n [namespace] -l ray.io/node-type=worker -f
```

Wait for status to show `Running` and application status to show `RUNNING`.

### 5. Verify Deployment

Check service endpoints:

```bash
# Get service details
kubectl get svc -n [namespace]

# Port forward to test
kubectl port-forward -n [namespace] \
  svc/rayservice-my-model-serve-svc 8000:8000
```

The example model in this repository (`models/binary_classifier.py`) uses FastAPI ingress and expects a **compressed binary request body** (LZ4), not JSON. The JSON `curl` example below is valid for JSON-based models but does not apply to `BinaryClassifier`.

## Production Considerations

### Resource Planning

**Calculate resource requirements:**

1. **Per-replica resources:**

   - CPU: Based on model complexity
   - Memory: Model size + working memory + overhead
   - GPU: Number of GPUs needed

2. **Total cluster resources:**

   ```
   Total CPUs = max_replicas × num_cpus + overhead
   Total Memory = max_replicas × memory + overhead
   ```

3. **Example calculation:**

   ```
   Model: 4 CPU, 4GB per replica
   Max replicas: 5

   Required per worker: 5 × 4 = 20 CPUs, 5 × 4GB = 20GB
   Overhead: +2 CPUs, +4GB for system

   Worker resources: 22 CPUs, 24GB memory
   ```

### Autoscaling Configuration

**Choose appropriate scaling parameters:**

```yaml
autoscaling_config:
  min_replicas: 1 # Always keep 1 running
  max_replicas: 10 # Scale up to 10
  target_ongoing_requests: 20 # Target load per replica

  # Advanced options
  upscale_delay_s: 30 # Wait 30s before scaling up
  downscale_delay_s: 600 # Wait 10m before scaling down
```

**Scaling behavior:**

- **Cold start**: Set `min_replicas: 0` for scale-to-zero
- **Always available**: Set `min_replicas: 1` or higher
- **High traffic**: Increase `max_replicas` and `target_ongoing_requests`
- **Batch processing**: Use higher `target_ongoing_requests`

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
  - name: HTTP_PROXY
    value: "http://proxy.example.com:3128"
  - name: HTTPS_PROXY
    value: "http://proxy.example.com:3128"
  - name: NO_PROXY
    value: ".svc.cluster.local,.cluster.local"
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
    ray.io/cluster: rayservice-my-model
  ports:
    - port: 80
      targetPort: 8000
```

## Multi-Model Deployment

Deploy multiple models in one RayService:

```yaml
serveConfigV2: |
  applications:
    - name: model-a
      import_path: models.model_a:app
      route_prefix: /model-a
      deployments:
        - name: ModelA
          ray_actor_options:
            num_cpus: 4
    
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

1. Update code in repository
2. Commit and push changes
3. RayService will automatically fetch new code from `working_dir` URL

### Update Configuration

```bash
# Edit configuration
vim ray-service.yaml # or any IDE

# Apply changes
kubectl apply -f ray-service.yaml -n [namespace]
```

KubeRay will reconcile the RayService and attempt a rolling-style update:

- New replicas are created with the new config
- Traffic is routed to healthy replicas
- Old replicas are eventually removed

### Update Model Weights

If using MLflow:

```yaml
user_config:
  model:
    artifact_uri: mlflow-artifacts:/65/NEW_RUN_ID/model.onnx
```

Apply update:

```bash
kubectl apply -f ray-service.yaml -n [namespace]
```

## Rollback

If deployment fails, rollback:

```bash
# RayService is a Custom Resource (CRD), so Kubernetes "rollout" doesn't apply.
# Instead, view KubeRay status and events, then re-apply a known-good spec.

# Inspect current state and recent events
kubectl get rayservice rayservice-my-model -n [namespace] -o yaml
kubectl describe rayservice rayservice-my-model -n [namespace]

# Check Ray Serve controller logs (usually shows the root cause)
kubectl logs -n [namespace] -l ray.io/node-type=head --tail=200
```

## Troubleshooting

### Deployment Stuck

**Check RayService status:**

```bash
kubectl describe rayservice rayservice-my-model -n [namespace]
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
kubectl port-forward -n [namespace] svc/rayservice-my-model-head-svc 8265:8265

# Check logs
kubectl logs -n [namespace] -l ray.io/node-type=worker --tail=100
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
kubectl port-forward -n [namespace] svc/rayservice-my-model-head-svc 8265:8265
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
7. **Backup**: Keep backup of working configurations

## Next Steps

- [Configuration reference](configuration-reference.md)
- [Architecture overview](../architecture/overview.md)
- [Adding new models](adding-models.md)
- [Troubleshooting](troubleshooting.md)
