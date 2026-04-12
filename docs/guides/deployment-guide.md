# Deployment Guide

Ready to go live? 🚀 Let's take your awesome local model and deploy it to a production cluster!

By the end of this guide, you will have a robust, scalable service running in Kubernetes. Grab a coffee, and let's go!

## Are we ready for liftoff?

Before we push the big red button, let's do a quick pre-flight check:

- [x] Is the KubeRay operator installed on your cluster?
- [x] Do you have your target namespace ready? (e.g. `rationai-notebooks-ns`)
- [x] Have you tested your Python code locally?
- [x] (Optional) Can your cluster reach MLflow to download model weights?

Everything checked? Great! Let's deploy.

---

## Step 1: Prep your Python code

First, let's make sure your Python code is structured correctly for Model Service. Create a file in `models/` (let's call it `my_model.py`):

```python
# models/my_model.py
from typing import TypedDict
from fastapi import FastAPI, Request
from ray import serve

# A clean way to type our dynamic config!
class Config(TypedDict):
    threshold: float

app_ingress = FastAPI()

@serve.deployment(num_replicas="auto")
@serve.ingress(app_ingress)
class MyModel:
    def __init__(self):
        # We load the weights ONCE when the replica starts
        self.model = self.load_model_once()
        self.threshold = 0.5 # Default

    def load_model_once(self):
        print("Pretend I'm loading an expensive model right now...")
        return object()

    def reconfigure(self, config: Config):
        # This gets called dynamically if we update the YAML!
        self.threshold = config["threshold"]
        print(f"Updated threshold to {self.threshold}")

    @app_ingress.post("/")
    async def predict(self, request: Request):
        data = await request.json()
        score = float(data["input"])
        return {"score": score, "label": score >= self.threshold}

# This is what Ray looks for!
app = MyModel.bind()
```

Look how elegant that is! We load our model weights once in the `__init__`, we accept dynamic updates via `reconfigure()`, and we handle HTTP traffic with standard FastAPI.

---

## Step 2: Write your RayService YAML

Now we need to tell Kubernetes about our cool new app. Open `ray-service.yaml` and add an entry under `applications`:

```yaml
spec:
  serveConfigV2: |
    applications:
      - name: my-model
        # Point to the code we just wrote!
        import_path: models.my_model:app 
        route_prefix: /my-model
        
        runtime_env:
          # Where to download your code from
          working_dir: https://gitlab.ics.muni.cz/rationai/infrastructure/model-service/-/archive/master/model-service-master.zip
        
        deployments:
          - name: MyModel
            # Don't let replicas get overwhelmed!
            max_ongoing_requests: 32
            max_queued_requests: 64
            
            autoscaling_config:
              min_replicas: 0 # Spin down to save money when idle!
              max_replicas: 4
              target_ongoing_requests: 16
              
            ray_actor_options:
              num_cpus: 2 # Give each replica 2 CPUs
              runtime_env:
                pip:
                  - fastapi
```

---

## Step 3: Deploy to Kubernetes! 🚢

In your terminal, apply the configuration to your specific namespace:

```bash
kubectl apply -f ray-service.yaml -n your-favorite-namespace
```

---

## Step 4: Watch the magic happen

Ray is now waking up computers, downloading your zipped code, pip-installing FastAPI, and spinning up your replicas. Let's watch the progress:

```bash
# Watch the high-level status (press Ctrl+C to exit)
kubectl get rayservice rayservice-model -n your-favorite-namespace -w

# Check on the workers!
kubectl get pods -n your-favorite-namespace -l ray.io/cluster=rayservice-model
```

If you ever get stuck, looking at the logs is the easiest way to figure out what happened:

```bash
# What is the Ray head node doing?
kubectl logs -n your-favorite-namespace -l ray.io/node-type=head -f
```

---

## Step 5: Test it out!

Once the status says `RUNNING`, it's time to talk to your model! Since you're on your local laptop, use port-forwarding to create a tunnel to the cluster:

```bash
kubectl port-forward -n your-favorite-namespace svc/rayservice-model-serve-svc 8000:8000
```

Now try hitting it with a request! (Since we used FastAPI, we can just send JSON):

```bash
curl -X POST http://localhost:8000/my-model/ -H "Content-Type: application/json" -d '{"input": 0.8}'
```

🎉 Boom! Production response!

---

## 🤯 The Biggest Production Trap: Pods vs Replicas

Before you deploy real models, you _must_ know the difference between Pods and Replicas. Getting this wrong is the #1 reason why models get stuck in "Pending".

### 1. Logical Resources (Replicas)

When you write `num_cpus: 4` in your Python code (or the `ray_actor_options` YAML), Ray sees this as a **Logical Slot**. It means "I need a slot on a computer that has at least 4 available CPUs."

### 2. Physical Resources (Pods)

When you define `requests: cpu: 8` in the `workerGroupSpecs` YAML, you are asking Kubernetes for an actual, physical box with 8 CPUs.

### 💖 How they work together

If you build a Pod with **8 CPUs**, Ray can fit exactly **two** copies of your **4 CPU** model replica inside that one Pod!

If you accidentally build a Pod with **3 CPUs**, your model replica will wait forever because it can't find a computer big enough to fit its **4 CPU** requirement.

**Golden Rule:**
`Pod CPU >= (Replica CPU * How many replicas you want per Pod) + A little extra for Ray overhead`

```text
Physical Request >= (Sum of Replicas × Logical Request) + System Overhead
```

**Recommended Overhead Buffer:**

- **CPU**: Add 0.5 - 2 CPU cores per pod for Ray system processes.
- **Memory**: Add 1-2 GiB + 30% of object store size.

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

For the exact formulas and definitions of these settings, see the [Configuration Reference](configuration-reference.md#22-autoscaling-settings-what-they-actually-mean).

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
kubectl get rayservice rayservice-model -n [namespace] -o yaml
kubectl describe rayservice rayservice-model -n [namespace]

# Check Ray Serve controller logs (usually shows the root cause)
kubectl logs -n [namespace] -l ray.io/node-type=head --tail=200
```

## Troubleshooting

### Deployment Stuck

**Check RayService status:**

```bash
kubectl describe rayservice rayservice-model -n [namespace]
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
kubectl port-forward -n [namespace] svc/rayservice-model-head-svc 8265:8265

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
kubectl port-forward -n [namespace] svc/rayservice-model-head-svc 8265:8265
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
