# Troubleshooting

This page lists the most common issues when deploying and running models in Model Service (Ray Serve on KubeRay).

## Quick Triage Checklist

Start here before digging deeper:

Use your RayService name from the Helm install (preferably a dedicated test release, e.g. `rayservice-model-<test>`) in the commands below.

```bash
kubectl get rayservice -n rationai-jobs-ns
kubectl describe rayservice <release-name> -n rationai-jobs-ns
kubectl get pods -n rationai-jobs-ns
```

Then inspect logs:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=200
```

## RayService Shows `DEPLOY_FAILED`

### What it usually means

Ray Serve could not start the application or deployment. The root cause is typically visible in the Ray Serve controller logs.

### What to do

1. Describe the RayService for events:

```bash
kubectl describe rayservice <release-name> -n rationai-jobs-ns
```

2. Open the Ray dashboard (helps with Serve deployment errors):

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-head-svc 8265:8265
```

Visit `http://localhost:8265`.

3. Look for Python import errors / missing dependencies:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=500
```

## ImportError / ModuleNotFoundError

### Symptoms

- Serve deployment fails immediately.
- Logs show `ModuleNotFoundError: No module named ...`.

### Causes

- Dependency not installed in the runtime environment.
- Wrong `import_path`.
- `working_dir` does not contain the expected code.

### Fix

- Ensure `import_path` matches your file:
  - Example: `models.binary_classifier:app` means there is `models/binary_classifier.py` defining `app = ...`.
- Add missing dependencies to `runtime_env.pip`.

In this repository, dependencies are typically installed per deployment:

```yaml
deployments:
  - name: BinaryClassifier
    ray_actor_options:
      runtime_env:
        pip: ["onnxruntime>=1.23.2", "mlflow<3.0", "lz4>=4.4.5"]
```

## Worker Crashes (OOMKilled)

### Symptoms

- Pods in `kubectl get pods` show status `OOMKilled` or high restart counts.
- `kubectl describe pod ...` shows "Last State: Terminated (Reason: OOMKilled)".
- Ray Dashboard shows unexpected actor deaths.

### Causes

- The model loaded into memory + the input batch size exceeds the container's memory limit.
- **Physical vs Logical Mismatch**: Ray was told the actor needs 2GB, so it scheduled it on a node, but the actual Python process used 4GB, causing Kubernetes to kill it.

### Fix

You must increase **both** the Ray logical allocation and the Kubernetes physical limit.

1. Increase `ray_actor_options.memory` (Software limit):

   ```yaml
   ray_actor_options:
     memory: 4294967296 # 4 GiB
   ```

2. Increase Kubernetes container limits (Hardware limit):
   Ensure the worker configurations in your Helm values (`helm/rayservice/values.yaml` or relevant worker definitions) provide **more** memory than the sum of all actors on that node plus overhead (~30%).

   ```yaml
   resources:
     limits:
       memory: "6Gi" # Must cover the 4GB actor + Ray overhead
   ```

## Autoscaling Not Working (Replicas Don’t Change)

### Serve replicas not scaling

Check your deployment has autoscaling configured:

```yaml
autoscaling_config:
  min_replicas: 0
  max_replicas: 4
  target_ongoing_requests: 32
```

Also note:

- Scale up/down is not instantaneous (delays and smoothing apply).
- If traffic is low, you may stay at `min_replicas`.

### Worker pods not scaling

Worker pod scaling requires cluster autoscaling enabled:

```yaml
rayClusterConfig:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    idleTimeoutSeconds: 60
```

Also ensure `workerGroupSpecs[*].minReplicas/maxReplicas` allow scaling.

## Not Enough CPU / Memory (Pods Pending)

### Symptoms

- Pods stay in `Pending`.
- Events mention `Insufficient cpu` or `Insufficient memory`.

### Fix

1.  **Check Physical vs Logical**:
    - _Physical_: Can K8s schedule the pod? `kubectl describe pod` will show if nodes are full.
    - _Logical_: Can Ray schedule the actor? Check `ray status` or the dashboard. Ray might say "0/X CPUs available" even if the pod exists, because other actors consumed the slots.

2.  **Adjust Resources**:
    - Reduce per-replica requirements (`ray_actor_options.num_cpus`, `memory`).
    - Increase cluster capacity (maxReplicas) or per-worker limits.

Inspect pod scheduling events:

```bash
kubectl describe pod <pod-name> -n rationai-jobs-ns
```

## MLflow / Artifact Download Problems

### Symptoms

- `mlflow.artifacts.download_artifacts` fails.
- Timeouts during replica initialization.

### Fix

- Ensure `MLFLOW_TRACKING_URI` is set and reachable from the cluster.
- Ensure the cluster has network access (proxy settings if needed).
- Verify the `artifact_uri` exists and permissions are correct.

In your model's Helm YAML definition this is typically configured via `env_vars`:

```yaml
ray_actor_options:
  runtime_env:
    env_vars:
      MLFLOW_TRACKING_URI: http://mlflow.rationai-mlflow:5000
```

## Code Updates Not Applying (Working Dir Cache)

### Symptoms

- You updated your model Python code, pushed it to GitHub, and ran `helm upgrade`, but Ray keeps deploying the old logic or throws errors that were already fixed.

### Cause

Ray downloads the source code defined in `working_dir`: `https://github.com/.../main.zip` and saves it strictly based on the URL string constraint. If the URL hasn't changed, Ray Serve will NOT re-download the archive, effectively clinging to an old snapshot.

### Fix

Append a cache buster query parameter directly to your `working_dir` setup:

```yaml
runtime_env:
  config:
    setup_timeout_seconds: 1800
  working_dir: https://github.com/RationAI/model-service/archive/refs/heads/main.zip?v=1
```

Whenever you push subsequent revisions, just manually bump the `v=1` to `v=2`. During the next Helm deployment process, Ray evaluates the URL as new, retrieves the fresh code zip, and deploys successfully.

## Helpful Commands

```bash
# list Serve and RayService resources
kubectl get rayservice -n rationai-jobs-ns
kubectl get svc -n rationai-jobs-ns

# see all pods for a RayService
kubectl get pods -n rationai-jobs-ns -l ray.io/cluster=<release-name>
```
