# Configuration Reference

This guide provides a reference for configuring Model Service. You will learn the core properties required to scale deployments and manage resources using Kustomize.

For a complete reference of the Ray Service API, you can consult the upstream Ray Serve and KubeRay documentation.

## 1. Anatomy of a Model Service Application

Model Service uses **Kustomize** to manage Kubernetes manifests.

The configuration source of truth is split across two places:

- `kustomize/components/applications/applications-definitions/`: Serve application definitions (routes, import paths, deployments, autoscaling, user config).
- `kustomize/base/ray-service-base.yaml`: RayService base object and cluster-level settings (`rayClusterConfig`).

To define an application, add a YAML file (e.g. `my-model.yaml`) to the `applications-definitions` directory:

```yaml
applications:
  - name: my-model
    import_path: models.my_model:app
    route_prefix: /my-model
```

A python script (`merge_applications.py`) combines all files from `applications-definitions` into `kustomize/components/applications/serve-config-patch.yaml` when executed via `./deploy.sh`.

`serve-config-patch.yaml` is generated, so you should not edit it manually.

This file instructs KubeRay on:

- **Serving definitions**: Your Python execution endpoints, autoscaling bounds, and route paths.
- **Resource requests**: Cluster node capacities and compute allocation (CPUs/GPUs).

---

## 2. Configuring Application Endpoints

An application is the interface clients use to communicate with your deployment. Let's configure one in the `applications-definitions/` directory:

```yaml
applications:
  - name: prostate-classifier
    import_path: models.binary_classifier:app
    route_prefix: /prostate-classifier
    runtime_env:
      working_dir: https://github.com/RationAI/model-service/archive/refs/heads/feature/my-new-model.zip
      pip:
        - onnxruntime>=1.23.2
```

**Property Breakdown:**

- `name`: The logical name of the application, used in the Ray dashboard and logs.
- `import_path`: The path to the Python entrypoint, structured as `module.path:variable`.
- `route_prefix`: The HTTP path root used for the Serve gateway.
- `runtime_env`: Configuration for dynamically loading dependent code and libraries. (See [RayService Configuration](../guides/adding-models.md#rayservice-configuration)).

### Regulating Scale with Deployments

Inside an application, "deployments" define computational capacity and scaling limits.

```yaml
deployments:
  - name: BinaryClassifier
    max_ongoing_requests: 64
    max_queued_requests: 128
    autoscaling_config:
      min_replicas: 0
      max_replicas: 4
      target_ongoing_requests: 32
    ray_actor_options:
      num_cpus: 6
      memory: 5368709120 # 5 GiB
    user_config:
      tile_size: 512
      threshold: 0.5
```

- `autoscaling_config`: Dictates how the system automatically scales worker replicas in response to incoming traffic.
- `ray_actor_options`: Specifies computation requirements per replica, such as CPUs or memory.
- `user_config`: Key-value properties passed directly into your Python model's `reconfigure()` function.

---

## 2.1 Backpressure and Concurrency

To ensure stability under heavy load, Model Service provides variables to handle backpressure. Using these variables correctly prevents models from crashing.

### `max_ongoing_requests`

**Definition:** The maximum number of requests a single replica can process concurrently.
**Usage:** Configure this variable based on the memory overhead per request. Exceeding available memory will result in Pod crashes. Requests that exceed this threshold wait in an input queue unless `max_queued_requests` is reached.

### `max_queued_requests`

**Definition:** The maximum total queue size representing requests waiting for an available replica.
**Usage:** When this limit is reached, incoming requests are rejected to prevent system-wide degradation.

> **Note:** Set `max_ongoing_requests` according to your application's per-request footprint, and configure `max_queued_requests` based on your desired trade-off between wait times and immediate error feedback.

---

## 2.2 Autoscaling Strategies

Autoscaling adds replicas during traffic spikes and decreases replicas during idle periods.

### Property: `target_ongoing_requests`

The target defines the desired number of concurrent requests per replica.

**Mechanism:**
The autoscaler attempts to adjust the number of active replicas to keep the average number of requests per replica matched to this value.
If `target_ongoing_requests` is `20`, and there are `100` incoming concurrent requests, the system scales up to `5` replicas automatically.

- **For aggressive scale-up**, select a lower target.
- **For high-throughput models**, increase the target.

### Properties: `min_replicas` and `max_replicas`

- **Scale to Zero (`min_replicas: 0`)**: Retains no instances of the container structure when there is zero traffic. Useful for saving resource costs. Note that zero-scale triggers a _cold start_ penalty (delay) on the next request.
- **Always Active (`min_replicas: 1` or higher)**: Keeps minimal instances permanently initialized to serve traffic without cold starts.

---

## 3. Configuring the Cluster Resources

Cluster-level configuration has moved to `kustomize/base/ray-service-base.yaml`.

<span style="color:#b00020; font-weight:700;">Before test deployment, change <code>metadata.name</code> in <code>kustomize/base/ray-service-base.yaml</code> to a unique test name (for example <code>rayservice-model-my-model</code>).</span>

This includes:

- Ray version and autoscaler options
- head/worker group templates
- CPU, memory, security context, and other Pod-level resource settings

Your application/deployment-level settings (for example `autoscaling_config`, `ray_actor_options`, and `user_config`) remain in `applications-definitions/*.yaml`.

Example cluster configuration from `ray-service-base.yaml`:

```yaml
rayClusterConfig:
  rayVersion: "2.53.0"
  headGroupSpec:
    rayStartParams:
      num-cpus: "0"
    template:
      spec:
        containers:
          - name: ray-head
  workerGroupSpecs:
    - groupName: cpu-workers
      replicas: 1
      template:
        spec:
          containers:
            - name: ray-worker
              resources:
                requests:
                  cpu: "4"
                  memory: "8Gi"
                limits:
                  cpu: "8"
                  memory: "16Gi"
```

### Pitfall: Logical vs. Physical Scaling Defaults

It is crucial to verify that the target K8s Pod (`resources.requests`) fulfills or exceeds the memory specified in the application deployment options (`ray_actor_options`).

- If the Pod `cpu` request is `4`.
- If the model replica `num_cpus` specifies `2`.
- You can run **2** running replicas per Pod.

If a deployment requests `5` CPUs while maximum Pod resources represent `4` CPUs, the system cannot schedule the replica, leaving the replica stuck in a `Pending` state.

---

## 4. Keeping Things Secure

If you want to be a good citizen in your cluster, add these security settings to your `workerGroupSpec`:

```yaml
template:
  spec:
    securityContext:
      runAsNonRoot: true
    containers:
      - name: ray-worker
        securityContext:
          allowPrivilegeEscalation: false
```

This just ensures your code runs safely without root access!

## 5. Putting It Together (Small Example)

In the current setup, you typically edit application definitions and base cluster settings, then run `./deploy.sh`.
The `serveConfigV2` patch is generated automatically.

```yaml
# kustomize/components/applications/applications-definitions/my-classifier.yaml
applications:
  - name: my-classifier
    import_path: models.classifier:app
    route_prefix: /classify
    deployments:
      - name: Classifier
        autoscaling_config:
          min_replicas: 1
          max_replicas: 5
          target_ongoing_requests: 32
        ray_actor_options:
          num_cpus: 4
```

```yaml
# kustomize/base/ray-service-base.yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-model-my-model # CHANGE THIS BEFORE DEPLOYMENT
spec:
  serveConfigV2: "" # Patched by generated serve-config-patch.yaml
  rayClusterConfig:
    rayVersion: 2.53.0
    enableInTreeAutoscaling: true
    # ... headGroupSpec / workerGroupSpecs
```

## Next Steps

- [Deployment guide](deployment-guide.md)
- [Architecture overview](../architecture/overview.md)
