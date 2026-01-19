# Configuration Reference

This page summarizes the **most important knobs** you will touch when configuring Model Service. For full API details, see the upstream Ray Serve and KubeRay documentation.

## 1. RayService Skeleton

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: <service-name>
  namespace: [namespace]
spec:
  serveConfigV2: |
    # Ray Serve applications
  rayClusterConfig:
    # Ray cluster (head + workers)
```

Think of it as two parts:

- **`serveConfigV2`**: what you serve (apps, deployments, autoscaling).
- **`rayClusterConfig`**: where it runs (Ray version, worker groups, resources).

## 2. Applications and Deployments

### Applications (HTTP endpoints)

```yaml
serveConfigV2: |
  applications:
    - name: prostate-classifier
      import_path: models.binary_classifier:app
      route_prefix: /prostate-classifier
      runtime_env:
        working_dir: https://.../model-service-master.zip
        pip:
          - onnxruntime>=1.23.2
```

- `name`: logical app name (used in Ray dashboard/logs).
- `import_path`: Python entrypoint (`module.path:variable`).
- `route_prefix`: HTTP path under the Serve gateway.
- `runtime_env`: dynamic environment setup (see [Managing Dependencies](../guides/adding-models.md#6-managing-dependencies)).

### Deployments (scaling + resources)

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

- `autoscaling_config`: how many replicas and when to scale.
- `ray_actor_options`: per‑replica CPU/GPU/memory.
- `user_config`: free‑form dict passed to `reconfigure()` in your model.

## 2.1 Backpressure and queueing settings (very important)

These two knobs often get confused because they both “limit load”, but they act at different points in the request path.

### `max_ongoing_requests` (replica-side concurrency)

**What it is:** the maximum number of in-flight requests a _single replica_ is allowed to have at once.

**What it controls:** per-replica concurrency and memory pressure.

**What happens when exceeded:** requests should not be dispatched onto that replica; they must wait upstream (typically in the caller-side queue).

### `max_queued_requests` (caller-side queue limit)

**What it is:** the maximum number of requests that are allowed to wait in the caller-side queue _before_ a replica slot is available.

**Where that queue lives:** in the component that is calling the deployment (commonly the HTTP Proxy when handling HTTP ingress).

**What happens when exceeded:** requests are rejected (client-visible overload/backpressure).

### Why the difference matters

- `max_ongoing_requests` protects the replica from being overloaded.
- `max_queued_requests` decides whether you prefer waiting or rejecting during spikes.

See: [Queues and Backpressure](../architecture/queues-and-backpressure.md).

## 2.2 Autoscaling settings (what they actually mean)

### `target_ongoing_requests`

**What it is:** The desired average number of **ongoing (in-flight)** requests per replica. This is the **primary scaling driver**.

**Formula:**
$$ \text{Desired Replicas} = \left\lceil \frac{\text{Total Ongoing Requests}}{\text{target\_ongoing\_requests}} \right\rceil $$

**Note:** "Total Ongoing Requests" refers to the **concurrency** (number of requests currently being processed or waiting in the queue), _not_ the Requests Per Second (RPS).

**Example:**
If your system receives 100 **concurrent** requests and `target_ongoing_requests` is set to 20, Serve will scale to 5 replicas.

**How it influences scaling:**

- **Lower value**: Scales up _earlier_. Use for latency-sensitive models or heavy tasks.
- **Higher value**: Scales up _later_. Use for high-throughput models where a single replica can handle many concurrent requests.

**Important interaction:** if you set `max_queued_requests` too low, requests may get rejected before ongoing requests rise enough for autoscaling to catch up.

### `min_replicas` / `max_replicas`

Hard bounds on how many replicas Serve is allowed to run for that deployment.

- **Scale to Zero**: Set `min_replicas: 0` to allow the deployment to stop all replicas when idle. The first request will trigger a "cold start" (latency spike).
- **High Availability**: Set `min_replicas: 2` (or more) to ensure at least two copies are always running, even if idle.

### `upscale_delay_s` / `downscale_delay_s`

Rules for how quickly the autoscaler reacts to load changes.

- **`upscale_delay_s`**: The "patience" period before scaling up. The autoscaler sees high load, but waits this many seconds to confirm the spike is real before launching new replicas.
  - _Risk_: Setting this too high makes the system sluggish to react to bursts.
- **`downscale_delay_s`**: The "grace period" before scaling down. Even if load drops to zero, the autoscaler keeps replicas alive for this duration.
  - _Recommendation_: Keep this high to avoid "thrashing" (rapidly creating/destroying replicas) during short pauses in traffic.

## 3. Ray Cluster (Workers and Autoscaling)

```yaml
rayClusterConfig:
  rayVersion: "2.52.1"
  enableInTreeAutoscaling: true
  headGroupSpec:
    rayStartParams:
      num-cpus: "0" # head only coordinates
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.52.1-py312
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
                requests:
                  cpu: "4"
                  memory: "8Gi"
                limits:
                  cpu: "8"
                  memory: "16Gi"
```

**Key Interactions:**

1.  **Head Node Isolation**: `rayStartParams: { num-cpus: "0" }` on the head node prevents workloads from scheduling there. The head is reserved for the Control Plane.
2.  **Worker Sizing**: `resources.requests` defines the physical guarantee. Your Pod must be bigger than your Replica (`ray_actor_options`).
    - _Physical_: Pod Requests (e.g., 4 CPU)
    - _Logical_: Model Replica Requirement (e.g., 2 CPU)
    - _Result_: One Pod can fit 2 Replicas (plus overhead).

## 4. Security and Placement (Optional but Recommended)

```yaml
template:
  spec:
    securityContext:
      runAsNonRoot: true
      fsGroupChangePolicy: OnRootMismatch
      seccompProfile:
        type: RuntimeDefault
    nodeSelector:
      nvidia.com/gpu.product: NVIDIA-A40
    containers:
      - name: ray-worker
        securityContext:
          allowPrivilegeEscalation: false
          runAsUser: 1000
          capabilities:
            drop: ["ALL"]
```

Use these to:

- Enforce non‑root containers and least privilege.
- Pin GPU workloads to specific node types.

## 5. Putting It Together (Small Example)

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-example
  namespace: rationai-notebooks-ns
spec:
  serveConfigV2: |
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
  rayClusterConfig:
    rayVersion: "2.52.1"
    enableInTreeAutoscaling: true
    headGroupSpec:
      rayStartParams:
        num-cpus: "0"
    workerGroupSpecs:
      - groupName: cpu-workers
        minReplicas: 1
        maxReplicas: 5
```

## Next Steps

- [Deployment guide](deployment-guide.md)
- [Architecture overview](../architecture/overview.md)
