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
- `runtime_env`: code location + extra Python deps.

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

Focus on:

- `rayVersion`: must match images you use.
- `workerGroupSpecs[*].{replicas,minReplicas,maxReplicas}`: cluster‑level scaling bounds.
- `resources.requests/limits`: how big each worker pod is.

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

- [Deployment guide](../guides/deployment-guide.md)
- [Architecture overview](../architecture/overview.md)
