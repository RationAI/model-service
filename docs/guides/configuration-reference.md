# Configuration Reference

This guide documents the current Helm-based configuration model for Model Service.

## 1. Configuration Layout

Model Service uses Helm chart files in `helm/rayservice/`:

- `helm/rayservice/applications/`: Ray Serve applications (routes, import paths, deployments, autoscaling, user config).
- `helm/rayservice/values.yaml`: Chart values and shared RayService settings.
- `helm/rayservice/workers/`: Worker group definitions (CPU/GPU pools).
- `helm/rayservice/templates/rayservice.yaml`: Template that renders RayService with combined `serveConfigV2`.

## 2. Application Definition

Each file in `helm/rayservice/applications/` defines one Serve application:

```yaml
- name: prostate-classifier-1
  import_path: models.binary_classifier:app
  route_prefix: /prostate-classifier-1
  runtime_env:
    working_dir: https://github.com/RationAI/model-service/archive/refs/heads/main.zip
  deployments:
    - name: BinaryClassifier
      max_ongoing_requests: 512
      max_queued_requests: 1024
      autoscaling_config:
        min_replicas: 0
        max_replicas: 4
        target_ongoing_requests: 128
      ray_actor_options:
        num_cpus: 4
        num_gpus: 1
        memory: 12884901888
      user_config:
        tile_size: 512
        max_batch_size: 256
        batch_wait_timeout_s: 0.05
```

Field summary:

- `name`: Logical Ray Serve application name.
- `import_path`: Python entrypoint in format `module.path:variable`.
- `route_prefix`: Public HTTP prefix for the app.
- `runtime_env`: Source code and runtime dependency configuration.
- `deployments`: One or more Serve deployments with scaling/resource settings.

## 3. Deployment-Level Tuning

Main knobs inside `deployments`:

- `max_ongoing_requests`: Max concurrent requests per replica.
- `max_queued_requests`: Backpressure queue limit.
- `autoscaling_config`: Min/max replicas and scale target.
- `ray_actor_options`: CPU/GPU/memory reservation per replica.
- `user_config`: Dynamic model settings passed into `reconfigure()`.

Example autoscaling block:

```yaml
autoscaling_config:
  min_replicas: 0
  max_replicas: 4
  target_ongoing_requests: 32
```

### Autoscaling Strategies

In addition to `min_replicas`, `max_replicas`, and `target_ongoing_requests`, Ray Serve supports stabilization knobs:

```yaml
autoscaling_config:
  min_replicas: 1
  max_replicas: 10
  target_ongoing_requests: 20
  upscale_delay_s: 30
  downscale_delay_s: 600
```

Practical guidance:

- Lower `target_ongoing_requests` for latency-sensitive or compute-heavy models.
- Keep `upscale_delay_s` relatively low to react faster to spikes.
- Keep `downscale_delay_s` higher to avoid frequent scale up/down oscillation.

## 4. Cluster and Worker Resources

Worker groups are defined under `helm/rayservice/workers/` and referenced by chart values.

Example worker group:

```yaml
- groupName: cpu-workers
  replicas: 1
  minReplicas: 0
  maxReplicas: 10
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

Sizing rule of thumb:

- Physical pod limits in Kubernetes must exceed total logical actor reservations plus Ray overhead.

## 5. Deploy Command

Deploy or update with Helm:

```bash
helm upgrade --install rayservice-model helm/rayservice -n rationai-jobs-ns
```

In this command, `rayservice-model` is the Helm release name parameter. You can change it (for example `rayservice-model-my-model`) to run isolated test releases.

## 6. Working Directory Cache Note

Ray caches `runtime_env.working_dir` by URL string. If code was updated but URL is unchanged, an older cached snapshot may be reused.

Cache-busting example:

```yaml
runtime_env:
  working_dir: https://github.com/RationAI/model-service/archive/refs/heads/main.zip?v=2
```

For troubleshooting details, see [Troubleshooting](troubleshooting.md).
