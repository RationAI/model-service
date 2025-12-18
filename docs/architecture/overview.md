# Architecture Overview

This document provides an overview of Model Service's architecture, components, and design principles.

## System Architecture

Model Service is built on a multi-layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                  │
│           (API Consumers, Web Apps, Notebooks)          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/HTTPS
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Kubernetes Service                     │
│              (Load Balancer / Ingress)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Ray Serve                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Model A    │  │   Model B    │  │   Model C    │   │
│  │  (Replicas)  │  │  (Replicas)  │  │  (Replicas)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Ray Cluster                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ Head Node  │  │  Worker 1  │  │  Worker 2  │ ...     │
│  └────────────┘  └────────────┘  └────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Ray Cluster

The foundation of Model Service, providing distributed computing infrastructure.

**Head Node:**

- Cluster coordination and management
- Dashboard for monitoring
- Scheduling decisions
- Does not run model workloads (no CPU/GPU assigned)

**Worker Nodes:**

- Execute model inference workloads
- Can be CPU-only or GPU-enabled

- Auto-scale based on demand
- Different worker groups for different hardware types

### 2. Ray Serve

Application layer for serving ML models as HTTP endpoints.

**Features:**

- HTTP request routing
- Load balancing across replicas
- Request batching
- Automatic retry and fault tolerance
- Dynamic model configuration

### 3. KubeRay Operator

Kubernetes operator that manages Ray clusters.

**Responsibilities:**

- Cluster lifecycle management (create, update, delete)
- Autoscaling worker nodes
- Health monitoring
- Configuration reconciliation

### 4. Model Implementations

Your ML models wrapped with Ray Serve decorators.

**Structure:**

```python
@serve.deployment
class YourModel:
    def __init__(self):
        # Model loading

    async def __call__(self, request):
        # Inference logic
```

## Data Flow

### Inference Request Flow

1. **Client Request**: HTTP POST to model endpoint
2. **Service Routing**: Kubernetes service routes to Ray Serve
3. **Load Balancing**: Ray Serve distributes to available replica
4. **Model Processing**: Replica executes inference
5. **Response**: Result returned to client

```
Client → K8s Service → Ray Serve Router → Model Replica → Response
                                       ↓
                                  (Autoscaler)
                                       ↓
                                   Add/Remove
                                    Replicas
```

### Model Loading Flow

1. **Initialization**: Ray Serve creates model replica
2. **Environment Setup**: Install dependencies from runtime_env
3. **Model Download**: Fetch from MLflow/storage
4. **Loading**: Initialize model in memory
5. **Ready**: Replica accepts requests

## Scaling Architecture

### Horizontal Scaling (Replicas)

Models scale horizontally by adding/removing replicas:

```
Load: ████████░░ (80%)
Replicas: [R1] [R2] [R3]

Load: ████████████████ (160%)
Replicas: [R1] [R2] [R3] [R4] [R5] [R6]
```

**Autoscaling Triggers:**

- `target_ongoing_requests`: Target requests per replica
- Scale up when: requests > (replicas × target)
- Scale down when: requests < (replicas × target)

### Vertical Scaling (Workers)

Ray cluster scales by adding/removing worker pods:

```yaml
workerGroupSpecs:
  - groupName: cpu-workers
    minReplicas: 0
    maxReplicas: 4
```

**Triggers:**

- Resource pressure (CPU, memory, GPU)
- Idle timeout (scale to zero)
- Manual scaling

## Resource Management

### CPU Resources

```yaml
ray_actor_options:
  num_cpus: 6 # CPUs per replica

containers:
  resources:
    requests:
      cpu: 12 # CPUs per worker pod
```

**Calculation:**

- Worker pod CPUs ≥ (replicas × num_cpus)
- Leave headroom for system processes

### Memory Resources

```yaml
ray_actor_options:
  memory: 5368709120 # 5 GiB per replica

containers:
  resources:
    limits:
      memory: 10Gi # Memory per worker pod
```

### GPU Resources

```yaml
ray_actor_options:
  num_gpus: 1 # GPUs per replica

nodeSelector:
  nvidia.com/gpu.product: NVIDIA-A40

resources:
  limits:
    nvidia.com/gpu: 1 # GPUs per worker pod
```

## High Availability

### Fault Tolerance

**Ray Cluster:**

- Head node failure → Cluster recreated by KubeRay
- Worker failure → Workload rescheduled to other workers
- Network partition → Automatic reconnection

**Ray Serve:**

- Replica failure → Requests routed to healthy replicas
- Failed replicas automatically restarted
- Graceful shutdown on updates

### Zero-Downtime Updates

Model updates use blue-green deployment:

1. New version deployed alongside old
2. Traffic gradually shifted to new version
3. Old version removed when no active requests

```yaml
spec:
  serveConfigV2: |
    applications:
      - name: my-model-v2  # New version
        # ... new configuration
```

## Security

### Pod Security

All pods run with security constraints:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

### Network Security

- Internal service communication only
- Ingress controls external access
- Proxy support for external dependencies

## Monitoring & Observability

### Ray Dashboard

Web UI for cluster monitoring:

- Resource utilization
- Active tasks
- Node status
- Serve deployments

### Kubernetes Monitoring

Standard Kubernetes tools:

```bash
kubectl get pods -n [namespace]
kubectl top pods -n [namespace]
kubectl logs -n [namespace] <pod-name>
kubectl describe rayservice <rayservice-name> -n [namespace]
```

### Metrics

Ray exports Prometheus metrics:

- Request latency
- Request throughput
- Replica count
- Resource usage

## Design Principles

### 1. Declarative Configuration

Infrastructure defined in YAML, managed by GitOps:

```yaml
apiVersion: ray.io/v1
kind: RayService
# ... configuration
```

### 2. Separation of Concerns

- **Model Code**: Python implementation
- **Infrastructure**: Kubernetes manifests
- **Configuration**: user_config section

### 3. Elastic Scaling

- Scale to zero when idle
- Scale up on demand
- Efficient resource utilization

### 4. Fault Tolerance

- Automatic recovery from failures
- No single point of failure (except data plane)
- Graceful degradation

### 5. Developer Experience

- Simple model implementation
- Easy local testing
- Fast iteration cycle

## Next Steps

- [Deployment guide](../guides/deployment-guide.md)
- [Configuration reference](../reference/configuration-reference.md)
- [Adding new models](../guides/adding-models.md)
