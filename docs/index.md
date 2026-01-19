# Model Service Documentation

Welcome to the Model Service documentation. This service provides a scalable, production-ready infrastructure for deploying machine learning models for the RationAI project using Ray Serve on Kubernetes.

## What is Model Service?

Model Service is a deployment framework that enables:

- **Scalable Model Serving**: Automatically scale model replicas based on request load
- **Distributed Inference**: Distribute inference workloads across multiple workers and nodes
- **Resource Management**: Efficiently manage CPU and GPU resources in Kubernetes
- **Model Versioning**: Integration with MLflow for model lifecycle management
- **Production-oriented**: Built on Ray Serve and KubeRay, with autoscaling and failure recovery features

## Key Features

### Auto-Scaling

Model Service automatically adjusts the number of model replicas based on incoming request volume, ensuring optimal resource utilization and response times.

### Multi-Model Deployment

Deploy multiple models simultaneously with isolated resource allocations and independent scaling policies.

### GPU/CPU Support

Flexible resource allocation supporting both CPU-based and GPU-accelerated models with hardware-specific worker groups.

### Kubernetes Native

Leverages KubeRay operator for seamless integration with Kubernetes, enabling declarative configuration and GitOps workflows.

## Why Ray Serve?

Model Service is built on top of Ray Serve because it combines a simple developer experience with strong production capabilities:

- **Unified batch and online inference**: The same Ray cluster can handle real-time HTTP requests and large batch jobs, which matches RationAI's mix of interactive and offline pathology workloads.
- **Python‑native API**: Models are implemented as regular Python classes or functions with decorators, making it easy for researchers to contribute without learning a heavy framework.
- **Autoscaling built in**: Ray Serve natively scales replicas based on request pressure and integrates with Ray's cluster autoscaler to add/remove worker pods.
- **Multi‑model support**: Multiple independent applications and deployments can run side‑by‑side on one cluster while isolating resources per model.

Alternative approaches (plain Kubernetes deployments, custom Flask/FastAPI services, or specialized serving stacks like TorchServe or TF Serving) either lack first‑class autoscaling orchestration across many models, or are tightly coupled to specific ML frameworks. Ray Serve, together with KubeRay, lets us:

- Express all infrastructure declaratively in a single `RayService` resource.
- Share the same cluster across heterogeneous models and hardware (CPU/GPU).
- Keep the operational surface smaller by relying on one general‑purpose serving layer instead of many ad‑hoc microservices.

## Use Cases

Model Service is designed for:

- **Pathology Image Analysis**: Deploy models for tissue classification, nuclei detection, and other pathology tasks
- **Batch Processing**: Handle large-scale inference workloads efficiently
- **Real-time Inference**: Serve predictions with low latency for interactive applications
- **Research Experiments**: Quickly deploy and test new model versions

## Documentation Contents

### Get Started

- [**Quick Start**](get-started/quick-start.md): Deploy the reference empty model in minutes.

### Guides

- [**Adding Models**](guides/adding-models.md): How to write, package, and integrate your own Python models.
- [**Deployment Guide**](guides/deployment-guide.md): Production checklist, resource planning (CPU/GPU), and networking.
- [**Configuration Reference**](guides/configuration-reference.md): Detailed explanation of `ray-service.yaml` settings.
- [**Troubleshooting**](guides/troubleshooting.md): Common errors (OOM, hang scenarios) and solutions.

### Architecture Deep Dive

- [**Overview**](architecture/overview.md): High-level system design and component hierarchy.
- [**Request Lifecycle**](architecture/request-lifecycle.md): Trace a request from Ingress to Worker.
- [**Queues & Backpressure**](architecture/queues-and-backpressure.md): Understanding flow control and overload protection.
- [**Batching**](architecture/batching.md): How request coalescing works under the hood.

## Getting Help

- **Documentation**: Browse the guides and reference materials in this documentation
- **Issues**: Report bugs or request features via [GitLab Issues](https://gitlab.ics.muni.cz/rationai/infrastructure/model-service/-/issues)
- **Contact**: Reach out to the RationAI team at Masaryk University

## Next Steps

Ready to get started? Follow our [Quick Start Guide](get-started/quick-start.md) to deploy your first model.

## Glossary

- **RayService**: KubeRay custom resource that manages a Ray cluster plus a Ray Serve application, including updates.
- **Deployment (Ray Serve)**: A scalable unit (replicas) that runs your model code.
- **Replica**: One running instance of a deployment.
- **Worker group (KubeRay)**: A set of Ray worker pods (e.g., CPU or GPU workers) with independent scaling bounds.
