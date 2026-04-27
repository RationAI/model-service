# Model Service Documentation

## Overview

Model Service is a Helm based deployment framework for serving ML models on Kubernetes with Ray Serve.

Problem solved:
turn Python model code into stable HTTP endpoints that can batch requests, scale replicas, and run on CPU or GPU workers.

Use this documentation when:

- you want to deploy a model as an API endpoint,
- you need to tune throughput and latency,
- or you operate multi-model workloads on one Ray cluster.

## What You Get

- Ray Serve applications managed by a Helm chart in `helm/rayservice/`.
- Per-model configuration in `helm/rayservice/applications/`.
- Default worker profiles in `helm/rayservice/workers/`.
- Operational guides for deployment, scaling, and troubleshooting.

## Start Here

### New user path

1. [Quick Start](get-started/quick-start.md)
2. [Deployment Guide](guides/deployment-guide.md)
3. [Troubleshooting](guides/troubleshooting.md)

### Model author path

1. [Adding New Models](guides/adding-models.md)
2. [Configuration Reference](guides/configuration-reference.md)
3. [Deployment Guide](guides/deployment-guide.md)

### Operator path

1. [Configuration Reference](guides/configuration-reference.md)
2. [Architecture Overview](architecture/overview.md)
3. [Troubleshooting](guides/troubleshooting.md)

## Documentation Map

### Getting Started

- [Quick Start](get-started/quick-start.md): first deployment using a dedicated test release name.

### Guides

- [Adding New Models](guides/adding-models.md): implement model code and bind routes.
- [Deployment Guide](guides/deployment-guide.md): safe deployment workflow and production checklist.
- [Configuration Reference](guides/configuration-reference.md): source of truth for Helm and Ray Serve settings.
- [Troubleshooting](guides/troubleshooting.md): diagnostics for deployment and runtime failures.

### Architecture

- [Overview](architecture/overview.md): system components and scaling model.
- [Request Lifecycle](architecture/request-lifecycle.md): end-to-end request flow.
- [Queues and Backpressure](architecture/queues-and-backpressure.md): overload behavior and queue tuning.
- [Batching](architecture/batching.md): how request coalescing works per replica.

## Important Notes

- Use a dedicated test release name (for example `rayservice-model-<test>`) while experimenting.
- Keep default worker profiles unless you need specific hardware or scheduling behavior.
- Tune application/deployment settings first, worker templates second.

## Glossary

- RayService: KubeRay custom resource managing a Ray cluster plus Serve applications.
- Deployment (Ray Serve): scalable unit running one part of model code.
- Replica: one running instance of a deployment.
- Worker group: pool of Ray worker pods with its own resource profile.

## Related Links

- [GitHub repository](https://github.com/RationAI/model-service)
- [KubeRay installation guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html)
- [Ray Serve on Kubernetes](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html)
