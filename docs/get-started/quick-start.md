# Quick Start

In this tutorial, you will deploy your first model to a Kubernetes cluster.

You will learn:

- How to get the project code.
- How to review and apply the Kustomize configuration.
- How to verify the deployment status.
- How to send requests to your running model.

## Prerequisites

Before you begin, ensure you have the following:

- Access to a Kubernetes cluster with the **KubeRay operator** installed. If you do not have KubeRay installed, follow the [Installation Guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html).
- The `kubectl` command-line tool configured to communicate with your cluster.
- Basic familiarity with Kubernetes concepts such as pods and namespaces.

---

## Step 1: Get the code

Open your terminal and clone the repository:

```bash
git clone https://github.com/RationAI/model-service.git
cd model-service
```

---

## Step 2: Review the Configuration

In Model Service, configurations are managed using **Kustomize**. The environment configuration is split into components inside the `kustomize/` directory.

Models are configured by adding simple YAML definitions into the `kustomize/components/models/models-definitions/` folder.

Let's look at a sample model definition (e.g. `kustomize/components/models/models-definitions/prostate.yaml`):

```yaml
applications:
  - name: prostate-classifier-1
    import_path: models.binary_classifier:app
    route_prefix: /prostate-classifier-1
    # ...
```

**Let's break down the application config:**

- `name`: Logical app name (visible in the Ray dashboard and logs).
- `import_path`: Python entrypoint (`module.path:variable`).
- `route_prefix`: HTTP path under the Serve gateway.

For your first deployment, we will use the existing configuration without changes.

---

## Step 3: Deploy the service

To deploy the service, use the provided deployment script:

```bash
./deploy.sh
```

This script automates the deployment process:

1. It runs `kustomize/components/models/merge_models.py`, which reads all individual model YAMLs from `models-definitions/` and merges them into a single `serve-config-patch.yaml` file.
2. It builds the Kustomize manifests from `kustomize/overlays`.
3. It applies the final manifest to the Kubernetes cluster using `kubectl`.

---

## Step 4: Monitor the deployment

Deploying models takes time as the cluster downloads images and starts worker pods.

Check the overall status of your RayService:

```bash
kubectl get rayservice rayservice-model -n rationai-jobs-ns
```

Check the status of the individual pods:

```bash
kubectl get pods -n rationai-jobs-ns
```

If the pods fail to start, you can inspect the details for troubleshooting:

```bash
kubectl describe rayservice rayservice-model -n rationai-jobs-ns
```

### Note: Using the Ray Dashboard

Ray provides a dashboard for visual monitoring. To access it, forward the port to your local machine:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-head-svc 8265:8265
```

Open a web browser and navigate to `http://127.0.0.1:8265`. Your models are ready when their Serve applications display a **RUNNING** status.

---

## Step 5: Send a request

To communicate with the model from your local machine, forward the Serve port:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-serve-svc 8000:8000
```

You can now send HTTP requests to `http://localhost:8000/prostate-classifier-1`.

---

## Step 6: Clean up

When you are finished, delete the deployment to free up cluster resources:

```bash
kubectl delete -f kustomize/base/ray-service-base.yaml -n rationai-jobs-ns
```

---

## Next Steps

- To deploy your own custom Python model, see [Adding New Models](../guides/adding-models.md).
- To configure scaling or memory settings, read the [Configuration Reference](../guides/configuration-reference.md).
- For a comprehensive walkthrough of a production deployment, see the [Deployment Guide](../guides/deployment-guide.md).
- To understand the internal architecture, explore the [Architecture Overview](../architecture/overview.md).
