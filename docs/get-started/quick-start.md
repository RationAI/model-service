# Quick Start

This guide will help you deploy your first model using Model Service in just a few minutes.

## Prerequisites

Before you begin, ensure you have:

- Access to a Kubernetes cluster with KubeRay operator installed
- `kubectl` configured to access your cluster
- Basic familiarity with Kubernetes concepts

Don't have KubeRay installed?
See the [Installation Guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html) for instructions on setting up KubeRay.

## Step 1: Clone the Repository

```bash
git clone https://gitlab.ics.muni.cz/rationai/infrastructure/model-service.git
cd model-service
```

## Step 2: Review the Configuration

The repository includes a sample RayService configuration in `ray-service.yaml`. This deploys a binary classifier model for prostate tissue analysis.

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-models
spec:
  serveConfigV2: |
    applications:
      - name: prostate-classifier-1
        import_path: models.binary_classifier:app
        route_prefix: /prostate-classifier-1
        # ... configuration continues
```

## Step 3: Deploy the Service

Apply the RayService configuration to your cluster.

Replace [namespace] with the desired namespace (e.g., `rationai-notebooks-ns, rationai-jobs-ns` etc.):

```bash
kubectl apply -f ray-service.yaml -n [namespace]
```

## Step 4: Monitor Deployment

Check the deployment status:

```bash
# Check RayService status
kubectl get rayservice rayservice-models -n [namespace]

# Check Ray cluster pods
kubectl get pods -n [namespace]
```

If the RayService is not becoming ready, inspect events and status:

```bash
kubectl describe rayservice rayservice-models -n [namespace]
```

## Step 5: Access the Service Locally

Once deployed, you can port-forward the service to access it locally:

```bash
# Port-forward to access the service locally
kubectl port-forward -n [namespace] svc/rayservice-models-serve-svc 8000:8000
```

## Step 6: Delete the Deployment

To delete the deployed RayService, run:

```bash
kubectl delete -f ray-service.yaml -n [namespace]
```

### Connection Issues

Ensure your cluster has proper network policies and that the namespace has access to required resources (MLflow, proxy, etc.).

## Next Steps

Congratulations! You've successfully deployed your first model with Model Service.

Now you can:

- [Learn how to add your own models](../guides/adding-models.md)
- [Understand the architecture](../architecture/overview.md)
- [Read the deployment guide](../guides/deployment-guide.md)
- [Check configuration reference](../guides/configuration-reference.md)
- [Troubleshooting](../guides/troubleshooting.md)
