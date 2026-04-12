# Quick Start

Welcome to Model Service! 🎉 In this tutorial, you'll learn how to deploy your very first AI model to a Kubernetes cluster. We'll do this step-by-step, and by the end, you'll have a live, running model service!

You will learn:

- How to grab the code you need.
- How to read and apply the configuration.
- How to check if your model is running.
- How to talk to your new model!

## Before we start

You'll need a few things set up before we can play:

- Access to a Kubernetes cluster that has the **KubeRay operator** installed.
- The `kubectl` command-line tool, configured to talk to your cluster.
- A basic understanding of what a Kubernetes "pod" or "namespace" is.

> **Wait, I don't have KubeRay!**
> No worries! Check out the [Installation Guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html) to get that set up first, then come right back.

---

## Step 1: Get the code

First things first, let's get the code onto your machine. Open up your terminal and clone our repository:

```bash
git clone https://github.com/RationAI/model-service.git
cd model-service
```

Great! You're now inside the project folder.

---

## Step 2: Meet the `ray-service.yaml` file

In Kubernetes, we declare what we want using YAML files. Open up `ray-service.yaml` in your favorite code editor.

This file is the magic recipe that tells the cluster how to run your models. By default, it's set up to run a few apps like `prostate-classifier-1`, `semantic-segmentation`, and `heatmap-builder`.

Take a look at this snippet:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-model
spec:
  serveConfigV2: |
    applications:
      - name: prostate-classifier-1
        import_path: models.binary_classifier:app
        route_prefix: /prostate-classifier-1
```

**What's happening here?**

- `name`: Just a friendly name for your app!
- `import_path`: Tells the system exactly where your Python code lives.
- `route_prefix`: The URL path you'll use to access this specific model.

For your first deployment, you don't need to change a thing. Let's just use it as is!

---

## Step 3: Deploy it! 🚀

Now it's time to send this recipe to the cluster. Choose a namespace (like `rationai-jobs-ns`) and run:

```bash
kubectl apply -f ray-service.yaml -n rationai-jobs-ns
```

_Whoosh!_ Your cluster is now reading the configuration and spinning up your models.

---

## Step 4: Watch it come to life

Deploying models can take a minute or two as it downloads images and starts up workers. Let's see how it's doing.

Check the overall status of your RayService:

```bash
kubectl get rayservice rayservice-model -n rationai-jobs-ns
```

Check on the individual pods (the tiny computers running your code):

```bash
kubectl get pods -n rationai-jobs-ns
```

If things aren't starting up, you can dig into the details:

```bash
kubectl describe rayservice rayservice-model -n rationai-jobs-ns
```

### The Ray Dashboard (Highly Recommended!)

Ray comes with a beautiful dashboard where you can see exactly what's going on visually. Let's open it up!

Open a new terminal window and run:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-head-svc 8265:8265
```

Now, open your web browser and go to `http://127.0.0.1:8265`.
Look for your Serve applications—once they say **RUNNING**, you're good to go!

---

## Step 5: Say hello to your model

Your model is running inside the cluster, but we want to talk to it from your local laptop. Let's create a tunnel (port-forward) to the model service:

```bash
kubectl port-forward -n rationai-jobs-ns svc/rayservice-model-serve-svc 8000:8000
```

Now you can send HTTP requests to `http://localhost:8000/prostate-classifier-1`!

---

## Step 6: Cleaning up

When you're done playing and want to shut everything down to save resources, it's as simple as:

```bash
kubectl delete -f ray-service.yaml -n rationai-jobs-ns
```

---

## What's next?

🎉 You did it! You successfully deployed an AI model.

Where to go from here?

- Want to deploy your own custom Python model? Check out [Adding Your Own Models](../guides/adding-models.md).
- Want to tweak the scaling or memory settings? Read the [Configuration Reference](../guides/configuration-reference.md).
- Deploying for real? Read the [Deployment Guide](../guides/deployment-guide.md).
- Curious how this all works under the hood? Explore the [Architecture Overview](../architecture/overview.md).
