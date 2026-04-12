# Configuration Reference

Welcome to the control room! 🎛️ Here, you'll learn how to tweak Model Service to perfectly fit your needs. Don't worry, we'll cover the **most important knobs** so you don't get overwhelmed.

If you ever need the super-detailed, nitty-gritty API details, you can always check out the upstream Ray Serve and KubeRay documentation. But for 99% of your work, this page is all you need!

## 1. The Anatomy of a RayService

Let's look at the basic skeleton of your configuration file:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: <service-name>
  namespace: [namespace]
spec:
  serveConfigV2: |
    # Your Apps go here!
  rayClusterConfig:
    # Your Cluster settings go here!
```

Think of this file in two main parts:

- **`serveConfigV2`**: **What** you are serving. (Your Python apps, autoscaling rules, API routes).
- **`rayClusterConfig`**: **Where** it runs. (How big the computers are, how many CPUs/GPUs you need).

---

## 2. Setting Up Your Applications

An application is the endpoint clients will talk to. Let's configure one!

```yaml
serveConfigV2: |
  applications:
    - name: prostate-classifier
      import_path: models.binary_classifier:app
      route_prefix: /prostate-classifier
      runtime_env:
        working_dir: https://.../model-service-main.zip
        pip:
          - onnxruntime>=1.23.2
```

**Let's break that down:**

- `name`: A friendly name you'll see in the dashboard.
- `import_path`: Where is your Python code? Tell it the module and variable (like `models.binary_classifier:app`).
- `route_prefix`: The URL magic. If you set this to `/prostate-classifier`, requests go to `http://<serve-host>:8000/prostate-classifier/...`.
- `runtime_env`: What does your code need to run? You can specify a zip file of your code (`working_dir`) and Python packages (`pip`).

### Deployments: Controlling Scale and Power

Inside your application, you have "deployments". This is where you tell the system how much power each model needs.

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

- `autoscaling_config`: Should we scale up automatically when busy? Here's where you decide!
- `ray_actor_options`: How beefy is this model? Does it need 6 CPUs? 5 GiB of RAM?
- `user_config`: A cool feature that lets you pass custom settings right into your Python code's `reconfigure()` method on the fly!

---

## 2.1 The Art of Backpressure (Super Important!)

Let's talk about traffic jams. What happens when too many requests hit your model at once? You have two important tools to manage this:

### `max_ongoing_requests` (The Door Coder)

**What it is:** How many requests a _single_ replica is allowed to work on simultaneously.
**Why you care:** If your model takes a lot of memory per request, setting this too high will crash your pod with an Out of Memory error! If a request tries to enter but the limit is reached, it waits in line.

### `max_queued_requests` (The Bouncer)

**What it is:** The size of the waiting line outside the club.
**Why you care:** If the line gets completely full, the Bouncer starts turning people away (rejecting requests). This is great because it prevents your whole system from slowing to a crawl.

> **Pro Tip:** Keep `max_ongoing_requests` safe for your replica's memory, and use `max_queued_requests` to decide whether you'd rather clients wait, or get a quick "too busy" error.

---

## 2.2 Mastering Autoscaling

How does the system know when to add more replicas?

### The Magic Number: `target_ongoing_requests`

This is the most important scaling setting. It means: "Try to keep the average number of active requests per replica close to this number."

**How it works:**
If `target_ongoing_requests` is `20`, and you suddenly get `100` concurrent requests, the system will look at the math (`100 / 20 = 5`) and quickly spin up `5` replicas for you!

- **Want it to scale up fast?** Set this lower.
- **Can your model handle a ton of traffic easily?** Set this higher!

### `min_replicas` and `max_replicas`

- **Scale to Zero (`min: 0`)**: Great for saving money! If no one is using the model, it shuts down completely. Just know that the _next_ person to call it will have to wait a few seconds for it to start up again (a "cold start").
- **Always Ready (`min: 1` or more)**: The model is always running, ready to answer instantly.

---

## 3. Configuring the Ray Cluster (The Hardware)

Now let's talk about the actual Kubernetes Pods that will run your code.

```yaml
rayClusterConfig:
  rayVersion: "2.52.1"
  headGroupSpec:
    rayStartParams:
      num-cpus: "0" # The head node is just the manager!
    template:
      spec:
        containers:
          - name: ray-head
          # ... image details
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

### The Golden Rule of Worker Sizing

Here is the secret to avoiding headaches:

**Your K8s Pod (worker `resources.requests`) MUST be bigger than your Model Requirement (`ray_actor_options`).**

- **Physical World:** Your Pod has `4` CPUs.
- **Logical World:** Your model asks for `2` CPUs.
- **Result:** You can fit exactly `2` copies (replicas) of your model on that one Pod!

If your model asks for `5` CPUs but your Pod only gives `4`, your model will sit in `Pending` forever, waiting for a bigger computer that will never arrive.

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
