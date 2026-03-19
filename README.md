# Model Service

Model deployment infrastructure for RationAI using Ray Serve on Kubernetes.

This repository contains:

- A KubeRay `RayService` manifest (`ray-service.yaml`) for deploying Ray Serve on Kubernetes.
- Model implementations under `models/` (reference: `models/binary_classifier.py`).
- Documentation under `docs/` (MkDocs).

## Documentation

- MkDocs content: `docs/`
- Key pages:
  - `docs/get-started/quick-start.md`
  - `docs/guides/deployment-guide.md`
  - `docs/guides/adding-models.md`
  - `docs/guides/configuration-reference.md`
  - `docs/guides/troubleshooting.md`
  - `docs/architecture/overview.md`
  - `docs/architecture/request-lifecycle.md`
  - `docs/architecture/queues-and-backpressure.md`
  - `docs/architecture/batching.md`

## Quick Start (Kubernetes)

Full walkthrough: `docs/get-started/quick-start.md`.

### Prerequisites

- Kubernetes cluster with KubeRay operator installed
- `kubectl` configured for the cluster

### Deploy

```bash
kubectl apply -f ray-service.yaml -n [namespace]
kubectl get rayservice rayservice-models -n [namespace]
```

### Access locally

```bash
kubectl port-forward -n [namespace] svc/rayservice-models-serve-svc 8000:8000
```

### Test the reference model (`BinaryClassifier`)

The reference deployment in `ray-service.yaml` exposes an app at the route prefix:

- `/prostate-classifier-1`

`models/binary_classifier.py` expects a **request body that is LZ4-compressed raw bytes** of a single RGB tile:

- dtype: `uint8`
- shape: `(tile_size, tile_size, 3)`
- byte order: row-major (NumPy default)

Example (Python):

```bash
pip install numpy lz4 requests
```

```python
import lz4.frame
import numpy as np
import requests

tile_size = 512  # must match RayService user_config.tile_size
tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

payload = lz4.frame.compress(tile.tobytes())

resp = requests.post(
		"http://localhost:8000/prostate-classifier-1/",
		data=payload,
		headers={"Content-Type": "application/octet-stream"},
		timeout=60,
)
resp.raise_for_status()
print(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)
```

## Repository Structure

```
model-service/
├── models/              # Model implementations
│   └── binary_classifier.py
├── providers/           # Model loading providers
│   └── model_provider.py
├── docs/               # Documentation
├── ray-service.yaml    # Kubernetes RayService configuration
├── pyproject.toml      # Python dependencies
└── README.md
```

## Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/RationAI/model-service/issues)
- **Contact:** RationAI team at Masaryk University

## License

This project is part of the RationAI infrastructure and is available for use by authorized members of the RationAI group.

## Authors

Developed and maintained by the RationAI team at Masaryk University, Faculty of Informatics.
