# Available Models & Endpoints

This document outlines the models currently running on the Model Service. It provides the necessary details for SDK developers and clients to construct valid requests, including the endpoints, expected input formats, and output formats.

## Overview of Endpoints

All endpoints receive and return data over HTTP using `POST` requests. To minimize network overhead, image payloads and tensor outputs are compressed using **LZ4**.

| Application Name          | Route Prefix             | Model Type                    |
| ------------------------- | ------------------------ | ----------------------------- |
| **Prostate Classifier 1** | `/prostate-classifier-1` | Binary Classification         |
| **Episeg 1**              | `/episeg-1`              | Semantic Segmentation         |
| **Virchow2**              | `/virchow2`              | Foundation Model / Embeddings |
| **Heatmap Builder**       | `/heatmap-builder`       | Pipeline / Custom Builder     |

---

## Model Specifications

### 1. Binary Classifier (`/prostate-classifier-1`)

A binary classification model (e.g., tumor vs. normal tissue).

- **Input**: LZ4-compressed raw bytes of an image (RGB format).
  - The model converts these bytes back to a NumPy array (`uint8`) of shape `(tile_size, tile_size, 3)`.
- **Output**: A single floating-point number representing the classification score.

### 2. Semantic Segmentation (`/episeg-1`)

A semantic segmentation model yielding a prediction map over the input image.

- **Input**: LZ4-compressed raw bytes of an image (`uint8`).
  - Expected layout before compression is `(tile_size, tile_size, 3)`.
- **Output**: LZ4-compressed sequence of bytes representing an `np.float16` NumPy array.
  - The client SDK should decompress this buffer and reconstruct the float16 array.

### 3. Virchow2 (`/virchow2`)

A versatile foundation model (Virchow2) used primarily for generating embeddings or classification features.

- **Input**: LZ4-compressed raw bytes of a tissue tile image (`uint8`, shape `(tile_size, tile_size, 3)`).
- **Output**: Output tensor matching the user's requested specification.
- **Headers**:
  - `x-output-dtype` (optional, default: `float32`): Sets the return precision. Can be `float32` or `float16`.
  - `x-pool-tokens` (optional, default: `true`): If `true`, returns a pooled result (usually `class_token` and `mean` patch tokens). If `false`, returns unpooled raw outputs.

### 4. Heatmap Builder (`/heatmap-builder`)

A processing pipeline element for aggregating inferences into spatial heatmaps.

- **Input**: Typically takes standard HTTP POST requests with localized predictions to stitch into a global heatmap representations.
- **Output**: Heatmap data structure (format depends on the implemented builder logic).

---

## SDK Integration Patterns

The RationAI SDK provides convenient methods for interacting with each model type. All image data is automatically compressed with LZ4 before transmission, and responses are automatically decompressed.

### Using the RationAI Client

Initialize the client and call model methods:

```python
from rationai import Client
from PIL import Image

client = Client()
image = Image.open("tissue_sample.tiff")

# Binary classification
score = client.models.classify_image("prostate-classifier-1", image)
print(f"Classification score: {score}")

# Semantic segmentation
segmentation = client.models.segment_image("episeg-1", image)
print(f"Segmentation shape: {segmentation.shape}")  # (num_classes, height, width)

# Embedding with custom options
embedding = client.models.embed_image(
    "virchow2", 
    image, 
    output_dtype=np.float16,
    pool_tokens=False
)
print(f"Embedding shape: {embedding.shape}")
```
