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

When writing functions for the SDK to interact with these models, use the following patterns for data serialization and deserialization.

### Preparing the Input (SDK side)

Data sent to models should be a flat byte buffer compressed with `lz4`.

```python
import lz4.frame
import requests
import numpy as np

def call_model(endpoint_url: str, tile: np.ndarray) -> bytes:
    # 1. Ensure the tile is in the correct format (e.g., uint8)
    tile_bytes = tile.tobytes()

    # 2. Compress the byte buffer
    compressed_payload = lz4.frame.compress(tile_bytes)

    # 3. Send the POST request
    response = requests.post(endpoint_url, data=compressed_payload)
    response.raise_for_status()

    return response.content
```

### Parsing the Output (SDK side)

For models that return raw floats (like `binary_classifier`), standard HTTP responses can be cast. For models returning compressed arrays (like `semantic_segmentation`), you must reverse the process:

```python
def parse_segmentation_response(response_bytes: bytes, shape: tuple) -> np.ndarray:
    # 1. Decompress the response
    decompressed_data = lz4.frame.decompress(response_bytes)

    # 2. Reconstruct the array (e.g., float16)
    array = np.frombuffer(decompressed_data, dtype=np.float16)
    return array.reshape(shape)
```
