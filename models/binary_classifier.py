from typing import Any, TypedDict

import numpy as np
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve


class Config(TypedDict):
    """Configuration for BinaryClassifier deployment."""

    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float
    intra_op_num_threads: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BinaryClassifier:
    """Binary classifier for tissue tiles using ONNX Runtime with GPU support."""

    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame
        self.handle = serve.get_deployment_handle("BinaryClassifier")

    async def reconfigure(self, config: Config) -> None:
        """Load the ONNX model and configure inference settings."""
        import importlib
        import logging

        import onnxruntime as ort

        logger = logging.getLogger("ray.serve")

        self.tile_size = config["tile_size"]

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        sess_options.inter_op_num_threads = 1

        # Load model from provider (e.g., MLflow)
        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        self.session = ort.InferenceSession(
            provider(**config["model"]),
            providers=[
                "TensorrtExecutionProvider",  # Try TensorRT first (fastest)
                "CUDAExecutionProvider",  # Fallback to CUDA
                "CPUExecutionProvider",  # Fallback to CPU
            ],
            session_options=sess_options,
        )

        # Log which provider is actually being used
        active_provider = self.session.get_providers()[0]
        logger.info(f"BinaryClassifier using ExecutionProvider: {active_provider}")
        logger.info(f"Available providers: {self.session.get_providers()}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Configure batching
        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        """Run inference on a batch of images."""
        print("BATCH SIZE:", len(images))

        # Stack images into batch and ensure uint8 dtype
        batch = np.stack(images, axis=0).astype(np.uint8)

        # Run ONNX model inference
        outputs = self.session.run([self.output_name], {self.input_name: batch})

        return outputs[0].squeeze(1).tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        """Handle inference request with LZ4-compressed image."""
        # Decompress LZ4 data
        data = self.lz4.decompress(await request.body())

        # Convert to numpy array (HWC format)
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        # Transpose to CHW format and run prediction
        return await self.handle.predict.remote(image.transpose(2, 0, 1))


app = BinaryClassifier.bind()  # type: ignore[attr-defined]
