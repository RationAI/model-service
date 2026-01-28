import os
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


@serve.deployment(num_replicas="auto", ray_actor_options={"num_gpus": 1})
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

        cache_path = "/tmp/trt_cache"
        os.makedirs(cache_path, exist_ok=True)

        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_max_workspace_size": 4294967296,  # 4GB
        }

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        # Load model from provider (e.g., MLflow)
        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        self.session = ort.InferenceSession(
            provider(**config["model"]),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                ),  # Try TensorRT first (fastest)
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

        logger.info("Starting TensorRT warm-up...")
        dummy_batch = np.zeros(
            (config["max_batch_size"], 3, self.tile_size, self.tile_size),
            dtype=np.uint8,
        )
        self.session.run([self.output_name], {self.input_name: dummy_batch})

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        """Run inference on a batch of images."""
        # Stack images into batch and ensure uint8 dtype
        batch = np.stack(images, axis=0).astype(np.uint8)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )

        return outputs[0].flatten().tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        """Handle inference request with LZ4-compressed image."""
        data = self.lz4.decompress(await request.body())

        # Convert and Transpose (HWC -> CHW)
        image = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape(self.tile_size, self.tile_size, 3)
            .transpose(2, 0, 1)
        )

        # Use ascontiguousarray to ensure the memory layout is optimal for the GPU
        image = np.ascontiguousarray(image)

        return await self.handle.predict.remote(image)


app = BinaryClassifier.bind()  # type: ignore[attr-defined]
