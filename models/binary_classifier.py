import asyncio
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


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BinaryClassifier:
    """Binary classifier for tissue tiles using ONNX Runtime with GPU support."""

    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        """Load the ONNX model and configure inference settings."""
        import importlib

        import onnxruntime as ort

        self.tile_size = config["tile_size"]

        cache_path = "/mnt/cache/trt_cache"
        os.makedirs(cache_path, exist_ok=True)

        min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
        opt_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )
        max_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )

        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4GB
            "trt_builder_optimization_level": 5,
            "trt_timing_cache_enable": True,
            "trt_profile_min_shapes": min_shape,
            "trt_profile_max_shapes": max_shape,
            "trt_profile_opt_shapes": opt_shape,
        }

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        sess_options.inter_op_num_threads = 1

        # Enable graph optimizations
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Load model from provider (e.g., MLflow)
        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        self.session = ort.InferenceSession(
            provider(**config["model"]),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            session_options=sess_options,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Configure batching
        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

        dummy_shape = (config["max_batch_size"], 3, self.tile_size, self.tile_size)
        dummy_input = np.random.randint(0, 256, dummy_shape, dtype=np.uint8)

        self.session.run([self.output_name], {self.input_name: dummy_input})

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        """Run inference on a batch of images."""
        batch = np.ascontiguousarray(np.stack(images, axis=0).astype(np.uint8))

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )

        return outputs[0].flatten().tolist()  # pyright: ignore[reportAttributeAccessIssue]

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        """Handle inference request with LZ4-compressed image."""
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        image = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape(self.tile_size, self.tile_size, 3)
            .transpose(2, 0, 1)
        )

        image = np.ascontiguousarray(image)

        result = await self.predict(image)
        return result


app = BinaryClassifier.bind()  # type: ignore[attr-defined]
