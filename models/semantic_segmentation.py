import logging
import os
import time
from typing import Any, TypedDict

import numpy as np
from fastapi import FastAPI, Request, Response
from numpy.typing import NDArray
from ray import serve


class Config(TypedDict):
    tile_size: int
    mpp: float
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float
    intra_op_num_threads: int
    trt_cache_path: str
    trt_max_workspace_size: int
    use_tensorrt: bool


fastapi = FastAPI()
logger = logging.getLogger(__name__)


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class SemanticSegmentation:
    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame
        self.tile_size = 1024  # default, will be overridden by reconfigure

    def reconfigure(self, config: Config) -> None:
        import importlib

        import onnxruntime as ort

        t0 = time.perf_counter()

        self.tile_size = config["tile_size"]
        self.mpp = config["mpp"]
        use_tensorrt = config.get("use_tensorrt", True)

        cache_path = config["trt_cache_path"]
        os.makedirs(cache_path, exist_ok=True)

        min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
        opt_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )
        max_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )

        # TensorRT optimization options:
        # - trt_fp16_enable: Enable FP16 mode for faster inference on Tensor Cores (default: False is slower)
        # - trt_engine_cache_enable: Cache TensorRT engines to disk to avoid rebuilding on restart (default: False rebuilds every time)
        # - trt_engine_cache_path: Directory to store cached engines
        # - trt_timing_cache_enable: Cache kernel timing info to speed up subsequent engine builds (default: False is slower)
        # - trt_builder_optimization_level: Set to 5 for maximum optimization (default: 3, which might miss optimal kernels)
        # - trt_max_workspace_size: Memory available for TensorRT to find optimal kernels (default: 1GB)
        #   Default 1GB is insufficient for high-resolution processing, restricting valid kernels.
        #   We default to 8GB as a reasonable balance, but can be overridden via config.
        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_max_workspace_size": config.get(
                "trt_max_workspace_size", 8 * 1024 * 1024 * 1024
            ),
            "trt_builder_optimization_level": config.get(
                "trt_builder_optimization_level", 3
            ),
            "trt_timing_cache_enable": True,
            "trt_profile_min_shapes": min_shape,
            "trt_profile_max_shapes": max_shape,
            "trt_profile_opt_shapes": opt_shape,
        }

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]

        # Enable all graph optimizations (constant folding, node fusion, etc.) for maximum inference performance.
        # ORT_SEQUENTIAL ensures ops run one at a time within a session, which avoids inter-op parallelism
        # overhead and is preferred when batching is handled externally (as done here via @serve.batch).
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        model_path = provider(**model_config)

        t_model = time.perf_counter()
        logger.info(
            "SemanticSegmentation model provider resolved in %.2fs (use_tensorrt=%s)",
            t_model - t0,
            use_tensorrt,
        )

        if use_tensorrt:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            session_options=sess_options,
        )

        t_session = time.perf_counter()
        logger.info(
            "SemanticSegmentation InferenceSession created in %.2fs (total %.2fs)",
            t_session - t_model,
            t_session - t0,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    def get_config(self) -> dict[str, Any]:
        return {"tile_size": self.tile_size, "mpp": self.mpp}

    @serve.batch
    async def predict(
        self, images: list[NDArray[np.uint8]]
    ) -> list[NDArray[np.float16]]:
        batch = np.stack(images, axis=0, dtype=np.uint8)
        outputs = self.session.run([self.output_name], {self.input_name: batch})

        return list(outputs[0].astype(np.float16))

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        data = self.lz4.decompress(await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        prediction = await self.predict(image.transpose(2, 0, 1))

        return Response(
            content=self.lz4.compress(prediction.tobytes()),
            media_type="application/octet-stream",
        )


app = SemanticSegmentation.bind()  # type: ignore[attr-defined]
