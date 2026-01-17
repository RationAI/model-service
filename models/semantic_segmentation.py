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


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class SemanticSegmentation:
    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    async def reconfigure(self, config: Config) -> None:
        import importlib

        import onnxruntime as ort

        self.tile_size = config["tile_size"]
        self.mpp = config["mpp"]

        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)

        min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
        opt_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )
        max_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                    "trt_profile_min_shapes": min_shape,
                    "trt_profile_max_shapes": max_shape,
                    "trt_profile_opt_shapes": opt_shape,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self.session = ort.InferenceSession(
            provider(**config["model"]), providers=providers
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
        batch = np.stack(images, axis=0)
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
