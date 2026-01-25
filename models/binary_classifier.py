from typing import Any, TypedDict

import numpy as np
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve


class Config(TypedDict):
    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float
    intra_op_num_threads: int


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BinaryClassifier:
    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    async def reconfigure(self, config: Config) -> None:
        import importlib

        import onnxruntime as ort

        self.tile_size = config["tile_size"]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        sess_options.inter_op_num_threads = 1

        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        self.session = ort.InferenceSession(
            provider(**config["model"]),
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            session_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        batch = np.stack(images, axis=0).astype(np.uint8)
        outputs = self.session.run([self.output_name], {self.input_name: batch})

        return outputs[0].squeeze(1).tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        data = self.lz4.decompress(await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        return await self.predict(image.transpose(2, 0, 1))


app = BinaryClassifier.bind()  # type: ignore[attr-defined]
