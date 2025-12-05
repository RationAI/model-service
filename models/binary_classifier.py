from typing import Any, TypedDict

import numpy as np
from fastapi import FastAPI, Request
from ray import serve


class Config(TypedDict):
    tile_size: int
    mean: list[float]
    std: list[float]
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

        self.decompress = lz4.frame.decompress

    async def reconfigure(self, config: Config) -> None:
        import importlib

        import onnxruntime as ort

        self.tile_size = config["tile_size"]

        self.mean = np.array(config["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.inv_std = 1 / np.array(config["std"], dtype=np.float32).reshape(1, 3, 1, 1)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        sess_options.inter_op_num_threads = 1

        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        self.session = ort.InferenceSession(
            provider(**config["model"]),
            providers=["CPUExecutionProvider"],
            session_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.predict.set_max_batch_size(config["max_batch_size"])
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])

    @serve.batch
    async def predict(self, images: list[bytes]) -> list[float]:
        batch = np.frombuffer(b"".join(images), dtype=np.uint8).reshape(
            -1, self.tile_size, self.tile_size, 3
        )
        batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)

        # Normalization
        batch -= self.mean
        batch *= self.inv_std

        outputs = self.session.run([self.output_name], {self.input_name: batch})

        return outputs[0].squeeze(1).tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        data = self.decompress(await request.body())
        return await self.predict(data)


app = BinaryClassifier.bind()
