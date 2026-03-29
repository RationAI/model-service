import asyncio
import logging
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import lz4.frame
import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from numpy.typing import NDArray
from PIL import Image
from ray import serve


class Config(TypedDict):
    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float


fastapi = FastAPI()
logger = logging.getLogger("ray.serve")


@dataclass
class PredictInput:
    array: NDArray[np.float32]
    dtype: np.dtype
    pool_tokens: bool


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class Virchow2:
    """Virchow2 foundation model for pathology."""

    model: torch.nn.Module
    transforms: Any
    tile_size: int

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reconfigure(self, config: Config) -> None:
        import importlib

        import timm
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked

        self.tile_size = config["tile_size"]

        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        repo_id = model_config["repo_id"]

        logger.info(f"Loading Virchow2 model from {repo_id}...")

        # The provider is required to register the custom Virchow2 architecture
        # and layers (like SwiGLUPacked) into the timm registry.
        # Without this, timm.create_model would fail with an 'Unknown model' error.
        provider(**model_config)

        self.model = timm.create_model(
            f"hf-hub:{repo_id}",
            pretrained=True,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.to(self.device).eval()

        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, inputs: list[PredictInput]) -> list[NDArray[Any]]:
        tensors = torch.stack([torch.from_numpy(inp.array) for inp in inputs]).to(
            self.device
        )
        device_type = self.device.type
        # PyTorch autocast does not support float16 on CPU (throws RuntimeError).
        # bfloat16 is the only supported low-precision option for CPU inference.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device_type, dtype=autocast_dtype),
        ):
            output = self.model(tensors)

        results = []
        for i, inp in enumerate(inputs):
            single_output = output[i]

            if inp.pool_tokens:
                class_token = single_output[0]
                patch_tokens = single_output[5:]
                embedding = torch.cat([class_token, patch_tokens.mean(dim=0)], dim=-1)
            else:
                embedding = torch.cat([single_output[0:1], single_output[5:]], dim=0)

            results.append(
                embedding.float().cpu().numpy().astype(inp.dtype, copy=False)
            )

        return results

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        data = await asyncio.to_thread(lz4.frame.decompress, await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        output_dtype = np.dtype(
            request.headers.get("x-output-dtype", "float32").lower()
        )
        pool_tokens = request.headers.get("x-pool-tokens", "true").lower() == "true"

        tensor = self.transforms(Image.fromarray(image))
        array = tensor.numpy()
        result = await cast(
            "Any",
            self.predict(
                PredictInput(array=array, dtype=output_dtype, pool_tokens=pool_tokens)
            ),
        )

        output_shape = ",".join(str(d) for d in result.shape)

        return Response(
            content=lz4.frame.compress(result.tobytes()),
            media_type="application/octet-stream",
            headers={
                "x-output-shape": output_shape,
            },
        )


app = Virchow2.bind()  # type: ignore[attr-defined]
