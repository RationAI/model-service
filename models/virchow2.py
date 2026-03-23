import asyncio
import logging
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


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class Virchow2:
    """Virchow2 foundation model for pathology."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.transforms: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile_size: int = 0

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
    async def predict(
        self, inputs: list[torch.Tensor | NDArray[np.float16] | NDArray[np.float32]]
    ) -> list[NDArray[np.float32]]:
        tensors = torch.stack(
            [
                item if isinstance(item, torch.Tensor) else torch.from_numpy(item)
                for item in inputs
            ]
        ).to(self.device)
        model = cast("torch.nn.Module", self.model)

        device_type = self.device.type
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device_type, dtype=autocast_dtype),
        ):
            output = model(tensors)
            class_token = output[:, 0]
            patch_tokens = output[:, 5:]
            embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

        return list(embedding.float().cpu().numpy())

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        data = await asyncio.to_thread(lz4.frame.decompress, await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        requested_dtype = request.headers.get("x-output-dtype", "float32").lower()

        output_dtype = np.dtype(requested_dtype)
        tensor = self.transforms(Image.fromarray(image))
        result: NDArray[np.float32] = await self.predict(tensor)
        output_shape = ",".join(str(d) for d in result.shape)

        return Response(
            content=lz4.frame.compress(
                result.astype(output_dtype, copy=False).tobytes()
            ),
            media_type="application/octet-stream",
            headers={
                "x-output-shape": output_shape,
            },
        )


app = Virchow2.bind()  # type: ignore[attr-defined]
