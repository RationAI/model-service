import asyncio
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


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class Virchow2:
    """Virchow2 foundation model for pathology."""

    def __init__(self) -> None:
        import os

        import lz4.frame

        # Enforce offline mode for timm/huggingface_hub
        os.environ["HF_HUB_OFFLINE"] = "1"

        import torch

        self.torch = torch
        self.lz4 = lz4.frame
        self.model: Any = None
        self.transforms: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile_size: int = 0

    def reconfigure(self, config: Config) -> None:
        import importlib
        import logging

        import timm
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked

        torch = self.torch

        logger = logging.getLogger("ray.serve")
        self.tile_size = config["tile_size"]

        # Load model using the provider
        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        repo_id = config["model"]["repo_id"]

        logger.info(f"Loading Virchow2 model from {repo_id}...")
        provider(**config["model"])

        # Load model with official architecture
        self.model = timm.create_model(
            f"hf-hub:{repo_id}",
            pretrained=True,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.to(self.device).eval()

        # Get transforms from model config
        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

        logger.info("Virchow2 model loaded and moved to GPU.")

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[list[float]]:
        from PIL import Image

        if self.model is None or self.transforms is None:
            raise RuntimeError("Model or transforms not initialized")

        torch = self.torch

        pil_images = [Image.fromarray(img) for img in images]
        tensors = torch.stack([self.transforms(img) for img in pil_images]).to(
            self.device
        )

        device_type = self.device.type
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device_type, dtype=autocast_dtype),
        ):
            output = self.model(tensors)

            # Extract embeddings as per official model card
            class_token = output[:, 0]  # CLS token: batch x 1280
            patch_tokens = output[
                :, 5:
            ]  # Skip register tokens (1-4): batch x 256 x 1280

            # Concatenate CLS token with mean of patch tokens
            embedding = torch.cat(
                [class_token, patch_tokens.mean(dim=1)], dim=-1
            )  # batch x 2560

        return embedding.half().cpu().tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> list[float]:
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        # Reshape to (height, width, channels) - RGB image
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        results = await self.predict(image)
        return results


app = Virchow2.bind()  # type: ignore[attr-defined]
