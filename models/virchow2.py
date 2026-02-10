import asyncio
from typing import Any, TypedDict

import numpy as np
import timm
import torch
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked


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

        self.lz4 = lz4.frame
        self.model: torch.nn.Module | None = None
        self.transforms: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reconfigure(self, config: Config) -> None:
        import importlib
        import logging

        logger = logging.getLogger("ray.serve")
        self.tile_size = config["tile_size"]

        # Load model using the provider
        module_path, attr_name = config["model"].pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)
        repo_id = config["model"]["repo_id"]

        logger.info(f"Loading Virchow2 model from {repo_id}...")
        provider(**config["model"])

        self.model = timm.create_model(
            f"hf-hub:{repo_id}",
            pretrained=True,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.to(self.device).eval()
        logger.info("Virchow2 model loaded and moved to GPU.")

        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

        # Configure batching
        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

        # Warmup
        logger.info("Starting warmup...")
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            # Create a dummy input tensor
            dummy_input = torch.randn(
                1, 3, self.tile_size, self.tile_size, device=self.device
            )
            self.model(dummy_input)
        logger.info("Warmup complete.")

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[list[float]]:
        from PIL import Image

        if self.model is None or self.transforms is None:
            raise RuntimeError("Model or transforms not initialized")

        # Convert numpy arrays to PIL Images and apply transforms
        pil_images = [Image.fromarray(img) for img in images]
        tensors = torch.stack([self.transforms(img) for img in pil_images]).to(
            self.device
        )

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            output = self.model(tensors)

            class_token = output[:, 0]  # size: batch x 1280
            patch_tokens = output[
                :, 5:
            ]  # size: batch x 256 x 1280 (skip register tokens 1-4)
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: batch x 2560
            embedding = embedding.to(torch.float16)

        return embedding.cpu().tolist()

    @fastapi.post("/")
    async def root(self, request: Request) -> list[float]:
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        # Reshape to (height, width, channels) - RGB image
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        results = await self.predict(image)
        return results  # type: ignore[attr-defined]


app = Virchow2.bind()  # type: ignore[attr-defined]
