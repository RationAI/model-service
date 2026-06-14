from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve


if TYPE_CHECKING:
    import torch


class Config(TypedDict):
    tile_size: int
    output_tile_size: int
    n_channels: int
    mpp: float
    max_batch_size: int
    batch_wait_timeout_s: float
    foundation_model_id: str
    model: dict[str, Any]


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class TissueLinear:
    """7-class tissue classifier: linear head over Virchow2 embeddings.

    Per tile: apply Virchow2's transform, fetch the ViT token sequence from
    the deployed Virchow2 service, pool tokens (class token + mean of patch
    tokens) into a 2560-d embedding, run the ONNX linear head, and return a
    7-channel softmax probability map of shape (n_classes, 1, 1). Softmax
    (rather than a hard class index) is used so HeatmapBuilder's resize to
    source resolution interpolates well-defined probabilities; the hard class
    map is recoverable via argmax over channels at full resolution.
    """

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        import onnxruntime as ort
        from timm.data.transforms_factory import create_transform

        self.tile_size = config["tile_size"]
        self.output_tile_size = config["output_tile_size"]
        self.n_channels = config["n_channels"]
        self.mpp = config["mpp"]

        self.foundation_model = serve.get_app_handle(config["foundation_model_id"])

        # Virchow2's eval transform, built from its pretrained_cfg (ImageNet
        # mean/std, bicubic, crop_pct 1.0) instead of loading the ~600M-param
        # model just to read it: saves ~2.4 GB RAM/replica and avoids HF Hub
        # access at init (gated repo). Embeddings come from the Virchow2 service.
        self.foundation_transform = create_transform(
            input_size=(3, self.tile_size, self.tile_size),
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
            crop_mode="center",
            interpolation="bicubic",
        )

        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)

        model_path = provider(**model_config)

        # Head runs on CPU: a single 2560->n_classes matmul, so GPU launch +
        # transfer overhead would exceed it, and embeddings arrive as CPU numpy.
        # num_gpus: 1 only pins the actor to a worker with torch/timm.
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._num_classes = int(self.session.get_outputs()[0].shape[-1])

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    async def get_config(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "output_tile_size": self.output_tile_size,
            "n_channels": self.n_channels,
            "mpp": self.mpp,
        }

    def _prepare_tile_for_virchow2(self, tile_chw: NDArray[np.uint8]) -> torch.Tensor:
        from PIL import Image

        tile_hwc = tile_chw.transpose(1, 2, 0)
        image = Image.fromarray(tile_hwc)
        # Return [3, 224, 224], not [1, 3, 224, 224].
        return self.foundation_transform(image)

    async def _create_embedding(self, tile: NDArray[np.uint8]) -> np.ndarray:
        import torch

        tile_tensor = await asyncio.to_thread(self._prepare_tile_for_virchow2, tile)

        virchow2_output = await self.foundation_model.predict.remote(tile_tensor)

        if isinstance(virchow2_output, np.ndarray):
            virchow2_output = torch.from_numpy(virchow2_output)

        # Virchow2 predict returns one tensor per tile, shape [tokens, dim].
        # Make it [1, tokens, dim] so pooling is batch-compatible.
        if virchow2_output.ndim == 2:
            virchow2_output = virchow2_output.unsqueeze(0)

        class_token = virchow2_output[:, 0]
        patch_tokens = virchow2_output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @serve.batch
    async def predict(
        self,
        tiles: list[NDArray[np.uint8]],
    ) -> list[NDArray[np.float32]]:
        embeddings = await asyncio.gather(
            *(self._create_embedding(tile) for tile in tiles)
        )
        batch = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

        # ONNX graph ends in Softmax -> (batch, n_classes) probabilities.
        # Reshape each row to (n_classes, 1, 1) for HeatmapBuilder.
        probs = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )[0]

        return [row.reshape(self._num_classes, 1, 1) for row in probs]

    @fastapi.post("/")
    async def root(self, request: Request) -> list[Any]:
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        tile = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size,
            self.tile_size,
            3,
        )
        tile_chw = tile.transpose(2, 0, 1)

        result = await self.predict(tile_chw)
        return result.tolist()


app = TissueLinear.bind()  # type: ignore[attr-defined]
