import asyncio
import importlib
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from PIL import Image
from ray import serve


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
class BreastCancerGradingVirchow2:
    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        import onnxruntime as ort
        import timm
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked

        # Grid and slide resolution metadata needed by universal builders
        self.tile_size = config["tile_size"]
        self.output_tile_size = config["output_tile_size"]
        self.n_channels = config["n_channels"]
        self.mpp = config["mpp"]

        # Connect this deployment to the cluster's running Virchow2 service
        self.foundation_model = serve.get_app_handle(config["foundation_model_id"])

        # Instantiates an offline token skeleton to match the exact Virchow2 transform logic
        virchow2 = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=False,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        self.foundation_transform = create_transform(
            **resolve_data_config(virchow2.pretrained_cfg, model=virchow2)
        )

        # Parse and fetch your trained 4-class linear head ONNX file from MLflow
        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)

        downloaded_path = Path(provider(**model_config))
        candidates = list(downloaded_path.rglob("model.onnx"))

        if not candidates:
            raise FileNotFoundError(
                "Downloaded MLflow artifact path is a directory, "
                f"but no model.onnx was found under: {downloaded_path}"
            )

        model_path = candidates[0]

        # Spin up your linear head ONNX session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Enforce micro-batching limits for your 4-class ONNX head evaluation pass
        self._predict_head.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self._predict_head.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    async def get_config(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "output_tile_size": self.output_tile_size,
            "n_channels": self.n_channels,
            "mpp": self.mpp,
        }

    def _prepare_tile_for_virchow2(self, tile_chw: NDArray[np.uint8]) -> torch.Tensor:
        # Flip layouts from Channel-Height-Width back to standard Image arrays
        tile_hwc = tile_chw.transpose(1, 2, 0)
        image = Image.fromarray(tile_hwc)

        # Returns the normalized [3, 224, 224] patch structure
        return self.foundation_transform(image)

    async def _create_embedding(self, tile: NDArray[np.uint8]) -> np.ndarray:
        tile_tensor = await asyncio.to_thread(self._prepare_tile_for_virchow2, tile)

        # Execute remote pipeline call to the shared Virchow2 microservice
        virchow2_output = await self.foundation_model.predict.remote(tile_tensor)

        if isinstance(virchow2_output, torch.Tensor):
            virchow2_output = virchow2_output.cpu().numpy()

        if virchow2_output.ndim == 2:
            virchow2_output = np.expand_dims(virchow2_output, axis=0)

        # Pool patch tokens matching the baseline foundation extraction layout
        class_token = virchow2_output[:, 0]
        patch_tokens = virchow2_output[:, 5:]

        embedding = np.concatenate(
            [class_token, patch_tokens.mean(axis=1)],
            axis=-1,
        )

        return np.squeeze(embedding, axis=0).astype(np.float32, copy=False)

    @serve.batch
    async def _predict_head(
        self,
        embeddings: list[NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        batch = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

        # Evaluates the batched tensors through your 4-class linear network layer
        probabilities = await asyncio.to_thread(
            self.session.run,
            [self.output_name],
            {self.input_name: batch},
        )
        probabilities = probabilities[0]

        # Modified to match 4-class heatmap dimensions:
        # Reshapes predictions to [1, 1, 4] so the universal system-level
        # HeatmapBuilder maps tissue grades over 4 channels instead of a binary scalar.
        return [prob.reshape(1, 1, 4).astype(np.float32) for prob in probabilities]

    async def predict(
        self,
        tile: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        embedding = await self._create_embedding(tile)
        return await self._predict_head(embedding)

    @fastapi.post("/")
    async def root(self, request: Request) -> list[list[list[float]]]:
        # 1. Unzip raw compressed image tile bytes coming from network traffic
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        # 2. Reconstruct the raw pixel array
        tile = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size,
            self.tile_size,
            3,
        )

        tile_chw = tile.transpose(2, 0, 1)

        # 3. Fire pipeline (Raw tile -> Virchow2 embedding -> Your 4-class Head)
        result = await self.predict(tile_chw)

        return result.tolist()


app = BreastCancerGradingVirchow2.bind()  # type: ignore[attr-defined]
