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
class BreastCancerVirchow2:
    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        import onnxruntime as ort
        import timm
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked

        self.tile_size = config["tile_size"]
        self.output_tile_size = config["output_tile_size"]
        self.n_channels = config["n_channels"]
        self.mpp = config["mpp"]

        self.foundation_model = serve.get_app_handle(config["foundation_model_id"])

        # Only used to construct the correct Virchow2 transform.
        # The real embeddings are produced by the deployed Virchow2 service.
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

        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)

        downloaded_path = Path(provider(**model_config))

        if downloaded_path.is_dir():
            candidates = list(downloaded_path.rglob("model.onnx"))

            if not candidates:
                raise FileNotFoundError(
                    "Downloaded MLflow artifact path is a directory, "
                    f"but no model.onnx was found under: {downloaded_path}"
                )

            model_path = candidates[0]

        elif downloaded_path.name == "model.onnx":
            model_path = downloaded_path

        else:
            candidates = list(downloaded_path.parent.rglob("model.onnx"))

            if not candidates:
                raise FileNotFoundError(
                    "Downloaded MLflow artifact path is not model.onnx and no "
                    f"model.onnx was found nearby. Downloaded path: {downloaded_path}"
                )

            model_path = candidates[0]

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        print(f"Using ONNX model path: {model_path}")
        print(f"ONNX model exists: {model_path.exists()}")
        print(f"ONNX model is file: {model_path.is_file()}")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

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
        tile_hwc = tile_chw.transpose(1, 2, 0)
        image = Image.fromarray(tile_hwc)

        # Important: return [3, 224, 224], not [1, 3, 224, 224].
        return self.foundation_transform(image)

    async def _create_embedding(self, tile_chw: NDArray[np.uint8]) -> np.ndarray:
        tile_tensor = self._prepare_tile_for_virchow2(tile_chw)

        virchow2_output = await self.foundation_model.predict.remote(tile_tensor)

        if isinstance(virchow2_output, np.ndarray):
            virchow2_output = torch.from_numpy(virchow2_output)

        # Virchow2 predict returns one tensor per tile, shape [tokens, dim].
        # Make it [1, tokens, dim] so the pooling code is batch-compatible.
        if virchow2_output.ndim == 2:
            virchow2_output = virchow2_output.unsqueeze(0)

        class_token = virchow2_output[:, 0]
        patch_tokens = virchow2_output[:, 5:]

        embedding = torch.cat(
            [class_token, patch_tokens.mean(dim=1)],
            dim=-1,
        )

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

        probabilities = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )[0]

        # Important for heatmap-builder:
        # each tile prediction must be 2D or 3D.
        # For one scalar probability per tile, return a 1x1 map.
        return [
            np.asarray([[float(prob)]], dtype=np.float32)
            for prob in probabilities.reshape(-1)
        ]

    @fastapi.post("/")
    async def root(self, request: Request) -> list[list[float]]:
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        tile = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size,
            self.tile_size,
            3,
        )

        tile_chw = tile.transpose(2, 0, 1)

        result = await self.predict(tile_chw)

        return result.tolist()


app = BreastCancerVirchow2.bind()  # type: ignore[attr-defined]
