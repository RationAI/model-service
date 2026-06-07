import asyncio
import importlib
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
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
                "but no model.onnx was found under: "
                f"{downloaded_path}"
            )

        model_path = candidates[0]

        # Spin up your linear head ONNX session using CPU Execution to prevent host<->device lag
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._num_classes = int(self.session.get_outputs()[0].shape[-1])

        if self.n_channels not in (4, 8):
            raise ValueError(
                f"n_channels config is set to {self.n_channels}, but must be "
                f"either 4 (Softmax only) or 8 (Softmax + Logits)."
            )

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

        if isinstance(virchow2_output, np.ndarray):
            virchow2_output = torch.from_numpy(virchow2_output)

        # Virchow2 predict returns one tensor per tile, shape [tokens, dim].
        # Make it [1, tokens, dim] so pooling is batch-compatible.
        if virchow2_output.ndim == 2:
            virchow2_output = virchow2_output.unsqueeze(0)

        class_token = virchow2_output[:, 0]
        patch_tokens = virchow2_output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

        # Safe squeeze that leaves multi-tile production batch axes untouched
        return embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @serve.batch
    async def _predict_head(
        self,
        embeddings: list[NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        batch = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

        # 1. Evaluate the [B, 4] raw logit matrix from your exported ONNX linear model
        logits_np = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )[0]

        # 2. Compute Softmax dynamically using PyTorch over the final dimension
        logits_tensor = torch.from_numpy(logits_np)
        softmax_tensor = torch.nn.functional.softmax(logits_tensor, dim=-1)
        softmax_np = softmax_tensor.cpu().numpy().astype(np.float32, copy=False)

        # 3. Concatenate along channel axis: shape transitions from (B, 4) + (B, 4) to (B, 8)
        # Put Softmax FIRST so HeatmapBuilder reads it when config is sliced to 4
        combined_outputs = np.concatenate([softmax_np, logits_np], axis=-1)

        # 4. DYNAMIC SLICE: Slice the array to match exactly what the platform requested
        # If config is 4, rows become shape (4, 1, 1) -> Safe for HeatmapBuilder
        # If config is 8, rows become shape (8, 1, 1) -> Full data, both softmax and logits
        return [
            row[: self.n_channels].reshape(self.n_channels, 1, 1).astype(np.float32)
            for row in combined_outputs
        ]

    # Entry point takes exactly ONE tile at a time from root
    async def predict(
        self,
        tile: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        embedding = await self._create_embedding(tile)
        results = await self._predict_head(embedding)
        return results[0]

    @fastapi.post("/")
    async def root(self, request: Request) -> list[list[list[float]]]:
        body_bytes = await request.body()

        try:
            data = await asyncio.to_thread(self.lz4.decompress, body_bytes)

            expected_bytes = self.tile_size * self.tile_size * 3
            if len(data) != expected_bytes:
                raise ValueError(
                    f"Decompressed payload byte length mismatch. "
                    f"Expected exactly {expected_bytes} bytes, but got {len(data)}."
                )

            # Reconstruct the raw pixel array
            tile = np.frombuffer(data, dtype=np.uint8).reshape(
                self.tile_size,
                self.tile_size,
                3,
            )
        except (RuntimeError, ValueError) as err:
            raise HTTPException(
                status_code=400,
                detail=f"Malformed or invalid compressed tile image payload: {err!s}",
            ) from err

        tile_chw = tile.transpose(2, 0, 1)

        result = await self.predict(tile_chw)

        return result.tolist()


app = BreastCancerGradingVirchow2.bind()  # type: ignore[attr-defined]
