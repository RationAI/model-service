from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
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

        # Build Virchow2's eval transform directly from its known pretrained_cfg
        # (verified against the model's config.json: ImageNet mean/std, bicubic,
        # crop_pct 1.0). This avoids instantiating the full ~600M-param model
        # just to read its transform config, saving ~2.4 GB RAM per replica and
        # removing any Hugging Face Hub access at init (the repo is gated). The
        # embeddings themselves are produced by the deployed Virchow2 service.
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

        # Resolve the .onnx file from the MLflow download. The provider may
        # return the file directly, a directory containing it, or a sibling
        # path. Resolved inline (no module-level helper) because Ray's
        # by-value deployment serialization does not reliably carry module
        # globals into the worker.
        downloaded_path = Path(provider(**model_config))
        if downloaded_path.is_file() and downloaded_path.suffix == ".onnx":
            model_path = downloaded_path
        else:
            search_root = (
                downloaded_path if downloaded_path.is_dir() else downloaded_path.parent
            )
            candidates = list(search_root.rglob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .onnx file found at or near downloaded path: {downloaded_path}"
                )
            model_path = candidates[0]
        print(f"Using ONNX model path: {model_path}")

        # Run the head on CPU. It is a single 2560->n_classes linear, so the
        # GPU kernel-launch and host<->device transfer overhead would exceed
        # the matmul itself, and the embeddings already arrive as CPU numpy.
        # The num_gpus: 1 reservation is only to land the actor on a worker
        # image that carries torch/timm, not for ONNX compute.
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._num_classes = int(self.session.get_outputs()[0].shape[-1])

        # Fail fast on config that contradicts the model's output contract:
        # one softmax probability per class per tile, shape (n_classes, 1, 1).
        # A mismatch (e.g. a stale n_channels) would silently corrupt the
        # HeatmapBuilder output instead of erroring.
        if self.n_channels != self._num_classes:
            raise ValueError(
                f"n_channels ({self.n_channels}) must equal the ONNX head's "
                f"number of classes ({self._num_classes})"
            )
        if self.output_tile_size != 1:
            raise ValueError(
                f"output_tile_size must be 1 for per-tile classification, "
                f"got {self.output_tile_size}"
            )
        # Virchow2's pooled embedding is 2560-d (class token + patch-token mean,
        # 1280 each). Guard against an artifact_uri pointing to a head trained
        # for a different foundation model, which would otherwise fail with a
        # cryptic shape error on the first session.run mid-slide.
        expected_embedding_dim = 2560
        onnx_input_dim = int(self.session.get_inputs()[0].shape[-1])
        if onnx_input_dim != expected_embedding_dim:
            raise ValueError(
                f"ONNX head expects input width {onnx_input_dim}, but the "
                f"Virchow2 embedding is {expected_embedding_dim}-d; the "
                f"artifact_uri likely points to a head for a different "
                f"foundation model"
            )

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

        # The ONNX graph ends in a Softmax, so this already returns per-class
        # probabilities of shape (batch, n_classes). Reshape each row to a
        # (n_classes, 1, 1) map for HeatmapBuilder.
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
