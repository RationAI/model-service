import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from ray import serve
from ray.runtime_env import RuntimeEnv

BATCH_SIZE = 8
fastapi = FastAPI()


class Result(BaseModel):
    polygons: list[list[list[float]]]
    embeddings: list[list[float]]
    model_config = ConfigDict(arbitrary_types_allowed=True)


@serve.deployment(
    num_replicas="auto",
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 8,
        "target_ongoing_requests": 64,
        "downscale_delay_s": 60,
        "upscale_delay_s": 60,
    },
    ray_actor_options={
        "num_cpus": 0.25,
        "num_gpus": 1,
        "memory": 3 * 1024**3,
        "runtime_env": RuntimeEnv(
            pip=[
                "torch>=2.8.0",
                "transformers>=4.55.0",
                "einops>=0.8.1",
                "torchvision>=0.23.0",
            ]
        ),
    },
)
@serve.ingress(fastapi)
class LSPDetr:
    device = "cuda"

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model = AutoModelForObjectDetection.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
            token="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",
        ).to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
            token="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",
        )

        self.runners = {
            2048: self.run_model_2048,
            1024: self.run_model_1024,
            512: self.run_model_512,
            256: self.run_model_256,
        }

    @serve.batch(max_batch_size=BATCH_SIZE)
    async def run_model_2048(self, images: list[NDArray[np.uint8]]):
        return self._run_model(images)

    @serve.batch(max_batch_size=BATCH_SIZE * 4)
    async def run_model_1024(self, images: list[NDArray[np.uint8]]):
        return self._run_model(images)

    @serve.batch(max_batch_size=BATCH_SIZE * 4**2)
    async def run_model_512(self, images: list[NDArray[np.uint8]]):
        return self._run_model(images)

    @serve.batch(max_batch_size=BATCH_SIZE * 4**3)
    async def run_model_256(self, images: list[NDArray[np.uint8]]):
        return self._run_model(images)

    def _run_model(self, images: list[NDArray[np.uint8]]):
        import torch

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            inputs = self.processor(images, device=self.device, return_tensors="pt")
            outputs = self.model(**inputs)
            results = self.processor.post_process(outputs)

        return [
            Result(polygons=r["polygons"].tolist(), embeddings=r["embeddings"].tolist())
            for r in results
        ]

    @fastapi.post("/{tile_size}", status_code=status.HTTP_200_OK)
    async def process_tile(self, tile_size: int, request: Request):
        if tile_size not in self.runners:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tile size {tile_size} not supported. Available sizes: {list(self.runners.keys())}",
            )

        data = await request.body()
        image = np.frombuffer(data, dtype=np.uint8).reshape(tile_size, tile_size, 3)
        return await self.runners[tile_size](image.copy())


app = LSPDetr.bind()
