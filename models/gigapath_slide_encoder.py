import numpy as np
from fastapi import FastAPI, Request, status
from numpy.typing import NDArray
from pydantic import BaseModel
from ray import serve
from ray.runtime_env import RuntimeEnv

BATCH_SIZE = 1
fastapi = FastAPI()


class Result(BaseModel):
    embeddings: list[list[float]]


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
        "num_cpus": 4,
        "num_gpus": 1,
        "memory": 24 * 1024**3,
        "runtime_env": RuntimeEnv(
            pip=[
                "fairscale",
                "numpy",
                "timm",
                "torch==2.8.0",
                "huggingface-hub",
                "flash-attn @ https://github.com/MiroPsota/torch_packages_builder/releases/download/flash_attn-2.8.3/flash_attn-2.8.3%2Bpt2.8.0cu129-cp312-cp312-linux_x86_64.whl",
            ]
        ),
    },
)
@serve.ingress(fastapi)
class GigapathSlideEncoder:
    def __init__(self) -> None:
        from huggingface_hub import login

        from models.prov_gigapath.gigapath.slide_encoder import create_model

        login("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")

        self.model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
        )
        self.model.eval()

    @serve.batch(max_batch_size=BATCH_SIZE)
    async def run_model(
        self, inputs: list[tuple[NDArray[np.float32], NDArray[np.float32]]]
    ) -> list[Result]:
        import torch

        results = []

        print(torch.cuda.is_available())

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for embeddings, coords in inputs:
                print("Device of model:", next(self.model.parameters()).device)
                embeddings = torch.from_numpy(embeddings)
                coords = torch.from_numpy(coords)
                print("Device of embeddings:", embeddings.device)
                print("Device of coords:", coords.device)
                output = self.model(embeddings, coords)

                results.append(Result(embeddings=output[-1].tolist()))

        return results

    @fastapi.post("/{length}", response_model=Result, status_code=status.HTTP_200_OK)
    async def process_embeddings(self, length: int, request: Request) -> Result:
        data = await request.body()

        embeddings = np.frombuffer(data, dtype=np.float32, count=length * 1536)
        coords = np.frombuffer(
            data, dtype=np.float32, count=length * 2, offset=length * 1536 * 4
        )

        embeddings = embeddings.reshape(1, length, 1536)
        coords = coords.reshape(1, length, 2)

        return await self.run_model((embeddings, coords))


app = GigapathSlideEncoder.bind()
