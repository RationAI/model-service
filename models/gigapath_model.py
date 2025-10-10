import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from ray import serve
from ray.runtime_env import RuntimeEnv

BATCH_SIZE = 1
fastapi = FastAPI()


class Result(BaseModel):
    embeddings: list[NDArray[np.uint8]]


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
                "flash-attn @ https://github.com/MiroPsota/torch_packages_builder/releases/download/flash_attn-2.8.3/flash_attn-2.8.3%2Bpt2.8.0cu129-cp312-cp312-linux_x86_64.whl",
                ]
        ),
    },
)
@serve.ingress(fastapi)
class gigapathModel:
    device = "cuda"

    def __init__(self) -> None:
        from gigapath import slide_encoder
        self.model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(torch.device("cuda"))
        self.model.eval()

    @serve.batch(max_batch_size=BATCH_SIZE)
    async def run_model(self, images: list[NDArray[np.uint8]]):
        return self.run_model(images)

    def _run_model(self, embeddings: Tensor, coords: Tensor):
        embeddings.to(self.device)
        coords.to(self.device)
        slide_level_output = slide_encoder(embeddings, coords)
        results.embeddings = slide_level_output

app = gigapathModel.bind()
