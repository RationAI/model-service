import asyncio

import mlflow.artifacts
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

fastapi = FastAPI()


class ProstateInput(BaseModel):
    input: list


@serve.deployment(ray_actor_options={"num_cpus": 2})
@serve.ingress(fastapi)
class ProstateModel:
    def __init__(self):
        artifact_uri = (
            "mlflow-artifacts:/65/aebc892f526047249b972f200bef4381/"
            "artifacts/checkpoints/epoch=0-step=6972/model_cpu.onnx"
        )

        model_path = mlflow.artifacts.download_artifacts(artifact_uri)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @fastapi.post("")
    async def predict(self, payload: ProstateInput):
        x = np.array(payload.input, dtype=np.float32)

        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        if x.ndim != 4:
            raise HTTPException(400, f"Input must be 3D or 4D. Got shape {x.shape}")

        outputs = await asyncio.to_thread(
            self.session.run, [self.output_name], {self.input_name: x}
        )

        return {"prediction": outputs[0].tolist()}


app = ProstateModel.bind()
