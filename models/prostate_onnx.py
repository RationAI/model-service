import os

import mlflow.artifacts
import numpy as np
import onnxruntime as ort
from ray import serve
from starlette.requests import Request


@serve.deployment(ray_actor_options={"num_cpus": 2})
class ProstateModel:
    def __init__(self):
        mlflow_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow.rationai-mlflow:5000"
        )
        artifact_uri = "mlflow-artifacts:/65/aebc892f526047249b972f200bef4381/artifacts/checkpoints/epoch=0-step=6972/model_cpu.onnx"

        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, tracking_uri=mlflow_uri
        )

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    async def __call__(self, request: Request):
        data = await request.json()
        input_data = np.array(data["input"], dtype=np.float32)

        outputs = self.session.run(None, {"input": input_data})

        return {"prediction": outputs[0].tolist()}


app = ProstateModel.bind()
