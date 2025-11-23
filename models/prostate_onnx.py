import os
import shutil

import numpy as np
import onnxruntime as ort
from mlflow.tracking import MlflowClient
from ray import serve
from starlette.requests import Request


@serve.deployment(ray_actor_options={"num_cpus": 2})
class ProstateModel:
    def __init__(self):
        model_dir = "model"
        model_filename = "model_cpu.onnx"
        os.makedirs(model_dir, exist_ok=True)
        cached_model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(cached_model_path):
            print(f"Using cached ONNX model at {cached_model_path}")
            model_path = cached_model_path
        else:
            try:
                client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
                run_id = "aebc892f526047249b972f200bef4381"
                remote_path = "checkpoints/epoch=0-step=6972/model_cpu.onnx"

                print(f"Downloading model from MLflow: {run_id}")
                tmp_dir = client.download_artifacts(run_id, remote_path, model_dir)

                if os.path.isdir(tmp_dir):
                    tmp_model_file = os.path.join(tmp_dir, model_filename)
                    shutil.move(tmp_model_file, cached_model_path)
                else:
                    shutil.move(tmp_dir, cached_model_path)

                model_path = cached_model_path
                print(f"Model downloaded and cached at {model_path}")
            except Exception as e:
                raise RuntimeError(f"Cannot initialize ProstateModel: {e}") from e

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        print("ProstateModel initialized successfully")

    async def __call__(self, request: Request):
        try:
            data = await request.json()
            input_data = np.array(data["input"], dtype=np.float32)

            print(f"Processing input shape: {input_data.shape}")

            outputs = self.session.run(None, {"input": input_data})

            return {"prediction": outputs[0].tolist()}
        except Exception as e:
            print(f"Error during inference: {e}")
            raise


app = ProstateModel.bind()
