import subprocess
import sys

from fastapi import FastAPI
from ray import serve


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class TestRunner:
    @fastapi.post("/")
    def run(self) -> dict:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/model_snapshots/",
                "-v",
                "--tb=short",
            ],
            capture_output=True,
            text=True,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0,
        }


app = TestRunner.bind()
