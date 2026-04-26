import subprocess
import sys

from fastapi import FastAPI
from ray import serve


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class TestRunner:
    def __init__(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest", "-q"],
            check=True,
        )

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
