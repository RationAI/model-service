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
    def run(self) -> str:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/model_snapshots/",
                "-v",
                "--tb=short",
                "--no-header",
                "-s",
                "--color=no",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout + (
            f"\nSTDERR:\n{result.stderr}" if result.returncode != 0 else ""
        )


app = TestRunner.bind()
