import subprocess
import sys

from fastapi import FastAPI
from ray import serve


fastapi = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi)
class ThroughputRunner:
    def __init__(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest", "-q"],
            check=True,
        )

    @fastapi.post("/")
    def run(
        self,
        duration_s: float = 60.0,
        concurrency: int = 8,
        timeout: float = 60.0,
    ) -> str:
        result = subprocess.run(
            [
                sys.executable,
                "misc/throughput_test.py",
                "--duration-s",
                str(duration_s),
                "--concurrency",
                str(concurrency),
                "--timeout",
                str(timeout),
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout + (
            f"\nSTDERR:\n{result.stderr}" if result.returncode != 0 else ""
        )


app = ThroughputRunner.bind()
