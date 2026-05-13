import subprocess
import sys

from fastapi import FastAPI, Response
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
    ) -> Response:
        result = subprocess.run(
            [
                sys.executable,
                "tests/benchmark/perf_throughput.py",
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
        output = result.stdout + (
            f"\nSTDERR:\n{result.stderr}" if result.returncode != 0 else ""
        )
        return Response(content=output, media_type="text/plain")


app = ThroughputRunner.bind()
