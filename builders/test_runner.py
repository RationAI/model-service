import subprocess
import sys

import requests
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

    def _model_statuses(self) -> str:
        try:
            resp = requests.get("http://localhost:52365/api/serve/applications/")
            resp.raise_for_status()
            data = resp.json()
            lines = ["Model statuses:"]
            for app_name, app_info in data.get("applications", {}).items():
                for dep_name, dep_info in app_info.get("deployments", {}).items():
                    status = dep_info.get("status", "UNKNOWN")
                    lines.append(f"  {app_name} ({dep_name}): {status}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not fetch model statuses: {e}"

    @fastapi.post("/")
    def run(self) -> str:
        statuses = self._model_statuses()

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

        output = statuses + "\n\n" + result.stdout
        if result.returncode != 0:
            output += f"\nSTDERR:\n{result.stderr}"
        return output


app = TestRunner.bind()
