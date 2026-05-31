from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock

import numpy as np
from rationai import Client


DEFAULT_MODELS = [
    ("prostate-classifier-1", "binary", 512),
    ("episeg-1", "semantic", 1024),
    ("virchow2", "embed", 224),
    ("prov-gigapath", "embed", 224),
]
POOL_SIZE_DEFAULT = 64


@dataclass
class Stats:
    ok: int = 0
    fail_503: int = 0
    fail_other: int = 0
    latencies: list[float] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    @property
    def total(self) -> int:
        return self.ok + self.fail_503 + self.fail_other

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, p))


def _models_base_url() -> str:
    return os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-tests-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
    )


def make_pool(tile_size: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed=42)
    return [
        rng.integers(0, 256, (tile_size, tile_size, 3), dtype=np.uint8)
        for _ in range(n)
    ]


def _call_model(
    client: Client, model_id: str, model_type: str, image: np.ndarray
) -> None:
    if model_type == "binary":
        client.models.classify_image(model=model_id, image=image)
    elif model_type == "semantic":
        client.models.segment_image(model=model_id, image=image)
    elif model_type == "embed":
        client.models.embed_image(model=model_id, image=image)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def wait_for_ready(
    model_id: str,
    model_type: str,
    tile_size: int,
    timeout: float,
    models_base_url: str,
    wait_timeout_s: float,
    wait_interval_s: float,
) -> None:
    image = make_pool(tile_size, 1)[0]
    start = time.perf_counter()
    reported = False

    while True:
        try:
            with Client(models_base_url=models_base_url, timeout=timeout) as client:
                _call_model(client, model_id, model_type, image)
            if reported:
                print(f"{model_id} ready after {time.perf_counter() - start:.1f}s")
            return
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code not in (None, 503, 504):
                raise
            if not reported:
                print(f"{model_id} waiting for readiness...")
                reported = True
            elapsed = time.perf_counter() - start
            if wait_timeout_s > 0 and elapsed >= wait_timeout_s:
                raise RuntimeError(
                    f"{model_id} not ready after {wait_timeout_s:.1f}s"
                ) from exc
            time.sleep(wait_interval_s)


def send_loop(
    model_id: str,
    model_type: str,
    pool: list[np.ndarray],
    stats: Stats,
    end_time: float,
    timeout: float,
    models_base_url: str,
) -> None:
    pool_len = len(pool)
    idx = 0
    with Client(models_base_url=models_base_url, timeout=timeout) as client:
        while time.perf_counter() < end_time:
            image = pool[idx % pool_len]
            idx += 1
            t0 = time.perf_counter()
            try:
                _call_model(client, model_id, model_type, image)
                latency = time.perf_counter() - t0
                with stats.lock:
                    stats.ok += 1
                    stats.latencies.append(latency)
            except Exception as exc:
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                with stats.lock:
                    if status_code == 503:
                        stats.fail_503 += 1
                    else:
                        stats.fail_other += 1


def run_model(
    name: str,
    model_type: str,
    tile_size: int,
    duration_s: float,
    concurrency: int,
    timeout: float,
    pool_size: int,
    models_base_url: str,
) -> dict[str, object]:
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0")

    pool = make_pool(tile_size, pool_size)
    stats = Stats()

    start = time.perf_counter()
    end_time = start + duration_s
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_loop,
                name,
                model_type,
                pool,
                stats,
                end_time,
                timeout,
                models_base_url,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            future.result()
    elapsed = time.perf_counter() - start

    throughput = stats.ok / elapsed if elapsed > 0 else 0.0
    return {
        "name": name,
        "model_type": model_type,
        "tile_size": tile_size,
        "elapsed_s": elapsed,
        "ok": stats.ok,
        "fail_503": stats.fail_503,
        "fail_other": stats.fail_other,
        "throughput": throughput,
        "p50": stats.percentile(50),
        "p95": stats.percentile(95),
        "p99": stats.percentile(99),
    }


def parse_models(values: list[str]) -> list[tuple[str, str, int]]:
    if not values:
        return DEFAULT_MODELS
    parsed: list[tuple[str, str, int]] = []
    for item in values:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError("--model expects: model_id,model_type,tile_size")
        model_id, model_type, tile_size = parts[0], parts[1], int(parts[2])
        parsed.append((model_id, model_type, tile_size))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-model throughput tests and report img/s via SDK."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec: model_id,model_type,tile_size (repeatable)",
    )
    parser.add_argument(
        "--models-base-url",
        default=_models_base_url(),
        help="Base URL for the SDK (default: MODEL_SERVICE_MODELS_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE_DEFAULT)
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Wait for each model to become ready before running the test",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=0.0,
        help="Max time to wait for readiness (0 = wait forever)",
    )
    parser.add_argument(
        "--wait-interval-s",
        type=float,
        default=10.0,
        help="Wait interval between readiness checks",
    )
    args = parser.parse_args()

    models = parse_models(args.model)

    print("=" * 72)
    print("Throughput Test (img/s) - SDK")
    print("=" * 72)
    print(f"Models base URL: {args.models_base_url}")
    print(f"Duration:        {args.duration_s:.0f}s")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Timeout:         {args.timeout}s")
    print()

    results = []
    for name, model_type, tile_size in models:
        if args.wait_ready:
            wait_for_ready(
                model_id=name,
                model_type=model_type,
                tile_size=tile_size,
                timeout=args.timeout,
                models_base_url=args.models_base_url,
                wait_timeout_s=args.wait_timeout_s,
                wait_interval_s=args.wait_interval_s,
            )
        result = run_model(
            name,
            model_type,
            tile_size,
            args.duration_s,
            args.concurrency,
            args.timeout,
            args.pool_size,
            args.models_base_url,
        )
        results.append(result)
        print(
            f"{name} stats: ok={result['ok']} fail_503={result['fail_503']} "
            f"fail_other={result['fail_other']} elapsed={result['elapsed_s']:.2f}s "
            f"img/s={result['throughput']:.2f} p50={result['p50']:.3f}s "
            f"p95={result['p95']:.3f}s p99={result['p99']:.3f}s"
        )

    print("Summary")
    print(
        "name".ljust(28),
        "img/s".rjust(10),
        "p50".rjust(10),
        "p95".rjust(10),
        "p99".rjust(10),
    )
    for r in results:
        print(
            r["name"].ljust(28),
            f"{r['throughput']:.2f}".rjust(10),
            f"{r['p50']:.3f}".rjust(10),
            f"{r['p95']:.3f}".rjust(10),
            f"{r['p99']:.3f}".rjust(10),
        )


if __name__ == "__main__":
    main()
