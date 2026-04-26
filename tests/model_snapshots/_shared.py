from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from numpy.typing import NDArray


def _models_base_url() -> str:
    return os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-tests-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
    )


def _client(timeout_s: float = 1200.0):
    try:
        from rationai import Client
    except ImportError:
        pytest.skip("Python package `rationai` is not installed.")
    return Client(models_base_url=_models_base_url(), timeout=timeout_s)


def _read_tile(slide_path: str, tile_size: int, level: int) -> NDArray[np.uint8]:
    try:
        from ratiopath.openslide import OpenSlide
    except ImportError:
        pytest.skip("Python package `ratiopath` is not installed.")

    with OpenSlide(slide_path) as slide:
        w, h = slide.level_dimensions[level]
        x = max(0, (w - tile_size) // 2)
        y = max(0, (h - tile_size) // 2)
        tile = slide.read_region_relative(
            (x, y), level, (tile_size, tile_size)
        ).convert("RGB")

    return np.asarray(tile, dtype=np.uint8)


def run_binary_classifier_case(
    model_id: str,
    slide_path: str,
    expected_score: float,
    tile_size: int = 512,
    level: int = 0,
    timeout_s: float = 600.0,
    tolerance: float = 0.00001,
) -> None:
    tile = _read_tile(slide_path, tile_size, level)

    with _client(timeout_s) as client:
        t0 = perf_counter()
        actual_score = float(client.models.classify_image(model=model_id, image=tile))
        elapsed = perf_counter() - t0

    print(
        f"\n  model={model_id} | tile={tile_size}px | time={elapsed:.2f}s | score={actual_score:.6f} | expected={expected_score:.6f}"
    )

    assert abs(actual_score - expected_score) <= tolerance, (
        f"Binary score mismatch: expected={expected_score}, actual={actual_score}, tolerance={tolerance}"
    )


def run_semantic_segmentation_case(
    model_id: str,
    slide_path: str,
    expected_array_path: Path | str,
    tile_size: int = 1024,
    level: int = 0,
    timeout_s: float = 1200.0,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    expected_array_path = Path(expected_array_path)
    if not expected_array_path.exists():
        pytest.fail(f"Reference file does not exist: {expected_array_path}")

    tile = _read_tile(slide_path, tile_size, level)
    expected = np.load(expected_array_path)

    with _client(timeout_s) as client:
        t0 = perf_counter()
        actual = np.asarray(client.models.segment_image(model=model_id, image=tile))
        elapsed = perf_counter() - t0

    max_diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32)).max()
    print(
        f"\n  model={model_id} | tile={tile_size}px | time={elapsed:.2f}s | shape={actual.shape} | max_diff={max_diff:.6f}"
    )

    if actual.shape != expected.shape:
        pytest.fail(f"Shape mismatch: expected={expected.shape}, actual={actual.shape}")

    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        pytest.fail(
            f"Output mismatch beyond tolerance (atol={atol}, rtol={rtol}, max_abs_diff={max_diff})"
        )
