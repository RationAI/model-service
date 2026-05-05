from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from numpy.typing import NDArray
from rationai import Client
from ratiopath.openslide import OpenSlide


def _models_base_url() -> str:
    return os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-tests-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
    )


def _read_tile_at(
    slide_path: str, x: int, y: int, tile_size: int, level: int
) -> NDArray[np.uint8]:
    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[level]
        x_rel = int(x / downsample)
        y_rel = int(y / downsample)
        tile = slide.read_region_relative(
            (x_rel, y_rel), level, (tile_size, tile_size)
        ).convert("RGB")
    return np.asarray(tile, dtype=np.uint8)


def run_binary_classifier_case(
    model_id: str,
    slide_path: str,
    x: int,
    y: int,
    expected_score: float,
    tile_size: int = 512,
    level: int = 0,
    timeout_s: float = 600.0,
    expected_is_positive: bool | None = None,
    threshold: float = 0.5,
) -> None:
    tile = _read_tile_at(slide_path, x, y, tile_size, level)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
        t0 = perf_counter()
        actual_score = float(client.models.classify_image(model=model_id, image=tile))
        elapsed = perf_counter() - t0

    print(
        f"\n  model={model_id} | tile={tile_size}px | time={elapsed:.2f}s | score={actual_score:.6f} | expected={expected_score:.6f}"
    )

    if expected_is_positive is not None:
        actual_is_positive = actual_score >= threshold
        assert actual_is_positive == expected_is_positive, (
            f"Binary class mismatch: expected_is_positive={expected_is_positive}, "
            f"actual_score={actual_score:.6f}, threshold={threshold:.3f}"
        )


def run_semantic_segmentation_case(
    model_id: str,
    slide_path: str,
    x: int,
    y: int,
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

    tile = _read_tile_at(slide_path, x, y, tile_size, level)
    expected = np.load(expected_array_path)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
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


def run_embed_case(
    model_id: str,
    slide_path: str,
    x: int,
    y: int,
    expected_array_path: Path | str,
    tile_size: int = 224,
    level: int = 0,
    timeout_s: float = 1200.0,
    min_cosine_similarity: float = 0.999,
) -> None:
    expected_array_path = Path(expected_array_path)
    if not expected_array_path.exists():
        pytest.fail(f"Reference file does not exist: {expected_array_path}")

    tile = _read_tile_at(slide_path, x, y, tile_size, level)
    expected = np.load(expected_array_path).flatten().astype(np.float32)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
        t0 = perf_counter()
        actual = (
            np.asarray(client.models.embed_image(model=model_id, image=tile))
            .flatten()
            .astype(np.float32)
        )
        elapsed = perf_counter() - t0

    similarity = float(
        np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    print(
        f"\n  model={model_id} | tile={tile_size}px | time={elapsed:.2f}s | shape={actual.shape} | cosine_similarity={similarity:.6f}"
    )

    if actual.shape != expected.shape:
        pytest.fail(f"Shape mismatch: expected={expected.shape}, actual={actual.shape}")

    if similarity < min_cosine_similarity:
        pytest.fail(
            f"Embedding similarity too low: {similarity:.6f} < {min_cosine_similarity}"
        )
