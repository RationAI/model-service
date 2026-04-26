from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_base_url() -> str:
    return os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
    )


def _read_tile_from_slide(
    slide_path: str,
    tile_size: int,
    level: int,
) -> NDArray[np.uint8]:
    try:
        from ratiopath.openslide import OpenSlide
    except ImportError:
        pytest.skip("Python package `ratiopath` is not installed.")

    with OpenSlide(slide_path) as slide:
        extent_x, extent_y = slide.level_dimensions[level]
        x = max(0, (extent_x - tile_size) // 2)
        y = max(0, (extent_y - tile_size) // 2)
        tile = slide.read_region_relative(
            (x, y), level, (tile_size, tile_size)
        ).convert("RGB")

    return np.asarray(tile, dtype=np.uint8)


def _client(timeout_s: float = 600.0):
    try:
        from rationai import Client
    except ImportError:
        pytest.skip("Python package `rationai` is not installed.")

    return Client(models_base_url=_models_base_url(), timeout=timeout_s)


def run_binary_classifier_case(
    model_id: str,
    slide_path: str,
    expected_score: float,
    tile_size: int = 512,
    level: int = 0,
    timeout_s: float = 600.0,
    tolerance: float = 0.00001,
) -> None:
    tile = _read_tile_from_slide(
        slide_path=slide_path, tile_size=tile_size, level=level
    )

    with _client(timeout_s=timeout_s) as client:
        actual_score = float(client.models.classify_image(model=model_id, image=tile))

    assert abs(actual_score - expected_score) <= tolerance, (
        f"Binary score mismatch: expected={expected_score}, actual={actual_score}, "
        f"tolerance={tolerance}"
    )


def run_semantic_segmentation_case(
    model_id: str,
    slide_path: str,
    expected_array_path: Path | str,
    tile_size: int = 1024,
    level: int = 0,
    timeout_s: float = 600.0,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    expected_array_path = Path(expected_array_path)
    if not expected_array_path.exists():
        pytest.fail(f"Reference file does not exist: {expected_array_path}")

    tile = _read_tile_from_slide(
        slide_path=slide_path, tile_size=tile_size, level=level
    )
    expected = np.load(expected_array_path)

    with _client(timeout_s=timeout_s) as client:
        actual = np.asarray(client.models.segment_image(model=model_id, image=tile))

    if actual.shape != expected.shape:
        pytest.fail(f"Shape mismatch: expected={expected.shape}, actual={actual.shape}")

    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        mismatch = np.abs(actual.astype(np.float32) - expected.astype(np.float32)).max()
        pytest.fail(
            f"Output mismatch beyond tolerance (atol={atol}, rtol={rtol}, max_abs_diff={mismatch})"
        )


def verify_file_hash(path: Path, expected_hash: str) -> None:
    actual_hash = _sha256(path)
    assert actual_hash == expected_hash, (
        f"Hash mismatch for {path}: expected={expected_hash}, actual={actual_hash}"
    )
