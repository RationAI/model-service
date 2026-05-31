from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from rationai import Client
from ratiopath.openslide import OpenSlide


def _models_base_url() -> str:
    return os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
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
    atol: float = 1e-6,
    rtol: float = 1e-5,
    case_name: str | None = None,
) -> None:
    tile = _read_tile_at(slide_path, x, y, tile_size, level)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
        actual_score = float(client.models.classify_image(model=model_id, image=tile))

    delta = actual_score - expected_score
    name = case_name or "case"

    if expected_is_positive is not None:
        actual_is_positive = actual_score >= threshold
        assert actual_is_positive == expected_is_positive, (
            f"Binary class mismatch: expected_is_positive={expected_is_positive}, "
            f"actual_score={actual_score:.6f}, threshold={threshold:.3f}"
        )

    if not np.isclose(actual_score, expected_score, rtol=rtol, atol=atol):
        pytest.fail(
            f"Binary score mismatch beyond tolerance (atol={atol}, rtol={rtol}, "
            f"expected={expected_score:.6f}, actual={actual_score:.6f})"
        )

    print(f"\n/{model_id}")
    print(
        f"{name} stats: score={actual_score:.6f} expected={expected_score:.6f} "
        f"delta={delta:+.6f} threshold={threshold:.3f}"
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
    atol: float = 1e-6,
    rtol: float = 1e-5,
    epithelium_threshold: float | None = None,
    min_epithelium_fraction: float | None = None,
    epithelium_channel: int | None = None,
    case_name: str | None = None,
) -> None:
    expected_array_path = Path(expected_array_path)
    if not expected_array_path.exists():
        pytest.fail(f"Reference file does not exist: {expected_array_path}")

    tile = _read_tile_at(slide_path, x, y, tile_size, level)
    expected = np.load(expected_array_path)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
        actual = np.asarray(client.models.segment_image(model=model_id, image=tile))

    max_diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32)).max()

    if actual.ndim == 4:
        stats_slice = actual[0, 0]
    elif actual.ndim == 3:
        stats_slice = actual[0]
    else:
        stats_slice = actual.squeeze()

    stats_slice = stats_slice.astype(np.float32)
    min_val = float(stats_slice.min())
    mean_val = float(stats_slice.mean())
    max_val = float(stats_slice.max())
    frac_05 = float((stats_slice >= 0.5).mean())
    name = case_name or "case"

    if actual.shape != expected.shape:
        pytest.fail(f"Shape mismatch: expected={expected.shape}, actual={actual.shape}")

    close_mask = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if not close_mask.all():
        mismatch_fraction = float((~close_mask).mean())
        pytest.fail(
            "Output mismatch beyond tolerance "
            f"(atol={atol}, rtol={rtol}, max_abs_diff={max_diff}, "
            f"mismatch_fraction={mismatch_fraction:.6f})"
        )

    if epithelium_threshold is not None and min_epithelium_fraction is not None:
        if actual.ndim == 4:
            channel = 0 if epithelium_channel is None else epithelium_channel
            epithelium = actual[0, channel]
        elif actual.ndim == 3:
            channel = 0 if epithelium_channel is None else epithelium_channel
            epithelium = actual[channel]
        else:
            epithelium = actual.squeeze()

        if epithelium.ndim != 2:
            pytest.fail(
                "Cannot determine epithelium channel; provide epithelium_channel explicitly."
            )

        fraction = float((epithelium >= epithelium_threshold).mean())
        if fraction < min_epithelium_fraction:
            pytest.fail(
                "Epithelium coverage too low: "
                f"fraction={fraction:.6f} < min_fraction={min_epithelium_fraction:.6f}"
            )

    print(f"\n/{model_id}")
    print(
        f"{name} stats: shape={actual.shape} max_diff={max_diff:.6f} "
        f"min={min_val:.6f} mean={mean_val:.6f} max={max_val:.6f} "
        f"frac>=0.5={frac_05:.6f}"
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
    case_name: str | None = None,
) -> None:
    expected_array_path = Path(expected_array_path)
    if not expected_array_path.exists():
        pytest.fail(f"Reference file does not exist: {expected_array_path}")

    tile = _read_tile_at(slide_path, x, y, tile_size, level)
    expected = np.load(expected_array_path).flatten().astype(np.float32)

    with Client(models_base_url=_models_base_url(), timeout=timeout_s) as client:
        actual = (
            np.asarray(client.models.embed_image(model=model_id, image=tile))
            .flatten()
            .astype(np.float32)
        )

    similarity = float(
        np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    name = case_name or "case"

    if actual.shape != expected.shape:
        pytest.fail(f"Shape mismatch: expected={expected.shape}, actual={actual.shape}")

    if similarity < min_cosine_similarity:
        pytest.fail(
            f"Embedding similarity too low: {similarity:.6f} < {min_cosine_similarity}"
        )

    print(f"\n/{model_id}")
    print(
        f"{name} stats: shape={actual.shape} cosine_similarity={similarity:.6f} "
        f"norm_actual={actual_norm:.6f} norm_expected={expected_norm:.6f}"
    )
