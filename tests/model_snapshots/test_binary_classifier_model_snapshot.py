import json

import pytest

from tests.model_snapshots._shared import run_binary_classifier_case, test_refs_dir


BINARY_POSITIVE_THRESHOLD = 0.5


@pytest.mark.parametrize(
    "label, slide_path, x, y",
    [
        (
            "prostate_positive",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_2386-06-1.mrxs",
            43390,
            45865,
        ),
        (
            "prostate_negative",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_0845-02-0.mrxs",
            32950,
            108990,
        ),
    ],
)
def test_prostate_classifier_snapshot(
    label: str, slide_path: str, x: int, y: int
) -> None:
    model_id = "prostate-classifier-1"
    json_path = test_refs_dir() / f"{label}_{model_id}_expected.json"

    with json_path.open() as f:
        reference = json.load(f)

    expected_score = reference["expected_score"]
    threshold = reference.get("threshold", BINARY_POSITIVE_THRESHOLD)
    expected_is_positive = reference.get("expected_is_positive")
    assert expected_is_positive is not None

    run_binary_classifier_case(
        model_id=model_id,
        slide_path=slide_path,
        x=x,
        y=y,
        expected_score=expected_score,
        tile_size=512,
        level=0,
        expected_is_positive=expected_is_positive,
        threshold=threshold,
        case_name=label,
    )
