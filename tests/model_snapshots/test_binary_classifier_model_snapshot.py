import json
from pathlib import Path

import pytest
from _shared import run_binary_classifier_case


BINARY_POSITIVE_THRESHOLD = 0.5


@pytest.mark.parametrize(
    "label, slide_path, x, y, is_positive",
    [
        (
            "prostate_positive",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_2386-06-1.mrxs",
            43390,
            45865,
            True,
        ),
        (
            "prostate_negative",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_0845-02-0.mrxs",
            32950,
            108990,
            False,
        ),
    ],
)
def test_prostate_classifier_snapshot(
    label: str, slide_path: str, x: int, y: int, is_positive: bool
) -> None:
    model_id = "prostate-classifier-1"
    json_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.json")

    if not json_path.exists():
        pytest.skip(
            f"Reference file {json_path} missing. Run generate_references.py first."
        )

    with json_path.open() as f:
        reference = json.load(f)

    assert reference.get("label") == label
    assert reference.get("model_id") == model_id
    assert reference.get("slide_path") == slide_path
    assert reference.get("x") == x
    assert reference.get("y") == y
    assert reference.get("tile_size") == 512
    assert reference.get("level") == 0

    expected_score = reference["expected_score"]
    threshold = reference.get("threshold", BINARY_POSITIVE_THRESHOLD)
    expected_is_positive = reference.get("expected_is_positive")
    assert expected_is_positive is not None
    assert expected_is_positive == is_positive

    if is_positive:
        assert expected_score >= threshold, (
            f"Reference score {expected_score:.4f} is below positive threshold {threshold:.3f} — "
            "was the reference generated on the correct tile?"
        )
    else:
        assert expected_score < threshold, (
            f"Reference score {expected_score:.4f} is above negative threshold {threshold:.3f} — "
            "was the reference generated on the correct tile?"
        )

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
