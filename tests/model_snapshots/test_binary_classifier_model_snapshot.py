import json
from pathlib import Path

import pytest
from _shared import run_binary_classifier_case


@pytest.mark.parametrize(
    "label, slide_path, x, y",
    [
        (
            "prostate_positive",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_2386-06-1.mrxs",
            43390,
            45865,
        ),
    ],
)
def test_prostate_classifier_positive(
    label: str, slide_path: str, x: int, y: int
) -> None:
    model_id = "prostate-classifier-1"
    json_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.json")

    if not json_path.exists():
        pytest.skip(
            f"Reference file {json_path} missing. Run generate_references.py first."
        )

    with json_path.open() as f:
        expected_score = json.load(f)["expected_score"]

    assert expected_score >= 0.5, (
        f"Reference score {expected_score:.4f} is below positive threshold 0.5 — "
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
    )
