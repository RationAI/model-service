import json
from pathlib import Path

import pytest
from _shared import run_binary_classifier_case


@pytest.mark.parametrize(
    "label, slide_path",
    [
        (
            "colon",
            "/mnt/data/MOU/colon/comparison_of_scanners/FLASH2021_5638-02-T.mrxs",
        ),
    ],
)
def test_binary_classifier(label: str, slide_path: str) -> None:
    model_id = "prostate-classifier-1"
    json_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.json")

    if json_path.exists():
        with json_path.open() as f:
            expected_score = json.load(f)["expected_score"]
    else:
        pytest.skip(
            f"Reference file {json_path} missing. Run generate_references.py first."
        )

    run_binary_classifier_case(
        model_id=model_id,
        slide_path=slide_path,
        expected_score=expected_score,
        tile_size=512,
        level=0,
    )
