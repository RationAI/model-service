from pathlib import Path

import pytest
from _shared import run_semantic_segmentation_case


@pytest.mark.parametrize(
    "label, slide_path, x, y",
    [
        (
            "colorectum_kos04",
            "/mnt/data/MOU/colorectum/tissue_microarray/he/KOS04.mrxs",
            46000,
            82400,
        ),
    ],
)
def test_semantic_episeg(label: str, slide_path: str, x: int, y: int) -> None:
    model_id = "episeg-1"
    expected_array_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.npy")

    run_semantic_segmentation_case(
        model_id=model_id,
        slide_path=slide_path,
        x=x,
        y=y,
        expected_array_path=expected_array_path,
        tile_size=1024,
        level=0,
        epithelium_threshold=0.5,
        min_epithelium_fraction=0.01,
        case_name=label,
    )
