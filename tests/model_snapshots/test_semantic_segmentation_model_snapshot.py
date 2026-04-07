from pathlib import Path

import pytest

from tests.model_snapshots._shared import run_semantic_segmentation_case


@pytest.mark.parametrize(
    "label, slide_path",
    [
        (
            "breast",
            "/mnt/bioptic_tree/2019/08/728/2019_08728-01-T/2019_08728-01-T.mrxs",
        ),
        (
            "colorectum",
            "/mnt/data/MOU/colorectum/colorectal_cancer_2020-2024-06/2020_00106-01-N.mrxs",
        ),
        (
            "colon",
            "/mnt/data/MOU/colon/comparison_of_scanners/FLASH2021_5638-02-T.mrxs",
        ),
    ],
)
def test_semantic_episeg(label: str, slide_path: str) -> None:
    model_id = "episeg-1"
    expected_array_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.npy")

    run_semantic_segmentation_case(
        model_id=model_id,
        slide_path=slide_path,
        expected_array_path=expected_array_path,
        tile_size=1024,
        level=0,
    )
