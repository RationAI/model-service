from pathlib import Path

import pytest

from tests.model_snapshots._shared import run_embed_case


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "label, slide_path, x, y",
    [
        (
            "prov-gigapath",
            "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_1367-01-0.mrxs",
            40000,
            70000,
        ),
    ],
)
def test_prov_gigapath(label: str, slide_path: str, x: int, y: int) -> None:
    model_id = "prov-gigapath"
    expected_array_path = Path(f"/mnt/test_refs/{label}_{model_id}_expected.npy")

    run_embed_case(
        model_id=model_id,
        slide_path=slide_path,
        x=x,
        y=y,
        expected_array_path=expected_array_path,
        tile_size=224,
        level=0,
        case_name=label,
    )
