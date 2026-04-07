import json
import os
from pathlib import Path

import numpy as np

from tests.model_snapshots._shared import _client, _read_tile_from_slide


OUT_DIR = Path("/mnt/test_refs")

CASES = [
    {
        "label": "breast",
        "slide_path": "/mnt/bioptic_tree/2019/08/728/2019_08728-01-T/2019_08728-01-T.mrxs",
        "model_id": "episeg-1",
        "type": "semantic",
        "tile_size": 1024,
        "level": 0,
    },
    {
        "label": "breast",
        "slide_path": "/mnt/bioptic_tree/2019/08/728/2019_08728-01-T/2019_08728-01-T.mrxs",
        "model_id": "prostate-classifier-1",
        "type": "binary",
        "tile_size": 512,
        "level": 0,
    },
    {
        "label": "colorectum",
        "slide_path": "/mnt/data/MOU/colorectum/colorectal_cancer_2020-2024-06/2020_00106-01-N.mrxs",
        "model_id": "episeg-1",
        "type": "semantic",
        "tile_size": 1024,
        "level": 0,
    },
    {
        "label": "colon",
        "slide_path": "/mnt/data/MOU/colon/comparison_of_scanners/FLASH2021_5638-02-T.mrxs",
        "model_id": "episeg-1",
        "type": "semantic",
        "tile_size": 1024,
        "level": 0,
    },
]


def generate_references() -> None:
    models_base_url = os.environ.get(
        "MODEL_SERVICE_MODELS_BASE_URL",
        "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Generating references to {OUT_DIR} via {models_base_url} ==")

    with _client(models_base_url=models_base_url, timeout_s=600) as client:
        for case in CASES:
            label = case["label"]
            model_id = case["model_id"]
            mtype = case["type"]
            slide_path = case["slide_path"]
            tile_size = case["tile_size"]
            level = case["level"]

            print(f"\nProcessing [{label}] => Model: {model_id} ({mtype})")
            print(f"Slide path: {slide_path}")

            try:
                tile = _read_tile_from_slide(
                    slide_path=slide_path, tile_size=tile_size, level=level
                )
            except Exception as e:
                print(f"  -> Failed to read tile: {e}")
                continue

            try:
                if mtype == "binary":
                    prediction = client.models.classify_image(
                        model=model_id, image=tile, timeout=600
                    )
                    out_file = OUT_DIR / f"{label}_{model_id}_expected.json"
                    with out_file.open("w") as f:
                        json.dump({"expected_score": float(prediction)}, f, indent=2)
                    print(f"  -> SUCCESS! Saved binary score to {out_file}")

                elif mtype == "semantic":
                    prediction = client.models.segment_image(
                        model=model_id, image=tile, timeout=600
                    )
                    arr = np.asarray(prediction)
                    out_file = OUT_DIR / f"{label}_{model_id}_expected.npy"
                    np.save(out_file, arr)
                    print(
                        f"  -> SUCCESS! Saved semantic array {arr.shape} to {out_file}"
                    )

            except Exception as e:
                print(f"  -> ERROR during prediction/saving: {e}")


if __name__ == "__main__":
    generate_references()
