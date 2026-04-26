import json
from pathlib import Path

import numpy as np
from _shared import _client, _models_base_url, _read_tile


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
        "label": "colon",
        "slide_path": "/mnt/data/MOU/colon/comparison_of_scanners/FLASH2021_5638-02-T.mrxs",
        "model_id": "prostate-classifier-1",
        "type": "binary",
        "tile_size": 512,
        "level": 0,
    },
]


def generate_references() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Generating references to {OUT_DIR} via {_models_base_url()} ==")

    with _client(timeout_s=1200) as client:
        for case in CASES:
            label, model_id, mtype = case["label"], case["model_id"], case["type"]
            print(f"\n[{label}] {model_id} ({mtype})")

            try:
                tile = _read_tile(case["slide_path"], case["tile_size"], case["level"])
            except Exception as e:
                print(f"  -> Failed to read tile: {e}")
                continue

            try:
                if mtype == "binary":
                    score = float(
                        client.models.classify_image(
                            model=model_id, image=tile, timeout=600
                        )
                    )
                    out_file = OUT_DIR / f"{label}_{model_id}_expected.json"
                    with out_file.open("w") as f:
                        json.dump({"expected_score": score}, f, indent=2)
                    print(f"  -> Saved {out_file}")

                elif mtype == "semantic":
                    arr = np.asarray(
                        client.models.segment_image(
                            model=model_id, image=tile, timeout=1200
                        )
                    )
                    out_file = OUT_DIR / f"{label}_{model_id}_expected.npy"
                    np.save(out_file, arr)
                    print(f"  -> Saved {out_file} shape={arr.shape}")

            except Exception as e:
                print(f"  -> ERROR: {e}")


if __name__ == "__main__":
    generate_references()
