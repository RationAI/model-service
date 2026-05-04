import json
import os
from pathlib import Path

import numpy as np
from _shared import _read_tile_at
from rationai import Client


OUT_DIR = Path("/mnt/test_refs")
MODELS_BASE_URL = os.environ.get(
    "MODEL_SERVICE_MODELS_BASE_URL",
    "http://rayservice-model-tests-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
)
BINARY_POSITIVE_THRESHOLD = 0.5

CASES = [
    {
        "label": "prostate_positive",
        "slide_path": "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_2386-06-1.mrxs",
        "model_id": "prostate-classifier-1",
        "type": "binary",
        "tile_size": 512,
        "level": 0,
        "x": 43390,
        "y": 45865,
    },
    {
        "label": "prostate_negative",
        "slide_path": "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_0845-02-0.mrxs",
        "model_id": "prostate-classifier-1",
        "type": "binary",
        "tile_size": 512,
        "level": 0,
        "x": 31017,
        "y": 113220,
    },
]


def generate_references() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Generating references to {OUT_DIR} via {MODELS_BASE_URL} ==")

    with Client(models_base_url=MODELS_BASE_URL, timeout=1200) as client:
        for case in CASES:
            label, model_id, mtype = case["label"], case["model_id"], case["type"]
            print(f"\n[{label}] {model_id} ({mtype})")

            try:
                tile = _read_tile_at(
                    case["slide_path"],
                    case["x"],
                    case["y"],
                    case["tile_size"],
                    case["level"],
                )
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
                        json.dump(
                            {
                                "label": label,
                                "model_id": model_id,
                                "slide_path": case["slide_path"],
                                "x": case["x"],
                                "y": case["y"],
                                "tile_size": case["tile_size"],
                                "level": case["level"],
                                "threshold": BINARY_POSITIVE_THRESHOLD,
                                "expected_is_positive": score
                                >= BINARY_POSITIVE_THRESHOLD,
                                "expected_score": score,
                            },
                            f,
                            indent=2,
                        )
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

                elif mtype == "embed":
                    arr = np.asarray(
                        client.models.embed_image(
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
