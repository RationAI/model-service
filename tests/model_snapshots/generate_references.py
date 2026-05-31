import json
import os
from typing import TypedDict

import numpy as np
from rationai import Client

from tests.model_snapshots._shared import _read_tile_at, test_refs_dir


OUT_DIR = test_refs_dir()
MODELS_BASE_URL = os.environ.get(
    "MODEL_SERVICE_MODELS_BASE_URL",
    "http://rayservice-model-tests-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
)
BINARY_POSITIVE_THRESHOLD = 0.5


class CaseConfig(TypedDict):
    label: str
    slide_path: str
    model_id: str
    type: str
    tile_size: int
    level: int
    x: int
    y: int


# Keep only one active case here. Store other candidate slides in new_images.txt
# and swap them in when you want to regenerate a different reference.
ACTIVE_CASE: CaseConfig = {
    "label": "prov-gigapath",
    "slide_path": "/mnt/data/MOU/prostate/tile_level_annotations/P-2016_1367-01-0.mrxs",
    "model_id": "prov-gigapath",
    "type": "embed",
    "tile_size": 224,
    "level": 0,
    "x": 40000,
    "y": 70000,
}

CASES: list[CaseConfig] = [ACTIVE_CASE]


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
