import json
from pathlib import Path

import httpx
import lz4.frame
import numpy as np
from _shared import _models_base_url, _read_tile_from_slide


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
]


def generate_references() -> None:
    base_url = _models_base_url()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Generating references to {OUT_DIR} via {base_url} ==")

    for case in CASES:
        label, model_id, mtype = case["label"], case["model_id"], case["type"]
        print(f"\n[{label}] {model_id} ({mtype})")

        try:
            tile = _read_tile_from_slide(
                case["slide_path"], case["tile_size"], case["level"]
            )
        except Exception as e:
            print(f"  -> Failed to read tile: {e}")
            continue

        url = f"{base_url}/{model_id}/"
        data = lz4.frame.compress(tile.tobytes())

        try:
            response = httpx.post(url, content=data, timeout=600)
            response.raise_for_status()

            if mtype == "binary":
                out_file = OUT_DIR / f"{label}_{model_id}_expected.json"
                with out_file.open("w") as f:
                    json.dump({"expected_score": float(response.json())}, f, indent=2)
                print(f"  -> Saved {out_file}")

            elif mtype == "semantic":
                h, w = tile.shape[:2]
                arr = np.frombuffer(
                    lz4.frame.decompress(response.content), dtype=np.float16
                ).reshape(-1, h, w)
                out_file = OUT_DIR / f"{label}_{model_id}_expected.npy"
                np.save(out_file, arr)
                print(f"  -> Saved {out_file} shape={arr.shape}")

        except Exception as e:
            print(f"  -> ERROR: {e}")


if __name__ == "__main__":
    generate_references()
