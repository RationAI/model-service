# Model snapshot tests

This repository contains end-to-end snapshot tests in `tests/model_snapshots/`.

Per-model test files:

- `tests/model_snapshots/test_binary_classifier_model_snapshot.py`
- `tests/model_snapshots/test_semantic_segmentation_model_snapshot.py`

Shared files:

- `tests/model_snapshots/_shared.py`
- `tests/model_snapshots/run_all_model_snapshot_tests.py`

These tests are meant as post-deploy use-case checks (not only liveness checks):

- they execute a real request path through Ray Serve deployment
- they verify request processing success (timeouts/errors fail the test)
- they verify result correctness for each deployment (`binary_classifier`, `semantic_segmentation`)
- they touch real slide paths, helping catch mount/filesystem issues

Each test calls its deployment-specific endpoint:

- binary classifier: SDK call `client.models.classify_image("prostate-classifier-1", tile)`
- semantic segmentation: SDK call `client.models.segment_image("episeg-1", tile)`

Input tile is read directly from a real WSI using `ratiopath.openslide.OpenSlide`.

## Adding a new model test

Přidání nového modelu do testů je nyní velmi jednoduché:

1. Vytvořte nový soubor v `tests/model_snapshots/`, např. `test_novy_model_snapshot.py`.
2. Importujte a zavolejte příslušnou case funkci z `_shared.py` a předejte jí konfiguraci napřímo parametrem:

```python
from pathlib import Path
from tests.model_snapshots._shared import run_binary_classifier_case

def test_novy_model_snapshot() -> None:
    # Parametry si rovnou zadefinujte v testovacím souboru
    run_binary_classifier_case(
        model_id="my-new-endpoint-id",
        slide_path="/mnt/bioptic_tree/.../slide.mrxs",
        expected_score=0.987,
        tile_size=512,
        level=0,
    )
```

Tím se stane automaticky součástí sady `pytest tests/model_snapshots`.

## Global environment variables

Common (pro celý cluster a všechny testy):

- `MODEL_SERVICE_MODELS_BASE_URL` (default: `http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000`)

Očekávané skóre/pole a cesty k datasetům pro stávající modely (`episeg-1` a `prostate-classifier-1`) se tahají z těchto proměnných ve stávajících testovacích souborech, pokud chcete zachovat původní CI chování (případně se dají časem snadno zahardkódit do testovacího souboru):

- `MODEL_TEST_BINARY_EXPECTED_SCORE`
- `MODEL_TEST_SEMANTIC_EXPECTED_ARRAY_PATH`

## Example (PowerShell)

```powershell
$env:MODEL_TEST_BINARY_EXPECTED_SCORE = "0.9732"
$env:MODEL_TEST_SEMANTIC_EXPECTED_ARRAY_PATH = "/mnt/path/to/reference/semantic_expected.npy"

# Models base URL is resolved directly from SDK fallback inside kubernetes:
# http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000

python tests/model_snapshots/run_all_model_snapshot_tests.py

# Alternative:
python -m pytest tests/model_snapshots -q
```

## SDK dependency

Install SDK package so that `import rationai` works in tests, e.g.:

```powershell
python -m pip install git+https://github.com/RationAI/rationai-sdk-python.git
```
