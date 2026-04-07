# kubectl apply -n rationai-jobs-ns -f c:\Users\jiris\muni-dp\dp\model-service\ray-service.yaml
# kubectl get pods -n rationai-jobs-ns | Select-String "episeg" (model name)
# kubectl cp tests/benchmark_batch_size.py rationai-jobs-ns/rayservice-model-optimized-7zwlk-head-fbzr5:/tmp/benchmark_batch_size.py
# kubectl exec -n rationai-jobs-ns rayservice-model-optimized-7zwlk-head-fbzr5 -- bash -c "python3 -u /tmp/benchmark_batch_size.py --url http://localhost:8000/episeg-1/ --batch-size 128"

# kubectl exec -n rationai-jobs-ns rayservice-model-optimized-7zwlk-head-fbzr5 -- bash -c "pip install httpx -q && python3 -u /tmp/benchmark_batch_size.py --url http://localhost:8000/episeg-1/ --batch-size 8 --concurrency-values 4,8,16,24,32,48,64 --tile-size 1024 --n 500 --warmup 100"

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import lz4.frame
import numpy as np


try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILE_SIZE_DEFAULT = 224
POOL_SIZE = 64
OUTPUT_CSV = "results.csv"


def make_pool(tile_size: int, n: int = POOL_SIZE) -> list[bytes]:
    rng = np.random.default_rng(seed=42)
    pool = []
    for _ in range(n):
        img = rng.integers(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)
        pool.append(lz4.frame.compress(img.tobytes()))
    return pool


async def run_batch(
    url: str,
    pool: list[bytes],
    total: int,
    concurrency: int,
    timeout: float,
) -> tuple[float, int, int]:
    """Pošle `total` requestů s `concurrency` souběžnými workery."""
    remaining = total
    ok = 0
    fail = 0
    pool_len = len(pool)
    counter = 0
    lock = asyncio.Lock()

    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
    )

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal remaining, ok, fail, counter
        while True:
            async with lock:
                if remaining <= 0:
                    return
                remaining -= 1
                idx = counter % pool_len
                counter += 1
            payload = pool[idx]
            try:
                r = await client.post(
                    url,
                    content=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=timeout,
                )
                if r.status_code == 200:
                    ok += 1
                else:
                    fail += 1
                    print(f"  [WARN] HTTP {r.status_code}: {r.text[:120]}")
            except Exception as e:
                fail += 1
                print(f"  [ERR] {type(e).__name__}: {e!r}")

    t0 = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        await asyncio.gather(*[worker(client) for _ in range(concurrency)])
    return time.perf_counter() - t0, ok, fail


def append_csv(path: str, row: dict) -> None:
    fieldnames = [
        "url",
        "batch_size",
        "concurrency",
        "n",
        "elapsed_s",
        "throughput_img_s",
        "ok",
        "fail",
    ]
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_csv(path: str, url: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["url"] == url]


def concurrency_sweep_values(batch_size: int) -> list[int]:
    """Pro MIG-2g.20gb: testujeme rozsah od batch_size/2 do batch_size*4.
    Jemnější kroky kolem batch_size kde bývá knee.
    """
    half = max(1, batch_size // 2)
    candidates = sorted(
        set(
            [
                half,
                batch_size,
                batch_size + batch_size // 2,
                batch_size * 2,
                batch_size * 3,
                batch_size * 4,
            ]
        )
    )
    # Přidej mezikroky kolem batch_size
    extras = [batch_size - batch_size // 4, batch_size + batch_size // 4]
    candidates = sorted(set(candidates + [e for e in extras if e > 0]))
    return candidates


def print_summary(rows: list[dict], batch_size: int | None = None) -> None:
    if not rows:
        return
    if batch_size is not None:
        rows = [r for r in rows if int(r["batch_size"]) == batch_size]
    if not rows:
        return

    best = max(rows, key=lambda r: float(r["throughput_img_s"]))

    header = f"{'batch_size':>12} {'concurrency':>12} {'throughput img/s':>18} {'ok':>8} {'fail':>8}"
    print(header)
    print("-" * len(header))
    for row in sorted(
        rows, key=lambda r: (int(r["batch_size"]), int(r["concurrency"]))
    ):
        marker = " ← BEST" if row is best else ""
        fail_val = int(row["fail"])
        fail_str = f"[!]{fail_val}" if fail_val > 0 else str(fail_val)
        print(
            f"{row['batch_size']:>12} {row['concurrency']:>12}"
            f" {row['throughput_img_s']:>18} {row['ok']:>8} {fail_str:>8}{marker}"
        )
    print()
    print("Doporučené YAML hodnoty pro batch_size =", best["batch_size"])
    tor = int(best["concurrency"])
    mor = int(tor * 1.25) + 8
    print(f"  max_batch_size:           {best['batch_size']}")
    print(f"  target_ongoing_requests:  {tor}   # = nejlepší concurrency")
    print(f"  max_ongoing_requests:     {mor}   # target * 1.25 + buffer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/virchow2/",
        help="Endpointová URL (default: http://localhost:8000/virchow2/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="max_batch_size nastavený v user_config (shodný s YAML)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Pevná hodnota concurrency – přeskočí sweep a naměří jen tuto",
    )
    parser.add_argument(
        "--concurrency-values",
        type=str,
        default=None,
        help="Čárkami oddělený seznam concurrency hodnot k otestování, "
        "např. '32,64,128,256'  (přepíše výchozí sweep)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Počet měřených requestů na jeden bod (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup requesty před měřením (default: 100)",
    )
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_DEFAULT)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help=f"Výstupní CSV soubor (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Přeskočí (batch_size, concurrency) kombinace už změřené v CSV",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"
    pool = make_pool(args.tile_size)

    # Determine sweep values
    if args.concurrency is not None:
        sweep = [args.concurrency]
    elif args.concurrency_values:
        sweep = [int(v.strip()) for v in args.concurrency_values.split(",")]
    else:
        sweep = concurrency_sweep_values(args.batch_size)

    # Already measured (for --skip-existing)
    existing: set[int] = set()
    if args.skip_existing:
        for row in load_csv(args.output, url):
            if int(row["batch_size"]) == args.batch_size:
                existing.add(int(row["concurrency"]))

    print("=" * 60)
    print("Virchow2 Benchmark Sweep")
    print("=" * 60)
    print(f"URL:              {url}")
    print(f"max_batch_size:   {args.batch_size}  (musí odpovídat YAML!)")
    print(f"concurrency sweep:{sweep}")
    print(f"n per point:      {args.n}")
    print(f"warmup:           {args.warmup}")
    print(f"output:           {args.output}")
    print()

    # Warmup – jednou, s prostředním concurrency
    warmup_conc = sweep[len(sweep) // 2]
    print(f"Warmup ({args.warmup} img, concurrency={warmup_conc})...")
    await run_batch(url, pool, args.warmup, warmup_conc, args.timeout)
    print("Warmup done.\n")

    results_this_run: list[dict] = []

    for conc in sweep:
        if conc in existing:
            print(f"[SKIP] concurrency={conc} (already in CSV)")
            continue

        print(f"▶ batch_size={args.batch_size}  concurrency={conc}  ({args.n} img)...")
        elapsed, ok, fail = await run_batch(url, pool, args.n, conc, args.timeout)
        rps = ok / elapsed if elapsed > 0 else 0.0

        row = {
            "url": url,
            "batch_size": args.batch_size,
            "concurrency": conc,
            "n": ok + fail,
            "elapsed_s": f"{elapsed:.2f}",
            "throughput_img_s": f"{rps:.1f}",
            "ok": ok,
            "fail": fail,
        }
        append_csv(args.output, row)
        results_this_run.append(row)

        status = f"  → {rps:.1f} img/s"
        if fail:
            status += f"  [{fail} failures!]"
        print(status)

        # Kratká pauza mezi body aby se server stabilizoval
        await asyncio.sleep(2)

    # Summary – jen aktuální batch_size
    print()
    print("=" * 60)
    print(f"Výsledky pro batch_size = {args.batch_size}")
    print("=" * 60)
    all_rows = load_csv(args.output, url)
    print_summary(all_rows, batch_size=args.batch_size)

    # Pokud existují data pro více batch_size, ukaž i celkové porovnání
    all_batch_sizes = sorted(set(int(r["batch_size"]) for r in all_rows))
    if len(all_batch_sizes) > 1:
        print()
        print("=" * 60)
        print("Celkové porovnání všech batch_size (best concurrency per batch)")
        print("=" * 60)
        # Pro každý batch_size vyber jen nejlepší concurrency
        best_per_batch = []
        for bs in all_batch_sizes:
            candidates = [r for r in all_rows if int(r["batch_size"]) == bs]
            if candidates:
                best_per_batch.append(
                    max(candidates, key=lambda r: float(r["throughput_img_s"]))
                )
        print_summary(best_per_batch)


if __name__ == "__main__":
    asyncio.run(main())
