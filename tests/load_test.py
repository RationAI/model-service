# kubectl cp tests/load_test.py rationai-jobs-ns/rayservice-model-virchow2-5qfmz-head-98tbv:/tmp/load_test.py
# kubectl exec -n rationai-jobs-ns rayservice-model-virchow2-5qfmz-head-98tbv -- bash -c "python3 -u /tmp/load_test.py --url http://localhost:8000/virchow2/ --tiles 5000 --concurrency 128"
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field

import lz4.frame
import numpy as np


try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)


TILE_SIZE_DEFAULT = 224
POOL_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pool(tile_size: int, n: int = POOL_SIZE) -> list[bytes]:
    rng = np.random.default_rng(seed=42)
    pool = []
    for _ in range(n):
        img = rng.integers(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)
        pool.append(lz4.frame.compress(img.tobytes()))
    return pool


@dataclass
class Stats:
    ok: int = 0
    fail_503: int = 0
    fail_other: int = 0
    latencies: list[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def total(self) -> int:
        return self.ok + self.fail_503 + self.fail_other

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]


async def send_tile(
    client: httpx.AsyncClient,
    url: str,
    payload: bytes,
    stats: Stats,
    timeout: float,
    progress_every: int,
) -> None:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            url,
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
        )
        latency = time.perf_counter() - t0
        async with stats.lock:
            if r.status_code == 200:
                stats.ok += 1
                stats.latencies.append(latency)
                if stats.ok % progress_every == 0:
                    print(
                        f"  ✓ {stats.ok} OK  |  503: {stats.fail_503}  |  other: {stats.fail_other}"
                    )
            elif r.status_code == 503:
                stats.fail_503 += 1
            else:
                stats.fail_other += 1
                print(f"  [WARN] HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        async with stats.lock:
            stats.fail_other += 1
        print(f"  [ERR] {e}")


async def run_wsi(
    url: str,
    pool: list[bytes],
    tiles: int,
    concurrency: int,
    timeout: float,
    wsi_id: int,
    stats: Stats,
) -> float:
    """Simuluje jeden WSI — pošle `tiles` requestů s max `concurrency` souběžně."""
    semaphore = asyncio.Semaphore(concurrency)
    pool_len = len(pool)

    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
    )

    async def bounded_send(client: httpx.AsyncClient, idx: int) -> None:
        async with semaphore:
            await send_tile(
                client,
                url,
                pool[idx % pool_len],
                stats,
                timeout,
                progress_every=max(tiles // 10, 100),
            )

    t0 = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [bounded_send(client, i) for i in range(tiles)]
        await asyncio.gather(*tasks)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", default="http://localhost:8000/virchow2/", help="Endpoint URL"
    )
    parser.add_argument(
        "--tiles",
        type=int,
        default=5000,
        help="Počet dlaždic na jeden WSI (default: 5000)",
    )
    parser.add_argument(
        "--wsi-count",
        type=int,
        default=1,
        help="Počet paralelních WSI slidů (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Max souběžných requestů na WSI (default: 128, "
        "mělo by odpovídat target_ongoing_requests)",
    )
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_DEFAULT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout na jeden request v sekundách (default: 120)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup requestů před testem (default: 50)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Přeskočit warmup")
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/"
    pool = make_pool(args.tile_size)
    total_tiles = args.tiles * args.wsi_count

    print("=" * 60)
    print("Virchow2 WSI Load Test")
    print("=" * 60)
    print(f"URL:              {url}")
    print(f"Tiles per WSI:    {args.tiles:,}")
    print(f"WSI count:        {args.wsi_count}")
    print(f"Total tiles:      {total_tiles:,}")
    print(f"Concurrency/WSI:  {args.concurrency}")
    print(f"Total concurrent: {args.concurrency * args.wsi_count}")
    print(f"Request timeout:  {args.timeout}s")
    print()

    # Warmup
    if not args.no_warmup:
        print(f"Warmup ({args.warmup} tiles)...")
        warmup_stats = Stats()
        await run_wsi(
            url,
            pool,
            args.warmup,
            min(args.concurrency, 32),
            args.timeout,
            wsi_id=0,
            stats=warmup_stats,
        )
        print(
            f"Warmup done (ok={warmup_stats.ok}, fail={warmup_stats.fail_503 + warmup_stats.fail_other}).\n"
        )

    # Actual test
    stats = Stats()
    print(
        f"▶ Spouštím {'paralelně ' + str(args.wsi_count) + ' WSI' if args.wsi_count > 1 else '1 WSI'}  "
        f"({total_tiles:,} tiles celkem)...\n"
    )

    t0 = time.perf_counter()

    if args.wsi_count == 1:
        await run_wsi(
            url, pool, args.tiles, args.concurrency, args.timeout, wsi_id=0, stats=stats
        )
    else:
        # Všechny WSI slidy spustit paralelně — simulace více scannerů najednou
        await asyncio.gather(
            *[
                run_wsi(
                    url,
                    pool,
                    args.tiles,
                    args.concurrency,
                    args.timeout,
                    wsi_id=i,
                    stats=stats,
                )
                for i in range(args.wsi_count)
            ]
        )

    elapsed = time.perf_counter() - t0
    rps = stats.ok / elapsed if elapsed > 0 else 0.0

    # Report
    print()
    print("=" * 60)
    print("Výsledky")
    print("=" * 60)
    print(f"Celkový čas:      {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"Throughput:       {rps:.1f} img/s")
    print()
    print(
        f"Úspěšné:          {stats.ok:,} / {total_tiles:,}  ({100 * stats.ok / total_tiles:.1f}%)"
    )
    print(
        f"503 backpressure: {stats.fail_503:,}  ({100 * stats.fail_503 / total_tiles:.1f}%)"
    )
    print(f"Jiné chyby:       {stats.fail_other:,}")
    print()
    if stats.latencies:
        print("Latence (úspěšné requesty):")
        print(f"  p50:  {stats.percentile(50) * 1000:.0f} ms")
        print(f"  p90:  {stats.percentile(90) * 1000:.0f} ms")
        print(f"  p99:  {stats.percentile(99) * 1000:.0f} ms")
        print(f"  max:  {max(stats.latencies) * 1000:.0f} ms")
    print()

    # Verdict
    fail_rate = (stats.fail_503 + stats.fail_other) / total_tiles
    if fail_rate == 0:
        print("✅ PASS — žádné chyby, nastavení je v pořádku pro WSI.")
    elif fail_rate < 0.01:
        print(
            f"⚠️  WARN — {fail_rate * 100:.2f}% chyb. Zvažte zvýšení max_queued_requests."
        )
    else:
        print(
            f"❌ FAIL — {fail_rate * 100:.1f}% chyb. Nastavení nestačí pro tento objem."
        )
        print("   → Zvyšte max_queued_requests nebo snižte --concurrency klientů.")


if __name__ == "__main__":
    asyncio.run(main())
