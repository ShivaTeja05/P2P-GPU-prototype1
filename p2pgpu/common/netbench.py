"""Measure the path between the two machines.

Worth doing before you start, because the link -- not the GPU -- is what makes
remote work feel slow:

  * Upload speed decides how long pushing a dataset takes.
  * Download speed decides how long pulling a checkpoint back takes.
  * RTT decides how laggy an interactive notebook feels.

A 4080 you can only feed at 5 Mbps is a 4080 you will mostly be waiting on.
"""

from __future__ import annotations

import statistics
import time

import httpx

from p2pgpu.common.config import AUTH_HEADER
from p2pgpu.common.schema import LinkMeasurement


def _auth(token: str | None) -> dict[str, str]:
    return {AUTH_HEADER: token} if token else {}


def measure_rtt(base_url: str, samples: int = 20, timeout: float = 10.0) -> list[float]:
    """Serial GET /health timings, in milliseconds.

    Serial and single-connection on purpose -- this measures the latency a
    pipeline stage will actually feel, not what a parallel benchmark can hide.
    """
    timings: list[float] = []
    with httpx.Client(timeout=timeout) as client:
        # One warm-up to pay TCP + TLS setup outside the measurement.
        try:
            client.get(f"{base_url}/health")
        except httpx.HTTPError:
            pass
        for _ in range(samples):
            start = time.perf_counter()
            resp = client.get(f"{base_url}/health")
            resp.raise_for_status()
            timings.append((time.perf_counter() - start) * 1000)
    return timings


def measure_upload(base_url: str, token: str | None, size_bytes: int, timeout: float = 120.0) -> float:
    """Mbps from this machine to the peer."""
    payload = b"\x00" * size_bytes
    with httpx.Client(timeout=timeout) as client:
        start = time.perf_counter()
        resp = client.post(
            f"{base_url}/v1/bench/sink",
            content=payload,
            headers={**_auth(token), "content-type": "application/octet-stream"},
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
    return (size_bytes * 8) / elapsed / 1e6 if elapsed > 0 else 0.0


def measure_download(base_url: str, token: str | None, size_bytes: int, timeout: float = 120.0) -> float:
    """Mbps from the peer to this machine."""
    received = 0
    with httpx.Client(timeout=timeout) as client:
        start = time.perf_counter()
        with client.stream(
            "GET",
            f"{base_url}/v1/bench/source",
            params={"size_bytes": size_bytes},
            headers=_auth(token),
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                received += len(chunk)
        elapsed = time.perf_counter() - start
    return (received * 8) / elapsed / 1e6 if elapsed > 0 else 0.0


def benchmark_link(
    base_url: str,
    token: str | None,
    rtt_samples: int = 20,
    payload_mb: int = 8,
) -> LinkMeasurement:
    base_url = base_url.rstrip("/")
    timings = measure_rtt(base_url, samples=rtt_samples)
    size = payload_mb * 1024 * 1024

    up = down = None
    if token:
        up = measure_upload(base_url, token, size)
        down = measure_download(base_url, token, size)

    ordered = sorted(timings)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return LinkMeasurement(
        peer=base_url,
        rtt_ms_p50=round(statistics.median(timings), 2),
        rtt_ms_p95=round(ordered[p95_index], 2),
        rtt_ms_min=round(min(timings), 2),
        upload_mbps=round(up, 2) if up is not None else None,
        download_mbps=round(down, 2) if down is not None else None,
        samples=len(timings),
    )


def estimate_transfer_s(size_mb: float, mbps: float | None) -> float | None:
    """Seconds to move size_mb megabytes at mbps megabits/second."""
    if not mbps or mbps <= 0:
        return None
    return round(size_mb * 8 / mbps, 1)


def transfer_estimates(
    upload_mbps: float | None,
    download_mbps: float | None,
) -> list[tuple[str, float, str]]:
    """Practical 'how long will I wait' table.

    Returns (label, size_mb, human_time) rows for the transfers that actually
    happen when you work on someone else's GPU.
    """
    jobs = [
        ("push a 100 MB dataset", 100.0, "up"),
        ("push a 1 GB dataset", 1000.0, "up"),
        ("pull back a 500 MB checkpoint", 500.0, "down"),
        ("pull back a 5 GB fine-tuned model", 5000.0, "down"),
    ]
    rows = []
    for label, size_mb, direction in jobs:
        mbps = upload_mbps if direction == "up" else download_mbps
        seconds = estimate_transfer_s(size_mb, mbps)
        rows.append((label, size_mb, _human_duration(seconds)))
    return rows


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"
