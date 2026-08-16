"""The worker agent: a small HTTP service the GPU machine runs.

Phase 1 scope is deliberately narrow -- identity, capability reporting and link
measurement. No code execution yet. Running a friend's Python on your machine is
a real security decision and it gets its own phase with a sandbox, not a
convenience endpoint bolted on here.
"""

from __future__ import annotations

import hmac
import os
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from p2pgpu import __version__
from p2pgpu.common.config import AUTH_HEADER, cluster_token, node_id
from p2pgpu.common.schema import HealthResponse, NodeCapabilities
from p2pgpu.worker.probe import probe

MAX_BENCH_BYTES = 256 * 1024 * 1024
_STARTED = time.monotonic()


def require_token(
    token: str | None = Header(default=None, alias=AUTH_HEADER),
) -> None:
    """Reject callers without the cluster secret.

    compare_digest keeps the check constant-time; a timing oracle on a LAN is a
    genuinely practical attack.
    """
    expected = cluster_token()
    if expected is None:
        raise HTTPException(
            status_code=500,
            detail="worker has no cluster token configured; run 'p2pgpu init'",
        )
    if token is None or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid or missing cluster token")


app = FastAPI(title="p2pgpu worker", version=__version__)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Unauthenticated on purpose: it is the RTT probe and carries no secrets."""
    return HealthResponse(
        node_id=node_id(),
        agent_version=__version__,
        uptime_s=round(time.monotonic() - _STARTED, 3),
    )


@app.get(
    "/v1/capabilities",
    response_model=NodeCapabilities,
    dependencies=[Depends(require_token)],
)
def capabilities() -> NodeCapabilities:
    # Probed per request rather than cached: free VRAM is the whole point and it
    # changes the moment someone opens a game.
    return probe()


@app.post("/v1/bench/sink", dependencies=[Depends(require_token)])
async def bench_sink(request: Request) -> dict[str, int]:
    """Swallow an upload and report how much arrived -> measures client upstream."""
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BENCH_BYTES:
            raise HTTPException(status_code=413, detail="bench payload too large")
    return {"received_bytes": total}


@app.get("/v1/bench/source", dependencies=[Depends(require_token)])
def bench_source(size_bytes: int = 8 * 1024 * 1024) -> Response:
    """Emit random bytes -> measures client downstream.

    Random rather than zeros so nothing along the path can compress the payload
    and report a throughput you will never see with real activation tensors.
    """
    if size_bytes <= 0 or size_bytes > MAX_BENCH_BYTES:
        raise HTTPException(status_code=400, detail="size_bytes out of range")
    return Response(content=os.urandom(size_bytes), media_type="application/octet-stream")


def serve(host: str = "0.0.0.0", port: int = 8777) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
