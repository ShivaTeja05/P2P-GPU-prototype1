"""The worker agent: a small HTTP service the GPU machine runs.

Phase 1 scope is deliberately narrow -- identity, capability reporting and link
measurement. No code execution yet. Running a friend's Python on your machine is
a real security decision and it gets its own phase with a sandbox, not a
convenience endpoint bolted on here.
"""

from __future__ import annotations

import hmac
import os
import sys
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


@app.get("/v1/share", dependencies=[Depends(require_token)])
def current_share() -> dict:
    """What this machine is offering right now.

    This is what removes the copy-paste handoff: the borrower asks each peer
    directly instead of waiting for someone to send them a link.
    """
    from p2pgpu.worker import share as sharing

    session = sharing.load_session()
    if session is None or not sharing.container_running():
        return {"sharing": False}
    return {
        "sharing": True,
        "url": session.url,
        "ssh_command": session.ssh_command,
        "hours_left": round(session.remaining_s / 3600, 2),
        "image": session.image,
    }


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


def port_owner(port: int, host: str = "127.0.0.1") -> str | None:
    """If something is listening, say what. None means the port is free.

    Returns "p2pgpu" when a healthy agent already holds it, "other" for anything
    else. The distinction matters: one is idempotent, the other is a conflict.
    """
    import socket

    import httpx

    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex((host, port)) != 0:
            return None
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=3)
        if resp.status_code == 200 and "node_id" in resp.json():
            return "p2pgpu"
    except Exception:  # noqa: BLE001, S110 - any failure means "not a p2pgpu agent"
        pass
    return "other"


def serve(host: str = "0.0.0.0", port: int = 8777) -> int:
    """Run the agent. Returns a process exit code.

    Exits 0 when an agent is already serving, so a service manager treats this
    as "nothing to do" rather than a failure worth restarting forever. Under
    launchd with KeepAlive that difference is the gap between idempotent and a
    crash loop.
    """
    import uvicorn

    owner = port_owner(port)
    if owner == "p2pgpu":
        print(f"An agent is already serving on port {port}. Nothing to do.")
        return 0
    if owner == "other":
        print(
            f"Port {port} is in use by something that is not a p2pgpu agent.\n"
            f"Free it, or start the agent elsewhere with --port.",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
