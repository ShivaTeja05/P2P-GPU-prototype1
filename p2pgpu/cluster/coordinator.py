"""The rendezvous point for a training round.

Deliberately tiny. It never sees a GPU, never runs a model, and holds no state
between rounds beyond the tensors currently being averaged. All it provides is
a barrier ("wait until everyone has submitted round N") and an arithmetic mean.

That smallness is what lets it run on the laptop with no GPU -- the machine that
is least useful for training is perfectly good at being a meeting point.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from p2pgpu import __version__
from p2pgpu.worker.agent import require_token

# A round's payload is a full state_dict per node. Generous, but not unbounded:
# somebody will eventually point this at a 7B model and deserve a clear error
# rather than a dead coordinator.
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_ROUND_TIMEOUT_S = 900


@dataclass
class Round:
    """One synchronisation point. Filled by submissions, drained by fetches."""

    number: int
    expected: int
    payloads: dict[str, bytes] = field(default_factory=dict)
    averaged: bytes | None = None
    started_at: float = field(default_factory=time.time)
    complete: threading.Event = field(default_factory=threading.Event)

    @property
    def received(self) -> int:
        return len(self.payloads)


@dataclass
class ClusterState:
    world_size: int = 0
    members: dict[str, float] = field(default_factory=dict)  # node_id -> last seen
    rounds: dict[int, Round] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = ClusterState()
app = FastAPI(title="p2pgpu cluster coordinator", version=__version__)


def _average(payloads: list[bytes]) -> bytes:
    """Element-wise mean of several serialised state_dicts.

    torch is imported here rather than at module scope so the coordinator can
    run on a machine that has no torch at all -- which is the common case, since
    the coordinator is deliberately the GPU-less laptop.
    """
    import torch

    states = [torch.load(io.BytesIO(p), map_location="cpu", weights_only=True) for p in payloads]
    if not states:
        raise ValueError("nothing to average")

    reference = states[0]
    keys = set(reference)
    for other in states[1:]:
        if set(other) != keys:
            missing = keys.symmetric_difference(set(other))
            raise ValueError(
                f"nodes submitted different model shapes; differing keys: "
                f"{sorted(missing)[:5]}"
            )

    out = {}
    for key, ref in reference.items():
        if not torch.is_floating_point(ref):
            # Integer buffers (step counters, num_batches_tracked) are not
            # meaningfully averageable; take the first node's value.
            out[key] = ref.clone()
            continue
        acc = ref.detach().to(torch.float32).clone()
        for other in states[1:]:
            acc += other[key].detach().to(torch.float32)
        acc /= len(states)
        out[key] = acc.to(ref.dtype)

    buf = io.BytesIO()
    torch.save(out, buf)
    return buf.getvalue()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "role": "coordinator", "version": __version__}


@app.post("/v1/cluster/join", dependencies=[Depends(require_token)])
def join(node_id: str, world_size: int) -> dict:
    """Register a node. The first caller fixes the world size."""
    if world_size < 1:
        raise HTTPException(status_code=400, detail="world_size must be >= 1")
    with STATE.lock:
        if STATE.world_size and world_size != STATE.world_size:
            raise HTTPException(
                status_code=409,
                detail=f"cluster already formed with world_size={STATE.world_size}",
            )
        STATE.world_size = world_size
        STATE.members[node_id] = time.time()
        rank = sorted(STATE.members).index(node_id)
    return {"rank": rank, "world_size": STATE.world_size, "members": sorted(STATE.members)}


@app.get("/v1/cluster/status", dependencies=[Depends(require_token)])
def status() -> dict:
    with STATE.lock:
        current = max(STATE.rounds) if STATE.rounds else None
        rnd = STATE.rounds.get(current) if current is not None else None
        return {
            "world_size": STATE.world_size,
            "members": sorted(STATE.members),
            "round": current,
            "received": rnd.received if rnd else 0,
            "waiting_for": (STATE.world_size - rnd.received) if rnd else 0,
        }


@app.post("/v1/cluster/round/{number}/submit", dependencies=[Depends(require_token)])
async def submit(number: int, node_id: str, request: Request) -> dict:
    body = await request.body()
    if len(body) > MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")

    with STATE.lock:
        if STATE.world_size == 0:
            raise HTTPException(status_code=409, detail="no cluster formed; join first")
        rnd = STATE.rounds.get(number)
        if rnd is None:
            rnd = Round(number=number, expected=STATE.world_size)
            STATE.rounds[number] = rnd
        rnd.payloads[node_id] = body
        STATE.members[node_id] = time.time()
        ready = rnd.received >= rnd.expected

    if ready and rnd.averaged is None:
        # Averaging happens outside the lock: it is the slow part, and holding
        # the lock through it would stall every other node's submit.
        try:
            rnd.averaged = _average(list(rnd.payloads.values()))
        except (ValueError, RuntimeError) as exc:
            rnd.complete.set()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Free the per-node copies; only the mean is still needed.
        rnd.payloads = {k: b"" for k in rnd.payloads}
        rnd.complete.set()

    return {"round": number, "received": rnd.received, "expected": rnd.expected}


@app.get("/v1/cluster/round/{number}/result", dependencies=[Depends(require_token)])
def result(number: int, timeout_s: int = DEFAULT_ROUND_TIMEOUT_S) -> Response:
    """Block until every node has submitted this round, then return the mean."""
    rnd = STATE.rounds.get(number)
    if rnd is None:
        raise HTTPException(status_code=404, detail=f"round {number} has no submissions yet")

    if not rnd.complete.wait(timeout=timeout_s):
        raise HTTPException(
            status_code=504,
            detail=(
                f"round {number} timed out with {rnd.received}/{rnd.expected} nodes. "
                "Someone's GPU is slow, offline, or their share expired."
            ),
        )
    if rnd.averaged is None:
        raise HTTPException(status_code=500, detail="averaging failed for this round")

    # Older rounds are dead weight once a newer one completes.
    with STATE.lock:
        for old in [n for n in STATE.rounds if n < number - 1]:
            STATE.rounds.pop(old, None)

    return Response(content=rnd.averaged, media_type="application/octet-stream")


@app.post("/v1/cluster/reset", dependencies=[Depends(require_token)])
def reset() -> dict:
    with STATE.lock:
        STATE.world_size = 0
        STATE.members.clear()
        STATE.rounds.clear()
    return {"ok": True}


def serve(host: str = "0.0.0.0", port: int = 8899) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
