"""The half of AVERAGE mode that runs next to a GPU.

`coordinator.py` is the meeting point; this is what actually trains. Each node
holds a complete copy of the model, walks its own shard of the data for a few
hundred steps, then meets the others to average weights. Between meetings the
nodes never talk, which is the entire point: at 110 ms round-trip, anything that
synchronises per step is dead on arrival.

The literature calls this local SGD / DiLoCo. What matters here is the ratio it
buys you. Averaging every step means one 110 ms stall per step; averaging every
500 steps amortises that stall over 500 steps of real compute, so the link stops
being the bottleneck and the GPUs start being it.

Two things are deliberately NOT synchronised:

  Optimizer state stays local. Momentum buffers are a property of the path a
  node took through its own data; averaging them mixes trajectories that were
  never comparable. Standard for local-SGD, and it halves what crosses the wire.

  Data is sharded by rank, not shuffled globally. Two nodes training on the
  same batches would compute nearly the same gradient, and averaging two copies
  of the same answer buys nothing. `shard()` is what makes the second GPU add
  information rather than confirm the first.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from p2pgpu.common.config import AUTH_HEADER

# Weight sync is a full state_dict in each direction, every round. A 100M
# parameter model in fp32 is 400 MB up and 400 MB down -- on a 50 Mbps upload
# that is a minute of stall. Past this we warn rather than refuse, because the
# right fix (sync less often) is the caller's decision, not ours.
WARN_PAYLOAD_MB = 256.0


class TrainerError(RuntimeError):
    """The cluster could not be formed or a round could not complete."""


@dataclass
class RoundResult:
    """What one meeting cost and what it was worth."""

    number: int
    local_steps: int
    mean_loss: float
    train_s: float
    sync_s: float
    payload_mb: float

    @property
    def sync_overhead(self) -> float:
        """Fraction of wall-clock spent talking instead of training."""
        total = self.train_s + self.sync_s
        return self.sync_s / total if total else 0.0

    def __str__(self) -> str:
        # Small models round to "0 MB" and make the sync look free, which is the
        # one number a reader should not be misled about.
        size = f"{self.payload_mb:.0f} MB" if self.payload_mb >= 10 else f"{self.payload_mb:.2f} MB"
        return (
            f"round {self.number:>3}  loss {self.mean_loss:.4f}  "
            f"train {self.train_s:6.1f}s  sync {self.sync_s:5.1f}s "
            f"({size}, {self.sync_overhead * 100:.0f}% overhead)"
        )


def shard(items: Sequence[Any], rank: int, world_size: int) -> list[Any]:
    """This node's slice of the dataset.

    Strided rather than contiguous (`items[rank::world_size]`) so every node
    sees the same distribution. A contiguous split of data that arrived in any
    kind of order -- sorted by class, by length, by collection date -- gives one
    node a biased sample, and averaging a biased model with an unbiased one is
    worse than not clustering at all.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    return list(items)[rank::world_size]


def _serialise_state(model: Any) -> bytes:
    """Model weights as bytes, on CPU.

    Buffers (BatchNorm running stats, and anything else registered but not a
    parameter) are included: they are part of what the model computes, and a
    node that averaged parameters but kept its own buffers would drift from its
    peers in a way that never shows up in the loss until evaluation.
    """
    import torch

    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def _load_state(model: Any, raw: bytes) -> None:
    import torch

    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    # load_state_dict copies into the existing tensors, so the model stays on
    # whatever device it was on -- no .to(device) round trip per round.
    model.load_state_dict(state)


class ClusterTrainer:
    """One node's view of a data-parallel training run.

    Owns the rendezvous and the weight exchange; owns nothing about the model.
    The caller supplies a `step_fn` that does one optimizer step and returns its
    loss, which keeps this usable for any torch model rather than only the ones
    we thought of.
    """

    def __init__(
        self,
        coordinator_url: str,
        token: str,
        node_id: str,
        world_size: int,
        timeout_s: float = 900.0,
    ) -> None:
        self.url = coordinator_url.rstrip("/")
        self.token = token
        self.node_id = node_id
        self.world_size = world_size
        self.timeout_s = timeout_s
        self.rank: int | None = None
        self._headers = {AUTH_HEADER: token}

    # -- rendezvous ---------------------------------------------------------

    def join(self, wait_s: float = 300.0, poll_s: float = 2.0) -> int:
        """Register, then block until every node has registered too.

        The wait is not politeness. The coordinator assigns ranks by sorting
        member ids, so the rank handed to the first node to call join is
        provisional -- it changes when a node sorting before it arrives. Since
        `shard()` keys the dataset off rank, acting on a provisional rank would
        silently give two nodes the same data. So we re-read the rank once the
        membership is final, and only then is it safe to use.
        """
        try:
            resp = httpx.post(
                f"{self.url}/v1/cluster/join",
                params={"node_id": self.node_id, "world_size": self.world_size},
                headers=self._headers,
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise TrainerError(
                f"could not reach the coordinator at {self.url}: {exc}\n"
                "Is 'p2pgpu cluster coordinator' running there, and is the "
                "machine on the tailnet?"
            ) from exc

        deadline = time.monotonic() + wait_s
        while True:
            members = self.status().get("members", [])
            if len(members) >= self.world_size:
                if self.node_id not in members:
                    raise TrainerError(
                        f"cluster formed without this node ({self.node_id}); "
                        f"members are {members}"
                    )
                self.rank = sorted(members).index(self.node_id)
                return self.rank
            if time.monotonic() > deadline:
                raise TrainerError(
                    f"waited {wait_s:.0f}s for {self.world_size} nodes; only "
                    f"{len(members)} joined ({members}). Start the trainer on "
                    "the other machine, or lower --world-size."
                )
            time.sleep(poll_s)

    def status(self) -> dict:
        resp = httpx.get(
            f"{self.url}/v1/cluster/status", headers=self._headers, timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()

    # -- the exchange -------------------------------------------------------

    def sync(self, model: Any, round_number: int) -> tuple[float, float]:
        """Submit this node's weights, block for the mean, adopt it.

        Returns (seconds, payload_mb). Every node comes out of this call with
        byte-identical weights, which is what makes the next round's local
        training comparable across nodes.
        """
        started = time.perf_counter()
        payload = _serialise_state(model)
        payload_mb = len(payload) / 1024**2

        try:
            submit = httpx.post(
                f"{self.url}/v1/cluster/round/{round_number}/submit",
                params={"node_id": self.node_id},
                content=payload,
                headers=self._headers,
                timeout=self.timeout_s,
            )
            submit.raise_for_status()

            result = httpx.get(
                f"{self.url}/v1/cluster/round/{round_number}/result",
                params={"timeout_s": int(self.timeout_s)},
                headers=self._headers,
                timeout=self.timeout_s + 30,
            )
            result.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise TrainerError(
                f"round {round_number} failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TrainerError(f"round {round_number} lost the coordinator: {exc}") from exc

        _load_state(model, result.content)
        return time.perf_counter() - started, payload_mb

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        model: Any,
        step_fn: Callable[[int], float],
        rounds: int,
        sync_every: int,
        on_round: Callable[[RoundResult], None] | None = None,
    ) -> list[RoundResult]:
        """Train locally for `sync_every` steps, meet, repeat `rounds` times.

        `step_fn(global_step)` runs exactly one optimizer step and returns its
        loss as a float. Everything about the model, data and optimizer lives in
        there; this loop only decides when to stop and talk.
        """
        if self.rank is None:
            raise TrainerError("call join() before run()")
        if rounds < 1 or sync_every < 1:
            raise TrainerError("rounds and sync_every must both be >= 1")

        history: list[RoundResult] = []
        step = 0
        for number in range(rounds):
            train_started = time.perf_counter()
            losses = []
            for _ in range(sync_every):
                losses.append(float(step_fn(step)))
                step += 1
            train_s = time.perf_counter() - train_started

            sync_s, payload_mb = self.sync(model, number)

            result = RoundResult(
                number=number,
                local_steps=sync_every,
                mean_loss=sum(losses) / len(losses),
                train_s=train_s,
                sync_s=sync_s,
                payload_mb=payload_mb,
            )
            history.append(result)
            if on_round is not None:
                on_round(result)
        return history


def estimate_sync_cost(
    param_count: int, upload_mbps: float | None, download_mbps: float | None
) -> dict:
    """What one round will cost on this link, before you commit to a run.

    Weights cross in fp32 regardless of training dtype, because averaging in
    fp16 loses precision the update depends on -- so this is sized on 4 bytes
    per parameter, not on whatever the model is stored as.
    """
    payload_mb = param_count * 4 / 1024**2
    up_s = (payload_mb * 8 / upload_mbps) if upload_mbps else None
    down_s = (payload_mb * 8 / download_mbps) if download_mbps else None
    return {
        "payload_mb": round(payload_mb, 1),
        "upload_s": round(up_s, 1) if up_s else None,
        "download_s": round(down_s, 1) if down_s else None,
        "round_trip_s": round((up_s or 0) + (down_s or 0), 1) if up_s and down_s else None,
        "heavy": payload_mb > WARN_PAYLOAD_MB,
    }


def sharded_batches(
    items: Sequence[Any], batch_size: int, rank: int, world_size: int
) -> Iterator[list[Any]]:
    """Endless batches from this node's shard, for use inside a `step_fn`."""
    mine = shard(items, rank, world_size)
    if not mine:
        raise ValueError(
            f"rank {rank} got no data: {len(items)} items across {world_size} nodes"
        )
    while True:
        for start in range(0, len(mine), batch_size):
            batch = mine[start : start + batch_size]
            if batch:
                yield batch


__all__ = [
    "ClusterTrainer",
    "RoundResult",
    "TrainerError",
    "estimate_sync_cost",
    "shard",
    "sharded_batches",
]
