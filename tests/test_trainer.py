"""Two nodes, one model: does weight averaging actually combine them?

These run a real coordinator over a real socket with two real trainer threads,
rather than mocking the HTTP. The failure this is guarding against is exactly
the kind that mocks cannot see -- a cluster where each node trains happily on
its own and the peer's weights never arrive. Loss falls either way; only
comparing the two nodes' weights afterwards tells them apart.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from p2pgpu.cluster.trainer import (
    ClusterTrainer,
    TrainerError,
    estimate_sync_cost,
    shard,
    sharded_batches,
)

TOKEN = "test-token-for-trainer"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def coordinator(monkeypatch):
    """A real coordinator on a real port, reset between tests."""
    import uvicorn

    monkeypatch.setenv("P2PGPU_TOKEN", TOKEN)
    os.environ["P2PGPU_TOKEN"] = TOKEN

    from p2pgpu.cluster import coordinator as coord

    coord.STATE.world_size = 0
    coord.STATE.members.clear()
    coord.STATE.rounds.clear()

    port = _free_port()
    config = uvicorn.Config(coord.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("coordinator did not start")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


# --- sharding --------------------------------------------------------------


def test_shard_is_disjoint_and_covers_everything():
    items = list(range(100))
    a = shard(items, 0, 2)
    b = shard(items, 1, 2)
    assert set(a) & set(b) == set()
    assert sorted(a + b) == items


def test_shard_is_strided_not_contiguous():
    """A contiguous split of ordered data hands one node a biased sample."""
    sorted_by_class = [0] * 50 + [1] * 50
    a = shard(sorted_by_class, 0, 2)
    b = shard(sorted_by_class, 1, 2)
    # Both nodes must see both classes; a contiguous split would give one node
    # only zeros and the other only ones.
    assert set(a) == {0, 1}
    assert set(b) == {0, 1}


def test_shard_rejects_a_rank_outside_the_world():
    with pytest.raises(ValueError, match="out of range"):
        shard([1, 2, 3], 2, 2)


def test_sharded_batches_refuses_to_starve_a_node():
    with pytest.raises(ValueError, match="no data"):
        next(sharded_batches([1], 4, rank=1, world_size=2))


def test_sharded_batches_cycles_forever():
    stream = sharded_batches(list(range(10)), 2, rank=0, world_size=2)
    first = [next(stream) for _ in range(3)]
    assert first[0] == [0, 2]
    # 5 items at batch 2 => 3 batches, then it wraps rather than stopping.
    assert next(stream) == [0, 2]
    assert all(len(b) > 0 for b in first)


# --- the cluster itself ----------------------------------------------------


def _train_one_node(url, node_id, world_size, seed, results, errors):
    try:
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 2)
        trainer = ClusterTrainer(url, TOKEN, node_id, world_size, timeout_s=60)
        rank = trainer.join(wait_s=30)

        x = torch.randn(32, 4) + rank  # different data per node
        y = torch.randn(32, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        def step(_i):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            return loss.item()

        history = trainer.run(model, step, rounds=3, sync_every=4)
        results[node_id] = {
            "rank": rank,
            "history": history,
            "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
        }
    except Exception as exc:  # noqa: BLE001 - re-raised in the main thread
        errors[node_id] = exc


def _run_two_nodes(url):
    results, errors = {}, {}
    threads = [
        threading.Thread(
            target=_train_one_node, args=(url, name, 2, seed, results, errors)
        )
        for name, seed in (("node-a", 1), ("node-b", 2))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    if errors:
        raise errors[next(iter(errors))]
    assert len(results) == 2, "a node never finished"
    return results


def test_two_nodes_end_with_identical_weights(coordinator):
    """The whole point. Different seeds, different data, same weights at the end."""
    results = _run_two_nodes(coordinator)
    a = results["node-a"]["weights"]
    b = results["node-b"]["weights"]

    assert set(a) == set(b)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), (
            f"{key} differs between nodes -- the averaging did not round-trip"
        )


def test_nodes_really_did_start_apart(coordinator):
    """Guards the test above from passing for the wrong reason.

    If both nodes happened to initialise identically, matching final weights
    would prove nothing at all.
    """
    torch.manual_seed(1)
    first = torch.nn.Linear(4, 2).state_dict()["weight"]
    torch.manual_seed(2)
    second = torch.nn.Linear(4, 2).state_dict()["weight"]
    assert not torch.allclose(first, second)


def test_ranks_are_unique_and_cover_the_world(coordinator):
    results = _run_two_nodes(coordinator)
    ranks = sorted(r["rank"] for r in results.values())
    assert ranks == [0, 1]


def test_every_round_is_accounted_for(coordinator):
    results = _run_two_nodes(coordinator)
    for record in results.values():
        history = record["history"]
        assert [r.number for r in history] == [0, 1, 2]
        assert all(r.local_steps == 4 for r in history)
        assert all(r.payload_mb > 0 for r in history)
        assert all(r.sync_s > 0 for r in history)


def test_averaging_is_the_mean_not_a_last_writer_wins(coordinator):
    """Two known state_dicts in, their arithmetic mean out."""
    import io

    import httpx

    from p2pgpu.common.config import AUTH_HEADER

    headers = {AUTH_HEADER: TOKEN}
    for name in ("m1", "m2"):
        httpx.post(
            f"{coordinator}/v1/cluster/join",
            params={"node_id": name, "world_size": 2},
            headers=headers,
            timeout=20,
        ).raise_for_status()

    def payload(value):
        buf = io.BytesIO()
        torch.save({"w": torch.full((3,), float(value))}, buf)
        return buf.getvalue()

    for name, value in (("m1", 2.0), ("m2", 8.0)):
        httpx.post(
            f"{coordinator}/v1/cluster/round/0/submit",
            params={"node_id": name},
            content=payload(value),
            headers=headers,
            timeout=30,
        ).raise_for_status()

    resp = httpx.get(
        f"{coordinator}/v1/cluster/round/0/result", headers=headers, timeout=30
    )
    resp.raise_for_status()
    mean = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=True)
    assert torch.allclose(mean["w"], torch.full((3,), 5.0))


def test_join_says_who_is_missing_rather_than_hanging(coordinator):
    trainer = ClusterTrainer(coordinator, TOKEN, "lonely", world_size=2)
    with pytest.raises(TrainerError, match="only 1 joined"):
        trainer.join(wait_s=2, poll_s=0.2)


def test_unreachable_coordinator_names_the_fix():
    trainer = ClusterTrainer(f"http://127.0.0.1:{_free_port()}", TOKEN, "x", 2)
    with pytest.raises(TrainerError, match="cluster coordinator"):
        trainer.join(wait_s=1)


def test_run_before_join_is_refused(coordinator):
    trainer = ClusterTrainer(coordinator, TOKEN, "x", 2)
    with pytest.raises(TrainerError, match="join"):
        trainer.run(torch.nn.Linear(2, 2), lambda _i: 0.0, rounds=1, sync_every=1)


# --- the cost of a round ---------------------------------------------------


def test_estimate_sizes_weights_in_fp32():
    """Averaging happens in fp32 even when training is fp16, so cost must too."""
    cost = estimate_sync_cost(100_000_000, upload_mbps=50, download_mbps=100)
    assert cost["payload_mb"] == pytest.approx(381.5, abs=1.0)
    assert cost["upload_s"] == pytest.approx(61.0, abs=2.0)
    assert cost["heavy"] is True


def test_estimate_handles_an_unmeasured_link():
    cost = estimate_sync_cost(1_000_000, upload_mbps=None, download_mbps=None)
    assert cost["upload_s"] is None
    assert cost["round_trip_s"] is None
    assert cost["payload_mb"] > 0
