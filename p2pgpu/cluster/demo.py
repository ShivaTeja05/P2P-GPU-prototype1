"""A training run small enough to debug, real enough to prove the cluster works.

The question this answers is not "can torch do gradient descent" -- it is "did
two machines in two houses actually train the same model". Those look identical
from one node's console: loss falls either way, whether or not the peer's
contribution ever arrived.

So the demo prints a **weight checksum** after every round. Nodes start from
different random weights and train on disjoint data, so their checksums can only
match if the averaging round-tripped. Same number on both screens is the proof;
a falling loss on its own is not.

Synthetic data on purpose: no download, no dataset paths to get wrong, and the
target function is known, so a loss that fails to fall is a bug in the cluster
rather than a hard learning problem.
"""

from __future__ import annotations

import hashlib
from typing import Any

from p2pgpu.cluster.trainer import ClusterTrainer, RoundResult, shard

# Big enough that averaging moves real bytes over the wire, small enough that a
# round is seconds rather than minutes on CPU.
HIDDEN = 256
FEATURES = 64
SAMPLES = 4096


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def weight_checksum(model: Any) -> str:
    """A short, order-stable fingerprint of every weight in the model.

    Rounded to 4 decimals before hashing. Averaging is float arithmetic and two
    machines can land a few ULPs apart on the same mean; a raw byte hash would
    call that a mismatch and send someone debugging a network that is fine.
    """
    import torch

    digest = hashlib.sha256()
    for key in sorted(model.state_dict()):
        tensor = model.state_dict()[key].detach().to("cpu", torch.float32)
        digest.update(key.encode())
        digest.update(torch.round(tensor, decimals=4).numpy().tobytes())
    return digest.hexdigest()[:12]


def build_model(device: str, seed: int):
    """A small MLP. Seeded per node so the nodes genuinely start apart."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(FEATURES, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )
    return model.to(device)


def build_dataset(rank: int, world_size: int, device: str):
    """One fixed regression problem, split so each node sees different rows.

    The *same* seed everywhere, so every node draws the identical pool, and then
    `shard` hands out disjoint slices of it. That ordering matters: a shared
    problem with split data is a cluster, whereas different problems per node is
    just several models being averaged into mush.
    """
    import torch

    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(SAMPLES, FEATURES, generator=generator)
    true_w = torch.randn(FEATURES, 1, generator=generator)
    y = x @ true_w + 0.1 * torch.randn(SAMPLES, 1, generator=generator)

    rows = shard(list(range(SAMPLES)), rank, world_size)
    index = torch.tensor(rows)
    return x[index].to(device), y[index].to(device)


def run_demo(
    coordinator_url: str,
    token: str,
    node_id: str,
    world_size: int,
    rounds: int,
    sync_every: int,
    device: str = "auto",
    console: Any = None,
) -> list[RoundResult]:
    import torch

    def say(message: str) -> None:
        if console is not None:
            console.print(message)
        else:
            print(message)

    resolved = pick_device(device)
    trainer = ClusterTrainer(
        coordinator_url=coordinator_url,
        token=token,
        node_id=node_id,
        world_size=world_size,
    )

    say(f"node [cyan]{node_id}[/] on [cyan]{resolved}[/], joining {coordinator_url} ...")
    rank = trainer.join()
    say(f"[green]cluster formed[/] -- this machine is rank {rank} of {world_size}")

    # Seeded by rank, so the nodes provably do not start from the same weights.
    model = build_model(resolved, seed=1000 + rank)
    x, y = build_dataset(rank, world_size, resolved)
    say(f"training on {len(x)} of {SAMPLES} rows (this node's shard)")
    say(f"[dim]weights before any sync: {weight_checksum(model)}[/]")

    optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    batch = 128

    def step_fn(global_step: int) -> float:
        start = (global_step * batch) % len(x)
        xb = x[start : start + batch]
        yb = y[start : start + batch]
        if len(xb) == 0:
            xb, yb = x[:batch], y[:batch]
        optimiser.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimiser.step()
        return loss.item()

    def on_round(result: RoundResult) -> None:
        say(f"{result}  weights [bold]{weight_checksum(model)}[/]")

    history = trainer.run(
        model=model,
        step_fn=step_fn,
        rounds=rounds,
        sync_every=sync_every,
        on_round=on_round,
    )

    final = weight_checksum(model)
    say("")
    say(f"[green]done.[/] loss {history[0].mean_loss:.4f} -> {history[-1].mean_loss:.4f}")
    say(f"final weight checksum: [bold]{final}[/]")
    say(
        "\nCompare that checksum with the other machine's. [bold]Identical means "
        "the two GPUs really trained one model.[/] Different means each was "
        "training alone and the averaging never landed."
    )
    return history
