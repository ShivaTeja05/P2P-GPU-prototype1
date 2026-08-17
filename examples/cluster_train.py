"""Train one model across two GPUs in two different houses.

Paste this into a Jupyter cell on EACH friend's GPU, change NODE_ID on the
second one, and run both. They will train the same model on different halves of
the data and meet periodically to average weights.

Deliberately self-contained: `p2pgpu` is not installed inside the session
container, so this talks to the coordinator with nothing but `urllib` from the
standard library. Copy the file, or paste it -- both work.

  Before running, on your Mac:   p2pgpu cluster coordinator
  It prints the URL to put in COORDINATOR below.

To train YOUR model instead of MNIST, replace build_model() and load_data().
Nothing else needs to change.
"""

import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import torch
from torch import nn

# ---------------------------------------------------------------------------
# CONFIG -- the only part you edit
# ---------------------------------------------------------------------------

COORDINATOR = "http://100.65.244.36:8899"   # from 'p2pgpu cluster coordinator'
TOKEN = "PASTE-YOUR-CLUSTER-TOKEN"          # same token on every machine
NODE_ID = "friend-a"                        # MUST differ on each machine
WORLD_SIZE = 2                              # how many GPUs are joining

ROUNDS = 10                                 # how many times we meet and average
SYNC_EVERY = 100                            # local steps between meetings
BATCH_SIZE = 128
LEARNING_RATE = 0.01

AUTH_HEADER = "x-p2pgpu-token"

# ---------------------------------------------------------------------------
# Minimal coordinator client -- stdlib only, so it runs in any torch container
# ---------------------------------------------------------------------------


def _request(method, path, *, params=None, body=None, timeout=900):
    url = COORDINATOR.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header(AUTH_HEADER, TOKEN)
    if body is not None:
        req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the coordinator at {COORDINATOR}: {exc.reason}\n"
            "Is 'p2pgpu cluster coordinator' running on the Mac, and is this "
            "container's host on the tailnet?"
        ) from exc


def join_cluster():
    """Register, then wait for everyone. Returns this node's rank.

    The wait matters: ranks are assigned by sorting node ids, so the rank you
    get before the others arrive can still change. Data is sharded by rank, so
    acting on a provisional one would quietly give two nodes the same data.
    """
    _request("POST", "/v1/cluster/join",
             params={"node_id": NODE_ID, "world_size": WORLD_SIZE}, timeout=60)

    print(f"joined as {NODE_ID}, waiting for {WORLD_SIZE} nodes...")
    while True:
        state = json.loads(_request("GET", "/v1/cluster/status", timeout=60))
        members = state.get("members", [])
        if len(members) >= WORLD_SIZE:
            rank = sorted(members).index(NODE_ID)
            print(f"cluster formed: {members} -- this machine is rank {rank}")
            return rank
        print(f"  {len(members)}/{WORLD_SIZE} here: {members}")
        time.sleep(3)


def sync_weights(model, round_number):
    """Send our weights, wait for everyone's, adopt the average."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    payload = buf.getvalue()

    _request("POST", f"/v1/cluster/round/{round_number}/submit",
             params={"node_id": NODE_ID}, body=payload)
    averaged = _request("GET", f"/v1/cluster/round/{round_number}/result")

    model.load_state_dict(
        torch.load(io.BytesIO(averaged), map_location="cpu", weights_only=True)
    )
    return len(payload) / 1024**2


def weight_checksum(model):
    """Fingerprint of every weight. Identical on both machines = it worked.

    Rounded before hashing: averaging is float arithmetic, and two machines can
    land a few ULPs apart on the same mean without anything being wrong.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(torch.round(state[key].detach().cpu().float(), decimals=4).numpy().tobytes())
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# YOUR MODEL AND DATA -- replace these two functions
# ---------------------------------------------------------------------------


def build_model():
    """A small CNN. Swap for your own architecture.

    Every node must build an *identical* architecture -- averaging compares
    state_dict keys and shapes, and a mismatch is rejected rather than silently
    producing nonsense.
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def load_data(rank, world_size):
    """This node's shard of the training set.

    Strided (`[rank::world_size]`), not contiguous. MNIST as downloaded is
    grouped, and a contiguous split would hand one node mostly low digits --
    averaging a model that never saw an 8 with one that never saw a 1 is worse
    than not clustering at all.
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full = datasets.MNIST(root="/workspace/data", train=True, download=True, transform=tf)

    indices = list(range(len(full)))[rank::world_size]
    mine = torch.utils.data.Subset(full, indices)
    print(f"this node trains on {len(mine)} of {len(full)} images")
    return torch.utils.data.DataLoader(
        mine, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA here. This will work, but slowly.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rank = join_cluster()

    torch.manual_seed(1000 + rank)  # start genuinely apart, so the sync is visible
    model = build_model().to(device)
    loader = load_data(rank, WORLD_SIZE)

    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    print(f"weights before any sync: {weight_checksum(model)}")
    print(f"\n{'round':>5} {'loss':>8} {'acc':>7} {'train':>8} {'sync':>7}  weights")

    batches = iter(loader)
    for round_number in range(ROUNDS):
        model.train()
        started = time.perf_counter()
        losses, correct, seen = [], 0, 0

        for _ in range(SYNC_EVERY):
            try:
                images, labels = next(batches)
            except StopIteration:            # one epoch done; go round again
                batches = iter(loader)
                images, labels = next(batches)

            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimiser.step()

            losses.append(loss.item())
            correct += (output.argmax(1) == labels).sum().item()
            seen += len(labels)

        train_s = time.perf_counter() - started

        sync_started = time.perf_counter()
        payload_mb = sync_weights(model, round_number)
        sync_s = time.perf_counter() - sync_started

        print(
            f"{round_number:>5} {sum(losses) / len(losses):>8.4f} "
            f"{correct / seen * 100:>6.1f}% {train_s:>7.1f}s {sync_s:>6.1f}s  "
            f"{weight_checksum(model)}"
        )

    print(f"\ndone. {payload_mb:.0f} MB crossed the wire each way, each round.")
    print(f"final weight checksum: {weight_checksum(model)}")
    print(
        "\nCompare that checksum with the other machine's.\n"
        "  IDENTICAL -> the two GPUs really trained one model.\n"
        "  DIFFERENT -> each was training alone; the averaging never landed."
    )
    torch.save(model.state_dict(), f"/workspace/cluster_model_rank{rank}.pt")
    print(f"weights saved to /workspace/cluster_model_rank{rank}.pt")


if __name__ == "__main__":
    main()
