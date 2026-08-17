# Changelog

Named versions you can return to. Every entry is a git tag — check one out with
`git checkout v1.0.0`.

---

## v1.0.0 — 2026-08-16

**First working version.** Verified end to end on real hardware: a MacBook in one
state ran PyTorch on an RTX 4050 in another.

```
GPU          : NVIDIA GeForce RTX 4050 Laptop GPU
torch        : 2.5.1+cu124 | cuda 12.4
matmul == cpu: True
MNIST CNN    : 98.87% accuracy, 3.6 s/epoch
link         : Tailscale direct, 84-110 ms, no relay
```

### What it does

Share one NVIDIA GPU with one trusted person over an encrypted overlay network.
The GPU goes into a Docker container; the machine does not.

### Features

- **`p2pgpu share`** — hand over the GPU for a fixed time, with hard expiry
- **`p2pgpu connect` / `discover`** — find shared GPUs on your tailnet, no URLs
- **`p2pgpu prepare`** — download the image at install time, not at use time
- **`p2pgpu doctor`** — preflight that actually runs `nvidia-smi` in a container
- **SSH access** — key-only, into the container, for VS Code Remote-SSH
- **Any NVIDIA GPU** — Pascal through Blackwell, image chosen automatically
- **Windows without a terminal** — `.bat` launchers and a guided PowerShell setup
- **Bench + probe** — measure the link and estimate real transfer times

### Security posture

- Nothing exposed to the internet; reachable only over the Tailscale tunnel
- Notebook port binds to the overlay IP, not `0.0.0.0`
- Only `~/p2pgpu-workspace` is mounted — `$HOME` stays out
- Random per-session token; SSH is key-only, passwords disabled
- Hard expiry via `timeout` inside the container, so nothing outlives attention

### Known limits

- One GPU, two machines. No clustering yet.
- Container isolation is good, not perfect — trusted friends only
- The GPU owner must be Windows or Linux (Docker GPU passthrough is NVIDIA-only)
- First `prepare` downloads 7–9 GB

### Bugs found by running it for real

Things a test suite never would have caught:

- 900 s timeout covering a 7–9 GB image pull
- `uv venv` prompting interactively on re-run
- Tailscale absent from `PATH` on Windows
- NVIDIA-runtime check false-negative under Docker Desktop + WSL2
- Backslashes in the Docker `-v` argument on Windows
- Windows mark-of-the-web blocking every `.bat` file
- Bare IPv6 literals mangled in the `-p` flag and share URL

---

## Unreleased — branch `v2.0-cluster`

Two GPUs, one training run. Proven locally; not yet run on two real GPUs.

### Added
- `p2pgpu cluster coordinator` — the meeting point nodes sync against. Never
  loads a model, so it runs on the GPU-less Mac
- `p2pgpu cluster demo` — proves a cluster combined, by printing a weight
  checksum that can only match across machines if the averaging round-tripped.
  `--as <name>` rehearses a whole cluster on one machine
- `p2pgpu cluster relay` — forwards the coordinator into the session container,
  for Windows hosts where the container's host is the WSL2 VM and cannot see
  the Windows Tailscale interface
- `p2pgpu cluster plan` — how a model would be cut across mismatched GPUs
- `p2pgpu cluster estimate` — what a sync round costs on a measured link
- `p2pgpu cluster status` — who has joined, which round they are on
- `cluster/trainer.py` — `ClusterTrainer`, the local-SGD loop; `shard()` for
  splitting data by rank; `estimate_sync_cost()`
- `examples/cluster_train.py` — MNIST across two GPUs, stdlib-only so it runs
  inside the session container where `p2pgpu` is not installed
- `docs/CLUSTER.md` — the walkthrough, including what does *not* pool

### Notes
- **Only AVERAGE mode can train.** PIPELINE (splitting one model across cards)
  runs a forward pass and is numerically exact, but training through it needs
  gradients sent back up the stages, which is not built
- VRAM does not pool transparently, and RAM/disk/CPU do not pool at all. What
  combines is compute. `docs/CLUSTER.md` opens with the table

---

## Unreleased — branch `v1.1-authkey`

### Added
- `p2pgpu invite` / `join` — one code carries the Tailscale auth key and the
  cluster token, replacing eight setup steps including the admin-console share
- `p2pgpu discover` / `connect` — find shared GPUs on the tailnet, no URLs
- `p2pgpu prepare` — download the session image at install time, not at use time
- `p2pgpu service install` — background agent via launchd / systemd / Task Scheduler
- `docker/` — purpose-built session image, jupyterlab and sshd baked in
- Docker Desktop is started automatically when it is installed but not running

### Fixed
- launchd crash loop: `KeepAlive=true` respawned an agent that could not bind
  its port, forever. Now restarts only on failure, and the agent exits 0 when
  another healthy agent is already serving
- Default images were `-devel` (7–9 GB) instead of `-runtime` (~4 GB)
- `discovery._tailscale_exe` had two identical branches behind a condition that
  was always true
- `agent` and `serve` were duplicate commands
- Unused import in `prepare`

Working toward v2: single installer, GUI, and multi-GPU clustering.
