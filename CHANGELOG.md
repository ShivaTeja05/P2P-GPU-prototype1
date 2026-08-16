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

## Unreleased

Working toward v2: single installer, GUI, and multi-GPU clustering.
