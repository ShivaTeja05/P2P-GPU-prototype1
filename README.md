# p2pgpu

Share one GPU between two computers over the internet.

Your friend has a good GPU. You have a laptop that doesn't. `p2pgpu share` hands
their GPU to you for a few hours — you get a notebook running on their hardware,
they keep their files, and the share stops itself when the time is up.

This is the same mechanism GPU rental marketplaces use for their hosts (a Docker
container with the GPU passed through), minus the marketplace, the billing, and
the strangers.

```
your Mac  ──── Tailscale (encrypted, through both routers) ────  friend's PC
                                                                      │
                                                            Docker container
                                                            ├── RTX 4080 ✓
                                                            ├── /workspace ✓
                                                            └── their files ✗
```

## Install

Both machines:

```bash
git clone https://github.com/ShivaTeja05/P2P-GPU-prototype1.git && cd P2P-GPU-prototype1
```

```bash
uv venv --python 3.12 && uv pip install -e .
```

The GPU machine also needs **Docker** and the **NVIDIA Container Toolkit**. Full
walkthrough for Linux and Windows in [docs/SETUP.md](docs/SETUP.md).

## Use it

**1. Both machines join the same Tailscale network.** Install
[Tailscale](https://tailscale.com/download), sign in with the same account on
both. Each machine gets a `100.x.y.z` address that works through both home
routers, with no port forwarding and nothing exposed to the internet.

**2. On the GPU machine, check everything is ready:**

```bash
p2pgpu doctor
```

This tells you exactly what's missing rather than failing later with a Docker
error. When it's green:

**3. Share the GPU:**

```bash
p2pgpu share --hours 4
```

It prints a URL. Send that to your friend.

**4. On the laptop, connect:**

```bash
p2pgpu attach "http://100.x.y.z:8888/lab?token=..."
```

A JupyterLab opens in your browser. `torch.cuda.is_available()` is `True`, and
it's their GPU. Train something.

**5. When you're done** — it stops on its own, or immediately with:

```bash
p2pgpu stop
```

## Commands

**GPU owner**

| Command | Does |
|---|---|
| `p2pgpu doctor` | Check Docker, NVIDIA runtime and Tailscale; say what's missing |
| `p2pgpu share --hours 4` | Share the GPU, print the URL, auto-stop when time's up |
| `p2pgpu status` | Is a share running, and how long is left |
| `p2pgpu stop` | Stop sharing now |
| `p2pgpu logs` | Container output, for when it won't start |

**Borrower**

| Command | Does |
|---|---|
| `p2pgpu attach <url>` | Connect to the shared GPU |
| `p2pgpu probe` | What compute does this machine have |
| `p2pgpu bench <url>` | Measure the link; estimate real transfer times |
| `p2pgpu inspect <url>` | See the other machine's GPU and free VRAM |

`bench` and `inspect` need `p2pgpu serve` running on the other machine. They're
optional — useful for knowing whether pushing a 1 GB dataset takes 20 seconds or
20 minutes before you try it.

## What the container does and doesn't protect

The GPU is shared. The machine is not.

**Protected:** home directory, SSH keys, browser data, other files. The container
only mounts `~/p2pgpu-workspace`, and the notebook port binds to the Tailscale
address specifically — not `0.0.0.0` — so it isn't reachable from the owner's
home Wi-Fi or the public internet.

**Not protected:** whoever holds the URL has code execution inside the container,
with the GPU. Container escapes exist. **Share with people you'd lend your laptop
to.** Details and the full threat model in [docs/SECURITY.md](docs/SECURITY.md).

## Notes

- The GPU is exclusive while shared — the owner's games will stutter. Agree on hours.
- Files in `/workspace` persist on the owner's machine after the session ends.
- Upload speed, not GPU speed, decides how long moving datasets and checkpoints takes.
- `--shm-size` is set to 8 GB because PyTorch dataloaders fail on Docker's 64 MB default.

## Status

Working and tested end to end for the two-machine case. See
[docs/PROGRESS.md](docs/PROGRESS.md) for the running log.

MIT licensed.
