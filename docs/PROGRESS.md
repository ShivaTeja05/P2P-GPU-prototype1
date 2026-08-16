# Progress Log

Running record of what changed and why. Newest first.

Entry format:

```
## YYYY-MM-DD — Title
**Status:** where things stand
**Changed:** what actually changed
**Why:** reasoning, especially anything non-obvious
**Measured:** numbers, if any
**Next:** immediate next action
```

---

## Status at a glance

| Piece | State |
|---|---|
| Node identity + shared token | done |
| GPU / capability probing (CUDA, Metal, CPU) | done |
| Link benchmark + transfer estimates | done |
| `p2pgpu share` — GPU passthrough container | **working on real NVIDIA hardware** |
| `p2pgpu attach` — guest connection | done |
| `p2pgpu doctor` — host preflight | done |
| Real two-machine run | **done — 2026-08-16, Mac ↔ RTX 4050, two states apart** |

**Machines**

| Role | Machine | Backend | Notes |
|---|---|---|---|
| guest | MacBook Pro M3 Pro, 36 GB | mps | control machine |
| host | Lenovo LOQ, RTX 4050 Laptop 6 GB, Win 11 | cuda | working |

---

## 2026-08-16 — GUIDE.md as the front door

**Status:** Docs complete. Still awaiting the real NVIDIA hardware run.

**Changed:** Added `GUIDE.md` — one beginner-facing walkthrough covering both
sides, split into "I'm lending my GPU" and "I'm borrowing a GPU" so nobody reads
the half that doesn't apply to them. README now points at it first.

**Why:** SETUP/WINDOWS/SECURITY are good reference but assume too much for a
first-timer, and the information was spread across three files. Someone landing
on the repo needs one link, one track, and a checkpoint at every step. The guide
uses explicit "you should see X" checks so people can tell whether a step
actually worked rather than discovering it three steps later.

Also folds in the two things people most reliably get wrong: the Tailscale
share-invite step (easy to skip entirely, and nothing works without it), and
installing Linux NVIDIA drivers inside WSL.

**Next:** unchanged — friend runs Setup.bat, then Check-Setup.bat.

---

## 2026-08-16 — IT WORKS: first real cross-state GPU share

**Status:** End to end verified. A MacBook ran PyTorch on an RTX 4050 in
another state.

**Measured, on the real link:**
```
hostname     : 68443f1941d8            (container on his laptop)
torch        : 2.5.1+cu124 | cuda 12.4
cuda avail   : True
GPU          : NVIDIA GeForce RTX 4050 Laptop GPU
capability   : 8.9
VRAM         : 6.0 GB total / 5.0 GB free
matmul == cpu: True
```
- Tailscale: `active; direct`, 84-110 ms RTT, no relay. NAT traversal worked
  first try on both home routers.
- Image auto-selection chose `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
  from cc 8.9 + driver 610.88 — correct, and the CPU-reference matmul proves
  the kernels actually match the card.
- Jupyter answered HTTP 200; kernel started and executed remotely.

**What the live run cost us (bugs only real hardware found):**
- *900 s timeout covering the image pull.* `docker run -d` was doing the pull,
  so one timeout had to cover a 7-9 GB download. Split into `pull_image()`
  with a 2 h budget and docker's own progress bars, since a detached run is
  silent and the owner just sees a frozen prompt.
- *`uv venv` prompts when `.venv` exists.* Re-running Setup.bat is normal
  (reboot after Docker, retry after a fix) and it stalled behind an
  unexpected question. Now reuses the environment.
- *Tailscale not on PATH on Windows*, *backslashes in the `-v` argument*, and
  *the NVIDIA-runtime check false-negative on Windows* — all found earlier the
  same day, all confirmed fixed by this run.

**Still true:** 6 GB VRAM is the real ceiling. Good for LoRA on small models
and 7B at 4-bit; not for anything larger.

**Next:** send the updated build to the host so his next share gets pull
progress, readiness waiting and SSH.

---

## 2026-08-16 — Windows double-click setup

**Status:** Windows users need no terminal. Still awaiting a real hardware run.

**Changed:**
- `scripts/setup.ps1` — detects Python 3.10-3.12, uv, Tailscale, Docker Desktop
  and the NVIDIA driver; offers winget installs with confirmation; then installs
  p2pgpu into `.venv`. Has `-GuestOnly` (skip Docker) and `-Yes`.
- `scripts/share.ps1`, `scripts/connect.ps1` — guided share/connect, with
  clipboard integration in both directions.
- Seven root `.bat` launchers so nothing needs a terminal.
- `p2pgpu url` — prints the bare share URL for scripting.
- `docs/WINDOWS.md` — full walkthrough and troubleshooting.

**Why (Windows-specific fixes that were real bugs):**
- *Tailscale is not on PATH on Windows.* `shutil.which("tailscale")` returned
  None on a perfectly working install. Now falls back to the Program Files
  locations, and the macOS in-bundle path too.
- *Backslashes in the `-v` argument.* `docker -v C:\Users\..:/workspace` mixes
  backslashes with a colon separator. `docker_mount_path()` normalises to
  forward slashes, which Docker accepts everywhere.
- *The NVIDIA-runtime check produced a false negative on Windows.* Docker
  Desktop reaches the GPU through WSL2 paravirtualisation and does not always
  advertise an `nvidia` runtime, so blocking on it would have rejected a working
  setup. `preflight()` now returns (problems, warnings) and this is a warning on
  Windows; the real passthrough test in `doctor` decides.
- *`.bat` wrappers rather than `.ps1`* because Windows blocks double-clicked
  PowerShell by default. Each wrapper sets ExecutionPolicy for that single run,
  so nothing about the machine's configuration changes.
- *`.gitattributes` pins CRLF for `.bat`/`.ps1`* so the scripts still work after
  a clone on Windows.

**Measured:** 43 tests pass, including Windows path rendering via
`PureWindowsPath` (verifiable from macOS). PowerShell scripts checked for
balanced blocks and correct `.bat`→`.ps1` wiring; **not executed on Windows.**

**Next:** friend runs `Setup.bat` → `Check-Setup.bat` on the real machine.

---

## 2026-08-16 — Works with any NVIDIA GPU, not one hardcoded image

**Status:** Unchanged otherwise — still needs a real run on NVIDIA hardware.

**Changed:**
- `worker/gpu_compat.py` — detects each GPU's compute capability, architecture
  and driver version, then picks a container image that can actually run it.
- `share --gpu 0 | 0,1 | all` selects which cards to hand over.
- `doctor` now prints a GPU table (name, arch, cc, VRAM, driver), the image it
  would choose and why, plus warnings. Added `--quick` to skip the image pull.
- 30 tests covering the generation/driver matrix.

**Why:**
- *One hardcoded CUDA image breaks in both directions.* CUDA 12.x needs driver
  >= 525.60.13; below that a 12.x container starts and then dies. And RTX
  50-series is Blackwell/sm_120, which only got PyTorch kernels in 2.7 + CUDA
  12.8 — older images see the GPU then fail with "no kernel image is available
  for execution on the device". Both look like a bug in this tool rather than a
  version mismatch, so they are detected up front.
- *Mixed generations target the weakest card,* since one image must serve all
  of them — except when a Blackwell card is present, which forces 12.8 because
  no older image can run it at all.
- *`format_gpu_flag` rejects malformed input* rather than defaulting to `all`.
  Silently sharing every GPU because of a typo is the wrong failure direction.

**Measured:** 41 tests pass. Verified selection by hand across GTX 1080 →
RTX 5090: Pascal and old-driver cards get CUDA 11.8, Turing/Ampere/Ada get
12.4, Blackwell gets 12.8.

**Next:** unchanged — real run on the friend's machine.

---

## 2026-08-16 — `p2pgpu share` built; project scoped to two machines

**Status:** Feature-complete for the two-machine case. Blocked only on doing a
real run against an actual NVIDIA host.

**Changed:**
- `worker/share.py` — Docker GPU-passthrough sharing: host preflight, container
  lifecycle, session persistence, log access.
- CLI gained `share`, `attach`, `status`, `stop`, `logs`, `doctor`.
- `common/netbench.py` — replaced the abstract throughput projection with
  concrete transfer estimates (how long a 1 GB dataset push actually takes).
- Docs rewritten around the two-machine workflow: `README`, `SETUP`, `SECURITY`.
- Earlier multi-GPU clustering design moved out of the repo; this project is now
  scoped to sharing one GPU between two trusted machines.

**Why (decisions worth recording):**
- *Docker + NVIDIA Container Toolkit, not SSH.* Plain SSH over Tailscale would
  have worked in ten minutes, but it hands over the whole machine. The container
  shares the GPU and nothing else. It's also exactly what GPU rental
  marketplaces do for their hosts, so the model is proven.
- *Bind to the Tailscale IP, not `0.0.0.0`.* `-p 100.x.y.z:8888:8888` means the
  notebook is unreachable from the host's own home Wi-Fi and from the internet.
  A one-character-ish change that removes a whole class of exposure.
- *Expiry via `timeout` inside the container, not a daemon.* The container kills
  itself on schedule; with `--rm` nothing is left behind. No background process
  to leak, no cron entry. The realistic failure mode is a forgotten share
  holding someone's GPU for days, so expiry is mandatory rather than optional.
- *Only `~/p2pgpu-workspace` is mounted.* The guest needs somewhere to leave
  checkpoints; that must not be `$HOME`.
- *`--shm-size=8g`.* Docker defaults to 64 MB of shared memory and PyTorch
  dataloaders crash on it. Costs nothing to set, saves a confusing bug report.
- *`doctor` runs a real `nvidia-smi` in a container.* Checking that the runtime
  is registered is fast but not conclusive; the actual passthrough test is the
  only thing that proves the setup works, and it's the step that usually fails.
- *jupyterlab installed at container start* rather than baked into a custom
  image, so the default stays a stock PyTorch image people may already have
  cached. Costs ~20 s per start. Revisit if that gets annoying.

**Measured:** 11 tests pass. CLI verified on the Mac: `doctor` correctly reports
Docker, NVIDIA runtime and Tailscale all missing with actionable fixes for each;
`attach` to an unreachable URL fails with a useful diagnostic rather than a
stack trace; `status` reports not-sharing cleanly.

**Next:**
1. Friend installs Tailscale, Docker and the NVIDIA Container Toolkit
   (`docs/SETUP.md`).
2. `p2pgpu doctor` on their machine until green.
3. `p2pgpu share --hours 2`, then `p2pgpu attach` from the Mac.
4. Train something small end to end and record what broke here.
