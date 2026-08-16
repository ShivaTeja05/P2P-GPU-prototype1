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
| `p2pgpu share` — GPU passthrough container | done, untested on real NVIDIA hardware |
| `p2pgpu attach` — guest connection | done |
| `p2pgpu doctor` — host preflight | done |
| Real two-machine run | **pending — needs the friend's PC** |

**Machines**

| Role | Machine | Backend | Notes |
|---|---|---|---|
| guest | MacBook Pro M3 Pro, 36 GB | mps | control machine |
| host | friend's PC, RTX 40-series | cuda | not yet set up |

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
