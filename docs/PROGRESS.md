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
| Auto-discovery (no URL passing) | done |
| One-code join (`invite` / `join`) | done, untested on a live tailnet |
| Background agent as an OS service | done, verified on macOS |
| Released | **v1.0.0 tagged** |
| Cluster — layer placement (`plan.py`) | done |
| Cluster — pipeline stage forward pass | done, numerically exact; **inference only** |
| Cluster — coordinator (weight averaging) | done |
| Cluster — trainer client (`trainer.py`) | **done — two nodes train one model** |
| Cluster — real run on two friends' GPUs | not yet |
| Cluster — training *through* a pipeline split | not built (needs backward pass) |

**Machines**

| Role | Machine | Backend | Notes |
|---|---|---|---|
| guest | MacBook Pro M3 Pro, 36 GB | mps | control machine |
| host | Lenovo LOQ, RTX 4050 Laptop 6 GB, Win 11 | cuda | working |

---

## 2026-08-17 — Real RTX 4050 verified; a networking conclusion retracted; the Mac's firewall found

**Status:** The GPU half is confirmed working on real hardware. A networking
conclusion I drew in this same session was **wrong and is retracted below** —
the test had an uncontrolled variable. The investigation did surface a genuine
blocker for the real run, on my own machine rather than the friend's.

**What was measured**, against a live share from the Windows RTX 4050 laptop
(Linux 6.6 WSL2 kernel, container from the stock session image):

```
GPU          NVIDIA GeForce RTX 4050 Laptop, 6.0 GB total / 4.95 GB free, cc 8.9
torch        2.5.1+cu124, cuda available
matmul       matches CPU reference, max diff 4.58e-04
throughput   21.4 TFLOP/s  (fp16, 4096^3)
training     loss 62.539 -> 0.0142 over 200 steps in 0.51 s, 104 MB peak
link         RTT min 58 ms / avg 147 ms — noticeably more variable than v1's 84-110 ms
```

**The networking probes, from inside the friend's container:**

```
pypi.org:443                       OK       124 ms   internet works
100.102.129.101:8888  (own host)   OK         3 ms   its own host works
host.docker.internal:8899          BLOCKED  DNS      the alias does not resolve
100.65.244.36:8899    (the Mac)    BLOCKED  timeout  ← INCONCLUSIVE, see below
```

**Retraction.** I first read that last line as proof that a container on Windows
cannot route to another machine's Tailscale address, and wrote it into
CLUSTER.md, CLUSTER-SETUP.md and the relay's own docstrings as a measured fact.
It is not one. I never verified the *control*: that anything at all could reach
that coordinator. Checking afterwards:

```
coordinator bound 0.0.0.0:8916, probed from the Mac itself
  127.0.0.1:8916        {"ok":true,...}     works
  100.65.244.36:8916    empty reply (52)    blocked
  192.168.0.3:8916      empty reply (52)    blocked   ← LAN too, so not Tailscale

socketfilterfw --getglobalstate
  Firewall is blocking all non-essential incoming connections. (State = 2)
```

The macOS Application Firewall was refusing every inbound connection on every
non-loopback interface. Nothing could have reached that port — not the friend's
container, not anything. The container's timeout says nothing about WSL2. The
WSL2 question is **still open**, exactly as it was before.

The tell was there and I walked past it: the container *timed out* while the Mac
itself got an *empty reply*. Two different failures against the same port should
have prompted the control test before the conclusion.

**What the probes do still support**, since these depend only on the friend's
machine and not on mine:

- *`host.docker.internal` is the wrong address to document.* It is the obvious
  Docker alias, and it does not resolve inside the session container at all —
  a DNS failure entirely local to that container. The GPU machine's own
  Tailscale address answered in 3 ms, so that is what the docs now recommend.
  This one holds regardless of the firewall.
- *A container can reach its own host over the tailnet* (3 ms), which is the
  asymmetry the relay is built on.

**The genuine blocker this turned up:** the coordinator cannot accept
connections on this Mac at all until the firewall allows it. That would have
stopped the real run dead, with the failure appearing to be on the friends'
side. Fix before the session:

```
System Settings → Network → Firewall → Options → allow incoming for python,
or turn "Block all incoming connections" off
```

**Also corrected:** `httpx` *is* present in a running session (0.28.1), contrary
to the earlier note. It is absent from the base image and arrives only because
the container runs `pip install jupyterlab` at start. Depending on a transitive
dependency of an unrelated package is exactly what disappears in a version bump,
so `examples/cluster_train.py` stays on stdlib `urllib` — the conclusion holds,
but the stated fact was wrong and is now accurate.

**Fixed in docs/SETUP.md:** it told Windows users to "run `p2pgpu` from inside
WSL, not PowerShell", which contradicts the `.bat` flow that actually worked.
`setup.ps1` builds a Windows-native venv, and `share.py` has Windows-only
handling — finding `tailscale.exe` off `PATH`, normalising paths for Docker's
`-v` — that never runs under WSL. The install and verify sections were also
Unix-only; both now have a Windows track.

### Retracted a second time — the "control" was still not a control

The section below concluded the WSL2 question was settled. **It was not, and the
error was the same one twice.** Later the same day, the friend's *host* — not a
container, plain Windows with Tailscale — also timed out reaching the
coordinator:

```
This machine cannot reach the coordinator either: timed out
```

A container being blocked is a WSL2 story. The bare host being blocked is not.
Both point at the far end.

What I used as a control was the Mac curling **its own** Tailscale address. That
connection never leaves the machine, so it does not exercise inbound filtering
at all. It is the same shape of mistake as the first retraction: treating a
local success as evidence of remote reachability.

**The actual cause, from `tailscale status --json`:** the two machines belong to
*different accounts*.

```
Self : Gaddam's MacBook Pro  100.65.244.36     shivateja1665@gmail.com
Peer : Loq                   100.102.129.101   motupallibhanu793-design@github
```

That is cross-account **node sharing**, and node sharing is directional. The
friend's `Loq` was shared into this tailnet, so the Mac reaches it — `tailscale
ping` succeeds via DERP(blr), 66–216 ms. The Mac was never shared the other way,
so nothing on `Loq` can open a connection to it. Every observation fits:

```
Mac       -> Loq:8888              OK        the shared direction
Loq host  -> Mac:8899              timeout   never shared this way
Loq ctr   -> Mac:8899              timeout   same reason, not WSL2
Loq ctr   -> Loq host / internet   OK        never involved the Mac
```

**So the WSL2 question is open for the third time, and is now untestable until
the sharing is fixed** — every probe of it so far has been measuring the return
path instead.

**Method note worth keeping:** a reachability claim needs a prober on the far
side of the boundary being tested. Both retractions came from probing the near
side and inferring the far one.

---

### The earlier (superseded) reasoning

Allowed the venv's Python through the firewall (it was listed explicitly as
*Block incoming connections*; the global "block all" toggle was necessary but
not sufficient), confirmed the control, then re-probed from the same container:

```
CONTROL    Mac coordinator, from the Mac's own tailnet IP   {"ok":true,...}   ✓

container -> 100.65.244.36:8899    (the Mac)        BLOCKED   timeout
container -> 100.102.129.101:8888  (its own host)   OK          3 ms
container -> pypi.org:443                           OK         60 ms
```

So the original reading was right, and is now earned rather than assumed: **a
container on a Windows host cannot reach another machine's Tailscale address**,
while reaching its own host's tailnet address and the public internet fine. The
relay is required on Windows, not a contingency.

The mechanism this implies: packets from the container do arrive at Windows —
that is why its own `100.102.129.101` answers — but Windows does not forward
packets destined for *other* tailnet addresses arriving from the WSL2 NAT.

That is also exactly the asymmetry the relay is built on, so every leg of the
relayed path is now individually measured except the relay's own listener:

```
container -> host's own tailnet IP     3 ms, measured
host      -> Mac over Tailscale        measured in reverse (Mac reaches host:8888)
```

**Caveat, stated rather than glossed:** the friend's *host* reaching the Mac was
not measured directly — only the Mac reaching the host. Tailscale sessions are
bidirectional once established, so this is a safe inference, but it is an
inference.

**Next:**
1. Get the cluster code onto the friend's machine so `p2pgpu cluster relay` can
   run there, then confirm the container reaches it at `100.102.129.101:8899`.
2. Two nodes training for real.

---

## 2026-08-17 — Windows readiness sweep, and an IPv6 bug from v1

**Status:** The container path is proven against the real image. Two bugs
fixed, one of which was shipped in v1.0.0.

**Changed:**
- Every user-facing address the CLI prints is now `escape()`d for rich.
- `cluster coordinator` binds the Tailscale address by default instead of
  `0.0.0.0`.
- `tests/test_cli_output.py` — 10 tests, including a grep over the CLI so a new
  unescaped print of a URL fails the suite rather than a friend's session.

**Why:**
- *IPv6 URLs were rendering with the host missing.* Commit b400e65 bracketed
  IPv6 literals when building the share URL and the docker `-p` flag — correct,
  and only half the problem. Rich parses `[...]` as markup, so the bracketed
  URL printed as `http://:8888/lab?token=...`. The share panel is the string
  the friend copies, so the failure landed on the person least able to diagnose
  it. Same root cause as yesterday's bug, one layer further out.
- *The coordinator handles model weights.* Binding `0.0.0.0` contradicted the
  posture the rest of the project holds ("bind the overlay IP, not `0.0.0.0`",
  v1.0.0). The token is not a reason to be reachable from the café Wi-Fi.

**Measured — the container path, run for real rather than reasoned about:**

Two `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` containers (linux/amd64,
the exact session image), each running `examples/cluster_train.py` against a
coordinator on the host:

```
                container A          container B
before sync     01658692ac0c         b5efc36ede6c
round 2         b318ba07598b         b318ba07598b   <- identical
```

Repeated through `p2pgpu cluster relay` — same result, so the relay carries a
full training round including multi-MB weight payloads.

Image contents, verified by running it: `torch` 2.5.1+cu124, `torchvision`
0.20.1+cu124, `numpy` 2.1.2. **`httpx` is absent**, which is why the example is
stdlib-only. `p2pgpu` is not installed in the container either.

111 tests pass.

**Known gaps for the real run:**
- `docker/Dockerfile` is unreferenced dead code — its tag is still the
  `<yourname>/` placeholder. Sessions use stock PyTorch images and pip-install
  jupyterlab at container start.
- `v2.0-cluster` is not pushed. Friends can reach v1.1 on GitHub but not the
  relay.
- The relay listening on the Windows host will trip a Windows Defender Firewall
  prompt the first time. Expected, not a fault.
- Still unproven: whether a container on **Windows** can reach a Tailscale IP
  directly. The relay exists precisely so this does not have to be answered
  during a live session.

---

## 2026-08-17 — Two machines train one model

**Status:** AVERAGE mode is end-to-end complete and proven locally. The
coordinator had existed since yesterday with nothing to talk to; this is the
half that trains.

**Changed:**
- `cluster/trainer.py` — `ClusterTrainer`: rendezvous, weight exchange, the
  local-SGD loop. `shard()` / `sharded_batches()` for splitting data by rank,
  `estimate_sync_cost()` for sizing a round against a measured link.
- `cluster/demo.py` — `p2pgpu cluster demo`, a run small enough to debug that
  proves the cluster combined rather than merely trained.
- CLI gained a `cluster` sub-app: `coordinator`, `status`, `plan`, `estimate`,
  `demo`. Before this, none of the cluster code was reachable from the CLI.
- `examples/cluster_train.py` — MNIST across two GPUs, for the Jupyter notebook.
- `docs/CLUSTER.md` — the walkthrough, including what does *not* pool.

**Why (decisions worth recording):**

- *Weight checksums, not loss curves, are the proof.* A node training entirely
  alone shows a falling loss too. Nodes start from different seeds on disjoint
  shards, so matching checksums can only happen if the average round-tripped.
  This is the single most valuable line of output in the whole feature — without
  it, a broken cluster looks exactly like a working one.
- *`join()` blocks until the world is full before returning a rank.* The
  coordinator ranks members by sorting ids, so the rank handed to the first
  joiner is provisional. Data is sharded by rank, so acting on a provisional
  rank would silently give two nodes the same data — a bug that shows up as
  "clustering didn't help much", never as an error.
- *Sharding is strided, not contiguous.* Datasets arrive ordered more often than
  people expect. A contiguous split hands one node a biased sample.
- *Optimizer state stays local.* Momentum describes the path a node took through
  its own data; averaging mixes trajectories that were never comparable. Also
  halves what crosses the wire.
- *The example is stdlib-only.* `p2pgpu` is not installed inside the session
  container, so the notebook script talks to the coordinator with `urllib`
  rather than assuming an import that isn't there.
- *Checksums are rounded to 4 decimals before hashing.* Float averaging can land
  two machines a few ULPs apart on the same mean, and a raw byte hash would send
  someone debugging a network that is fine.

**Measured:** 91 tests pass (15 new). Two nodes against a real coordinator over
a real socket, different seeds, disjoint MNIST shards:

```
                node friend-a          node friend-b
before sync     01658692ac0c           b5efc36ede6c     <- genuinely apart
round 0         e61012befa35           e61012befa35
round 1         5edd1f791830           5edd1f791830
round 2         85c6a07eee44           85c6a07eee44     <- identical
```

Synthetic demo over 4 rounds: loss 39.5 -> 0.72, checksums identical throughout.

**Known limits:**
- Never run on two *real* GPUs in two houses. Local only so far.
- PIPELINE mode still cannot train. Forward pass only.
- Both nodes run the same number of local steps, so a slow GPU sets the pace.
- No resume: a node dropping mid-round fails the round.

**Next:**
1. Two friends share (`p2pgpu share`), Mac runs the coordinator, run
   `examples/cluster_train.py` in both notebooks. Compare checksums.
2. Record the real sync overhead — home upload is the binding constraint and
   the estimate is currently theoretical.
3. Still unproven: whether container A can reach a port container B publishes
   on its Tailscale IP. AVERAGE mode does not need it (both containers only
   dial *out* to the coordinator), but PIPELINE mode does.

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

## 2026-08-16 — v1.1 work: automation pass (branch `v1.1-authkey`)

**Status:** v1.0.0 tagged on `main`. Automation continuing on a branch so the
working version stays safe.

**Changed, in order:**
1. **Auto-discovery.** `p2pgpu discover` / `connect` find shared GPUs by asking
   Tailscale's local API for peers and probing them in parallel. No coordinator,
   no accounts. Agent gained `/v1/share`.
2. **`p2pgpu prepare`** moves the multi-GB pull to install time, and
   `start_docker_desktop()` launches the engine instead of telling someone to
   go find a tray icon.
3. **`docker/`** — a purpose-built session image with jupyterlab and sshd baked
   into layers, on a `-runtime` base.
4. **v1.0.0 released** with `CHANGELOG.md`.
5. **Join codes.** `p2pgpu invite` → one code → `p2pgpu join <code>` does the
   tailnet join, the token, and the image pull.
6. **Background service.** `p2pgpu service install` registers with launchd,
   systemd or Task Scheduler.

**Why the join code exists:** setting up the second machine was eight steps, and
the one people actually missed was buried in a web console. All of it is really
two secrets moving between two people, so the code carries both — a Tailscale
auth key and the cluster token. Because both machines then join the *same*
tailnet, node sharing stops being necessary at all.

**Bugs found and fixed this pass:**
- *launchd crash loop.* `KeepAlive=true` restarts on **any** exit, so an agent
  that could not bind its port respawned forever, burning CPU with nothing
  visible to the user. Found by installing the service on this Mac while a
  stale agent from earlier still held 8777. Now `KeepAlive` only fires on
  failure, and the agent detects an existing healthy agent and exits 0.
- *Every image was a `-devel` tag* — 7-9 GB versus ~4 GB for `-runtime`, for
  nvcc and CUDA headers a training session never touches. This is the download
  that hurt most in the live run.
- *`discovery._tailscale_exe`* had `x if shutil.which else x`: identical
  branches, condition always truthy. Dead code that read like a real check.
- *`agent` and `serve`* were duplicate commands.
- *`prepare`* imported `gpu_compat` and never used it.

**Measured:** 56 tests. Service verified on macOS — installs, serves,
auto-restarts on crash, exits cleanly when already running, survives `kill`.

**Next:** publish the session image; single installer; then the GUI.

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
