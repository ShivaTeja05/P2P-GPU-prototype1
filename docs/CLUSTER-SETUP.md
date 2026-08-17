# Cluster setup — from download to a running two-GPU training job

Everything, in order, for both Windows and Linux. Follow your own track and
ignore the other one.

If you only want to *share one GPU* with one person, you want
[GUIDE.md](../GUIDE.md) instead. This document is for **two or more GPUs
training one model together**.

---

## Step 0 — Get the right version

This matters more than anything else on this page. The repo has three branches
and only one of them has the cluster.

| Branch | What it has | Use it? |
|---|---|---|
| `main` | v1.0.0 — share one GPU with one person | ❌ no cluster code |
| `v1.1-authkey` | join codes, background service | ❌ still no cluster |
| **`v2.0-cluster`** | **cluster, coordinator, relay** | ✅ **this one** |

**If you have repo access:**

```bash
git clone -b v2.0-cluster https://github.com/ShivaTeja05/P2P-GPU-prototype1.git
```

**If you were sent a ZIP:** extract it, then confirm you got the right branch —
this folder must exist:

```
p2pgpu/cluster/
```

If `p2pgpu/cluster/trainer.py` is missing, you have the wrong version. Stop and
ask for the `v2.0-cluster` copy.

> **Windows only — unblock the ZIP before extracting.** Right-click the `.zip` →
> Properties → tick **Unblock** → OK. Windows silently marks downloaded files,
> and every `.bat` file will refuse to run if you skip this. This wastes more
> people's time than any other step here.

---

## Who does what

Three roles. One machine can hold two of them.

| Role | Machine | Needs a GPU? | Runs |
|---|---|---|---|
| **Coordinator** | any laptop, incl. a Mac | ❌ no | `p2pgpu cluster coordinator` |
| **GPU node** | Windows or Linux + NVIDIA | ✅ yes | `p2pgpu share` |
| **Driver** | whoever runs the training | ❌ no | opens both notebooks |

The coordinator never loads a model and never touches a GPU. Put it on the
machine that *can't* train — a Mac is perfect.

**macOS cannot be a GPU node.** Docker GPU passthrough is NVIDIA-only. A Mac can
coordinate and it can train as an extra node natively, but it cannot share its
GPU to the cluster.

---

# Part 1 — Everyone does this

## 1.1 Install Tailscale and sign in

This is what lets the machines reach each other without port forwarding, public
IPs, or touching your router.

- **Windows** — [download the installer](https://tailscale.com/download/windows), run it, sign in
- **Linux** — `curl -fsSL https://tailscale.com/install.sh | sh` then `sudo tailscale up`
- **macOS** — `brew install --cask tailscale-app`, open it, sign in

**Everyone can use their own account.** You do not share a login. The person who
owns the tailnet either invites the others, or hands out a join code (below).

✅ **Check it worked:**

```bash
tailscale ip -4
```

You should get an address starting with `100.`. Write it down.

## 1.2 Join the same tailnet

The person who started the tailnet runs:

```bash
p2pgpu invite
```

That prints one code. Send it privately — it contains an auth key. Everyone
else runs:

```bash
p2pgpu join <code>
```

✅ **Check it worked** — everyone should see everyone:

```bash
tailscale status
```

Every machine in the cluster must appear, and none should say `offline`.

---

# Part 2 — GPU machines only

Skip this entirely if your machine has no NVIDIA GPU.

## Track A — Windows

### A1. Run the setup

Double-click:

```
Setup.bat
```

It installs Python, creates a virtual environment, installs `p2pgpu`, and checks
for Docker. **It may ask you to reboot — do it**, then run `Setup.bat` again.

> If Windows says *"Windows protected your PC"*, click **More info** →
> **Run anyway**. If PowerShell says *"running scripts is disabled"*, you missed
> the Unblock step in Step 0.

### A2. Install Docker Desktop with the WSL2 backend

If `Setup.bat` said Docker is missing:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Settings → General → tick **Use the WSL 2 based engine**
3. Settings → Resources → WSL Integration → enable your distro
4. Update your **Windows** NVIDIA driver to the latest version
5. Start Docker Desktop and wait for the whale icon to go steady

> **Do not install Linux NVIDIA drivers inside WSL.** Recent Windows drivers
> expose the GPU to WSL2 automatically, and installing them inside WSL breaks
> the passthrough. This is the classic mistake.

### A3. Verify the GPU really reaches a container

Double-click:

```
Check-Setup.bat
```

✅ **You want to see:** `GPU is visible inside Docker. Ready to share.`

This actually runs `nvidia-smi` inside a container — it is the only check that
proves the passthrough works. Its output is also saved to
`check-setup-output.txt`, which is the file to send if you need help.

### A4. Pre-download the session image

```
Prepare.bat
```

This pulls ~4 GB. **Do it before the session, not during it** — otherwise
everyone sits waiting. It only happens once.

---

## Track B — Linux

### B1. Install Docker

```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER
```

Log out and back in so the group change takes effect.

### B2. Install the NVIDIA Container Toolkit

Follow [NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for your distro. On Ubuntu/Debian it ends with:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### B3. Install p2pgpu

From inside the repo folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

Add `.venv/bin` to your PATH, or prefix commands with `.venv/bin/`.

### B4. Verify

```bash
p2pgpu doctor
```

✅ **You want:** `GPU is visible inside Docker. Ready to share.`

### B5. Pre-download the session image

```bash
p2pgpu prepare
```

---

# Part 3 — The coordinator machine

Any machine, GPU or not. Install `p2pgpu` the same way as Linux above (a Mac
uses the identical commands), then:

```bash
p2pgpu cluster coordinator
```

✅ **It prints the URL every GPU machine needs:**

```
coordinator listening on 100.65.244.36:8899

Point each GPU machine at one of these:
  http://100.65.244.36:8899
```

**Copy that URL.** Leave this window open for the whole session — closing it
ends the cluster.

### Let it accept connections — do not skip this

The coordinator is the one thing in the cluster that receives inbound
connections. A firewall here blocks every GPU node, and the symptom appears on
*their* screen, not yours, so it is easy to spend an hour debugging the wrong
machine.

**macOS** — check with:

```bash
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

If it says *"blocking all non-essential incoming connections"*, open
**System Settings → Network → Firewall → Options** and either turn off
**Block all incoming connections**, or add your Python binary to the allowed
list. Nothing reaches the coordinator until you do.

**Windows** — Defender Firewall prompts the first time; allow it on **private
networks**.

**Linux** — if `ufw` is active: `sudo ufw allow in on tailscale0 to any port 8899`

✅ **Verify from the coordinator machine itself**, using its Tailscale address
rather than `localhost` — that is the address the GPU nodes will use:

```bash
curl http://100.x.y.z:8899/health
```

You want `{"ok":true,"role":"coordinator",...}`. An empty reply or a hang means
the firewall is still in the way. Testing `127.0.0.1` instead will succeed even
when every remote machine is blocked, which is why it is the wrong check.

---

# Part 4 — Rehearse before involving anyone

Ten minutes, on the coordinator machine alone. Do this once, before you
coordinate three people.

Open two more terminals and run:

```bash
p2pgpu cluster demo --coordinator http://127.0.0.1:8899 --as node-a
```

```bash
p2pgpu cluster demo --coordinator http://127.0.0.1:8899 --as node-b
```

✅ **Both must end with the same weight checksum.** If they match, the whole
mechanism works and all that is left is distance.

---

# Part 5 — Run the real cluster

## 5.1 Each GPU machine starts sharing

**Windows:** double-click `Share-My-GPU.bat`
**Linux:** `p2pgpu share --hours 3`

Each prints a URL. That URL is a notebook running on that machine's GPU.

## 5.2 The driver opens both notebooks

```bash
p2pgpu discover
```

```bash
p2pgpu connect --host <their-hostname>
```

Do this once per GPU machine, so you end up with one browser tab per GPU.

## 5.3 Check the container can reach the coordinator

**Do this before anything else.** Paste into a cell in **each** notebook,
replacing the address with your coordinator URL:

```python
import urllib.request; print(urllib.request.urlopen("http://100.x.y.z:8899/health", timeout=8).read())
```

- ✅ Prints `{"ok":true,...}` → carry on to 5.4
- ❌ Times out → do **5.3b** first

> **If it times out, check the coordinator's machine before blaming Windows.**
> A firewall on the coordinator gives exactly the same symptom, and it is the
> more likely cause — see Part 3. Measured inside a real session container, the
> public internet and the container's own host were both reachable, so the
> container's networking is not broken in general.

### 5.3b If it still times out — start the relay

The container's host is the WSL2 virtual machine, which does not carry the
Windows Tailscale interface. The relay forwards from an address the container
*can* reach, out over the machine's own Tailscale connection.

On that GPU machine, in a second terminal:

```bash
p2pgpu cluster relay --coordinator http://100.x.y.z:8899
```

> Windows Defender Firewall will ask to allow it the first time. Allow it on
> **private networks**, or the container still will not get through.

Then in that notebook, point at **that GPU machine's own Tailscale IP** — the
address `tailscale ip -4` prints on the machine running the relay, *not* the
coordinator's:

```python
COORDINATOR = "http://100.102.129.101:8899"   # ← the GPU machine's own address
```

> **Use the host's Tailscale IP, not `host.docker.internal`.** The obvious
> choice is the documented Docker alias, but on the machine we measured it does
> not resolve inside the container at all, while the host's own Tailscale
> address answered in 3 ms. If your setup does resolve `host.docker.internal`,
> that works too — try the Tailscale IP first.

## 5.4 Paste the training script into every notebook

Open `examples/cluster_train.py`, copy the whole file into a cell in **each**
notebook, and edit the four lines at the top:

```python
COORDINATOR = "http://100.x.y.z:8899"   # the URL from Part 3
TOKEN       = "..."                     # same on every machine
NODE_ID     = "friend-a"                # ← MUST be different on each machine
WORLD_SIZE  = 2                         # how many GPUs are joining
```

The token is the same one every machine already has from `p2pgpu join`. Read it
from `%USERPROFILE%\.p2pgpu\cluster_token` on Windows, or
`~/.p2pgpu/cluster_token` on Linux and macOS.

> **`NODE_ID` must differ on every machine.** Two nodes with the same id are
> counted as one member and the cluster never forms — it just waits forever.

## 5.5 Run every cell

Run them in any order. The first node to start waits for the others.

✅ **You should see:**

```
cluster formed: ['friend-a', 'friend-b'] -- this machine is rank 0
this node trains on 30000 of 60000 images

round     loss     acc    train    sync  weights
    0   2.2576   27.7%     1.4s    0.8s  e0d03035d455
    1   2.2458   39.3%     1.1s    0.0s  5473b25badc4
    2   2.0373   48.2%     1.1s    0.0s  b318ba07598b
```

## 5.6 Confirm it actually clustered

**Compare the `weights` column across the notebooks.**

- **Identical** → the GPUs really trained one model together ✅
- **Different** → each was training alone; the averaging never landed ❌

A falling loss proves nothing on its own — a node training entirely by itself
also shows a falling loss. The checksum is the only real evidence, because the
nodes start from different random seeds on disjoint data.

---

## Training your own model instead of MNIST

Replace two functions in `examples/cluster_train.py` and change nothing else:

- `build_model()` — your architecture. **Must be identical on every machine**;
  averaging compares parameter names and shapes and rejects a mismatch.
- `load_data()` — your dataset. Keep the `[rank::world_size]` striding so each
  node gets a different, unbiased slice.

Your model must fit on the **smallest** GPU in the cluster. This mode gives
every node a full copy — it combines compute, not VRAM.

### Tuning the sync interval

Every sync sends the whole model in each direction. Check the cost first:

```bash
p2pgpu bench http://100.x.y.z:8777
```

```bash
p2pgpu cluster estimate --params 100 --upload-mbps 40 --download-mbps 120 --sync-every 200
```

If sync overhead is above ~25%, raise `SYNC_EVERY`. Averaging less often costs a
little convergence quality and buys back most of the wall clock.

---

## When something goes wrong

| Message | Cause and fix |
|---|---|
| `waited 300s for 2 nodes; only 1 joined` | The other node has not started, or both used the same `NODE_ID`. |
| `Could not reach the coordinator` | Coordinator window closed; or machine off the tailnet (`tailscale status`); or the container can't see the tailnet → **5.3b**. |
| `nodes submitted different model shapes` | `build_model()` differs between machines. They must match exactly. |
| `round N timed out with 1/2 nodes` | Someone's share expired mid-run, or their machine went to sleep. |
| `cluster already formed with world_size=N` | Leftover state. Restart the coordinator. |
| Checksums differ at the end | The averaging never landed. Check the coordinator window — each round should show two submissions. |
| `Windows protected your PC` | You skipped the Unblock step in Step 0. |
| `running scripts is disabled` | Same cause as above. |
| `p2pgpu is not installed yet` | Run `Setup.bat` first. |
| Everything is very slow | Check `train` vs `sync` in the output. If `sync` dominates, raise `SYNC_EVERY`. |

---

## What this does and does not give you

Worth knowing before you expect the wrong thing:

| Resource | Pools across machines? |
|---|---|
| **Compute** | ✅ yes — both GPUs work at once |
| **VRAM** | ⚠️ no, not in this mode — each GPU holds a full copy |
| **System RAM** | ❌ no — separate machines |
| **Disk** | ❌ no |

Two GPUs do not become one bigger GPU. They become two GPUs working on the same
problem. See [CLUSTER.md](CLUSTER.md) for the full explanation and the
mode that *does* combine VRAM.
