# Training on two GPUs at once

You and two friends have three machines. This is how you turn two of their GPUs
into one training run.

Read the next section before you start — it will save you from expecting
something the design cannot give you.

---

## What "clustering" does and does not give you

The obvious mental model is that two GPUs merge into one big virtual machine
with pooled everything. That is what a cloud instance looks like, and it is
**not** what happens here. Being clear about it upfront is cheaper than
discovering it three hours in.

| Resource | Pools across machines? | Why |
|---|---|---|
| **Compute** | ✅ yes | Both GPUs do useful work at once. This is the real win. |
| **VRAM** | ⚠️ only by splitting the model | No CUDA-level way to make 6 GB + 16 GB *look like* 22 GB. A model can be cut across cards (PIPELINE, below), but that is deliberate and model-aware, not automatic. |
| **System RAM** | ❌ no | Two separate kernels. Nothing merges them. |
| **Disk** | ❌ no | Two filesystems. You can copy files between them; that is not one volume. |
| **CPU cores** | ❌ no | Same reason. |

A cloud "8×A100 box" is **one machine, one kernel**, with GPUs on NVLink at
~600 GB/s and sub-microsecond latency. Your cluster is separate computers at
~110 ms — roughly a hundred thousand times slower per hop. Everything below is
shaped by that one number, and any design that pretends otherwise fails quietly
rather than loudly.

**What you get instead:** one control node, several GPUs, one training script.
That is genuinely useful, and it is honest.

### One terminal across both machines?

There is no such thing — a shell belongs to one kernel. What you actually want
is one *driver*: a single script on your Mac (or one notebook per GPU running
the same script) that both machines follow. That is what `p2pgpu cluster` is.

SSH still gets you a shell on *each* machine, one at a time, for poking around.
Useful, but it is not the cluster.

---

## The two modes, and which one you want today

```
AVERAGE  ── combines COMPUTE ────────────────── works today ✅
  Each GPU holds a FULL copy of the model, trains on a different
  half of the data, and the machines meet every N steps to average
  weights.  Model must fit on the smaller card.

PIPELINE ── combines VRAM ─────────────── forward pass only ⚠️
  The model is CUT into layer ranges, one range per GPU. Lets you
  load a model that fits on NEITHER card. Inference works and is
  numerically exact; TRAINING through it is not built yet — that
  needs gradients sent back up the stages.
```

**For training today: AVERAGE.** Check your model fits on the RTX 4050's 6 GB.
If it does, you are in the right mode and everything below applies.

Why not the thing datacenters use (tensor parallelism)? It all-reduces twice per
layer. An 80-layer model at 110 ms costs about 4.8 seconds *per token*. It needs
NVLink-class latency, and there is no way around that over the internet.

---

## What the session container gives you

Verified by running the actual image, not by reading the Dockerfile:

```
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime      measured on a live RTX 4050 session
  torch        2.5.1+cu124   ✅        GPU visible, 6.0 GB total / 4.95 GB free
  torchvision  0.20.1+cu124  ✅        so the MNIST example works
  numpy        2.1.2         ✅
  httpx        0.28.1        ⚠️        present, but only as a jupyterlab dependency
  p2pgpu       absent        ❌        never installed in the container
```

`httpx` is not in the base image — it arrives because the container runs
`pip install jupyterlab` at start, and jupyterlab pulls it in. That is a
transitive dependency of an unrelated package, which is exactly the kind of
thing that disappears in a version bump. So `examples/cluster_train.py` uses
stdlib `urllib` and depends on nothing but torch.

Have both GPU machines pull the image **before** the session, not during:

```bash
p2pgpu prepare
```

---

## Rehearse it on one machine first

Ten minutes, no friends, no GPUs. Do this before coordinating three people.

**Terminal 1 — the coordinator:**

```bash
p2pgpu cluster coordinator
```

**Terminal 2 and 3 — two pretend nodes:**

```bash
p2pgpu cluster demo --coordinator http://127.0.0.1:8899 --as node-a
```

```bash
p2pgpu cluster demo --coordinator http://127.0.0.1:8899 --as node-b
```

Both should end with the **same weight checksum**. That is the whole test. If
the checksums match, the plumbing is correct and the only thing left to add is
distance.

---

## The real run

### 1. Everyone joins the tailnet

Nothing new — same as sharing a single GPU. You send each friend a join code:

```bash
p2pgpu invite
```

```bash
p2pgpu join <code>
```

Check all three machines can see each other:

```bash
p2pgpu discover
```

### 2. Your Mac runs the coordinator

The coordinator is a meeting point. It never loads a model and never touches a
GPU — it holds a barrier ("wait until everyone submitted round N") and computes
an average. The machine that is *least* useful for training is exactly right
for it.

```bash
p2pgpu cluster coordinator
```

It prints the URL to hand to each GPU machine. Leave it running.

### 3. Each friend shares their GPU

On each friend's Windows/Linux machine, unchanged from v1:

```bash
p2pgpu share --hours 3
```

### 4. You attach to each notebook

Two browser tabs, one per friend's GPU:

```bash
p2pgpu discover
```

```bash
p2pgpu connect --host <their-hostname>
```

### 5. Paste the training script into both notebooks

Copy [`examples/cluster_train.py`](../examples/cluster_train.py) into a cell on
**each** GPU. It is deliberately self-contained — stdlib `urllib` plus torch —
because `p2pgpu` is not installed inside the session container.

Edit the four lines at the top. **`NODE_ID` must differ on each machine**; two
nodes with the same id are seen as one member and the cluster never forms.

```python
COORDINATOR = "http://100.x.y.z:8899"   # the exact URL step 2 printed
TOKEN       = "..."                     # same on every machine
NODE_ID     = "friend-a"                # ← different on each!
WORLD_SIZE  = 2
```

Run both cells. The first one to start waits for the second.

> **If a notebook says `Could not reach the coordinator`, this is the fix.**
>
> The GPU machine may be on the tailnet while the *container* is not. On Windows
> the container's host is the WSL2 VM, and the Windows Tailscale interface does
> not live there — so `100.x.y.z` may have no route from inside the container
> even though the machine itself reaches it fine.
>
> **Check the coordinator's own machine first.** A firewall there produces an
> identical symptom, and it is the more common cause. On macOS, "Block all
> incoming connections" silently drops everything; see the setup guide.
>
> On that friend's machine, in a second terminal:
>
> ```bash
> p2pgpu cluster relay --coordinator http://100.x.y.z:8899
> ```
>
> Then point their notebook at **that machine's own Tailscale IP** — the one
> `tailscale ip -4` prints there, not the coordinator's:
>
> ```python
> COORDINATOR = "http://100.102.129.101:8899"
> ```
>
> Not `host.docker.internal` — measured inside a real session container, that
> alias does not resolve at all, while the host's own Tailscale address answered
> in 3 ms.
>
> The relay forwards over that machine's own Tailscale connection. It adds a
> hop, not a privilege — it only carries traffic the tailnet already allows.

### 6. Read the output

```
round     loss     acc    train    sync  weights
    0   2.2955   15.3%     1.6s    0.5s  e61012befa35
    1   2.2919   17.5%     0.2s    0.0s  5edd1f791830
    2   2.2735   33.3%     0.2s    0.0s  85c6a07eee44
```

**Compare the `weights` column across the two notebooks.** They start from
different random seeds and train on disjoint data, so the checksums can only
match if the averaging round-tripped across the internet.

- **Identical** → the two GPUs really trained one model.
- **Different** → each was training alone. The averaging never landed.

A falling loss on its own proves nothing — a node training entirely by itself
also shows a falling loss. The checksum is the actual evidence.

---

## Tuning `SYNC_EVERY`, the one knob that matters

Every sync sends the **full model in each direction**, in fp32. A 100M-parameter
model is 382 MB up and 382 MB down, *per round, per node*.

Check what that costs on your link before a long run:

```bash
p2pgpu bench http://100.x.y.z:8777
```

```bash
p2pgpu cluster estimate --params 100 --upload-mbps 40 --download-mbps 120 --sync-every 200
```

```
weights on the wire    │ 381.5 MB each way
upload                 │ 76.3 s
training between syncs │ 20 s (200 steps)
sync overhead          │ 84%
```

84% of wall-clock spent talking is a cluster that is slower than one GPU alone.
The fix is always the same: **raise `SYNC_EVERY`.** Averaging less often costs a
little convergence quality and buys back most of the wall clock. Aim for sync
overhead under ~10%.

Home broadband upload is usually the binding constraint, not download — and the
weights go *up* from both GPUs. Size your expectations on the upload number.

---

## Why the data is split the way it is

`shard()` gives node 0 items `[0, 2, 4, ...]` and node 1 `[1, 3, 5, ...]` —
strided, not two contiguous halves.

Contiguous looks tidier and is a trap. Datasets arrive ordered more often than
people expect — by class, by length, by collection date. A contiguous split
hands one node a biased sample, and averaging a model that never saw an `8` with
one that never saw a `1` is worse than not clustering at all.

---

## Two things that stay local on purpose

**Optimizer state.** Momentum buffers describe the path a node took through its
own data. Averaging them mixes trajectories that were never comparable. Keeping
them local is standard for local-SGD and halves what crosses the wire.

**Everything between syncs.** The nodes do not talk during the `SYNC_EVERY`
steps. That is the entire design: at 110 ms, anything synchronising per step is
dead on arrival. The stall is paid once and amortised over hundreds of steps.

---

## When it goes wrong

| Symptom | Cause |
|---|---|
| `waited 300s for 2 nodes; only 1 joined` | Second node not started, or both used the same `NODE_ID`. |
| `Could not reach the coordinator` | Coordinator not running; or the GPU machine is off the tailnet (`tailscale status`); or the machine is on but the *container* is not, which is the normal case on Windows — run `p2pgpu cluster relay` on the GPU machine and point the script at that machine's own Tailscale IP. |
| `nodes submitted different model shapes` | The two scripts build different architectures. They must be identical. |
| `round N timed out with 1/2 nodes` | Someone's share expired mid-run, or their GPU is much slower. Both nodes run the *same* number of local steps, so the faster one waits. |
| Checksums differ at the end | The averaging never landed. Check the coordinator's log; each round should show two submissions. |
| `cluster already formed with world_size=N` | A previous run left state behind. Restart the coordinator. |

---

## Planning a PIPELINE split (not trainable yet)

To see whether a model *could* fit across the combined VRAM:

```bash
p2pgpu cluster plan --layers 32 --params 8 --gpu rtx4050:5 --gpu rtx4080:15
```

```
rank 0  rtx4050   layers [  0,   8)    3.7 GB of   5.0 GB free  (88%)
rank 1  rtx4080   layers [  8,  32)   11.2 GB of  15.0 GB free  (88%)

combined: 14.9 GB of model across 20.0 GB of VRAM in 2 machines
```

Note it does not split 16/16 — an even split would OOM the 4050 while the 4080
sat two-thirds empty. It weights by free VRAM, which is what makes mismatched
cards usable, and mismatched is the normal case when they belong to different
people.

The forward pass through a split model is built and verified numerically exact
(`max |whole − split| = 0.000e+00` on GPT-2 and Llama). Generation and training
through it are not built yet.
