# Setup

Two machines. One has the GPU ("**host**"), one wants to use it ("**guest**").

---

## Both machines: Tailscale

This is what lets the two talk to each other through two home routers without
port forwarding, dynamic DNS, or exposing anything to the public internet.

Install from [tailscale.com/download](https://tailscale.com/download), then:

```bash
sudo tailscale up
```

### Which account should each of you use?

You do **not** have to share one login. Three options:

| | How | Who sees what | Verdict |
|---|---|---|---|
| **Same account** | Both sign into one login | everything | Works, but you're sharing credentials |
| **Invite to your tailnet** | Friend gets their own login, joins your tailnet | all devices on the tailnet | Fine for close friends |
| **Node sharing** | Each keeps their own tailnet; they share *only* the GPU machine | just that one machine | **Recommended** |

**Node sharing** is the right default. You each keep your own identity, and you
get access to exactly one machine instead of their whole home network.

The GPU owner does this:

1. Open the [admin console](https://login.tailscale.com/admin/machines) → Machines
2. Find the GPU machine → **Share...**
3. Send the invite link to the person borrowing it

The borrower accepts, and that machine appears in their own Machines list with
its `100.x.y.z` address — reachable exactly as if it were their own device.

Tailscale **quarantines shared machines by default**: the shared machine can
*receive* connections from your tailnet but cannot *initiate* them into it.
That's the right direction for us — you connect in to the notebook, and their
GPU box gets no ability to reach back into your network.

The free Personal plan covers 6 users with unlimited personal devices, so none
of this costs anything. If either side has a custom ACL policy file, make sure
it permits the shared device; the default policy already does.

### Check it worked

```bash
tailscale ip -4
```

You should get a `100.x.y.z` address on each. From the guest, confirm you can
reach the host:

```bash
ping 100.x.y.z
```

If that works, the hard networking part is done.

---

## Both machines: p2pgpu

```bash
git clone https://github.com/ShivaTeja05/P2P-GPU-prototype1.git && cd P2P-GPU-prototype1
```

### Linux and macOS

```bash
uv venv --python 3.12 && uv pip install -e .
```

No `uv`? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`, or
use a normal `python3.12 -m venv .venv && .venv/bin/pip install -e .`.

### Windows

Double-click `Setup.bat`. It does the same thing, plus installs Python if it is
missing and checks for Docker. It may ask you to reboot — do it, then run it
again.

If Windows says *"Windows protected your PC"* or PowerShell says *"running
scripts is disabled"*, you downloaded the repo as a ZIP and did not unblock it:
right-click the `.zip` → Properties → tick **Unblock** → extract again.

Python 3.12 is pinned because PyTorch has no 3.14 wheels yet.

---

## Host only: Docker + NVIDIA Container Toolkit

This is the part that actually hands the GPU to a container, and it's the step
most likely to need troubleshooting.

### Linux

Install Docker Engine:

```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER
```

Log out and back in so the group change takes effect.

Then install the NVIDIA Container Toolkit — follow
[NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for your distro. On Ubuntu/Debian it ends with:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### Windows

Docker Desktop with the **WSL2 backend**. Recent NVIDIA Windows drivers expose
the GPU to WSL2 automatically — you do **not** install Linux NVIDIA drivers
inside WSL, which is the classic mistake.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Settings → General → enable "Use the WSL 2 based engine"
3. Settings → Resources → WSL Integration → enable your distro
4. Make sure your Windows NVIDIA driver is current

Then run `p2pgpu` **on Windows** — `Setup.bat` and the other `.bat` launchers,
or `.venv\Scripts\p2pgpu.exe` directly. Not inside WSL.

WSL2 is where the *container* runs; it is not where you drive it from. `p2pgpu`
has Windows-specific handling that only takes effect when it is running on
Windows: it finds `tailscale.exe`, which the Windows installer leaves off
`PATH`, and it normalises Windows paths for Docker's `-v` flag. Run it from
inside WSL and it misses both, then binds the notebook to an address your
friend cannot reach.

### Verify

```bash
p2pgpu doctor
```

On Windows, double-click `Check-Setup.bat` instead — same check, and it saves
the output to `check-setup-output.txt`, which is the file to send if you need
help.

This checks Docker, the NVIDIA runtime and Tailscale, then actually runs
`nvidia-smi` inside a container. If it prints your GPU, you're ready.

Doing it by hand is the same thing:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Optional: the capability agent

Only needed for `p2pgpu inspect` and `p2pgpu bench`. On both machines:

```bash
p2pgpu init          # first machine — prints a token
p2pgpu init --token <token>   # second machine
```

Send the token privately (Signal, a password manager). Never commit it — it's
gitignored, but be careful anyway.

Then on the host:

```bash
p2pgpu serve
```

And from the guest:

```bash
p2pgpu bench http://100.x.y.z:8777
```

You'll get RTT and throughput, plus estimates for how long a 1 GB dataset push
or a 5 GB checkpoint pull will actually take. Worth knowing before you start.

---

## Troubleshooting

**`doctor` says "Docker cannot see the NVIDIA runtime"**
The toolkit is installed but not registered. Run
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

**`doctor` says "No Tailscale address"**
Run `sudo tailscale up`. If Tailscale is genuinely not wanted, pass
`--bind-ip <lan-ip>` to `share` — but then it only works on the same network.

**Share starts, but the guest can't reach the URL**
Check `tailscale status` on both. Confirm the host still shows the share with
`p2pgpu status`. The port binds to the Tailscale IP specifically, so a LAN IP in
the URL will not work from outside.

**Notebook starts then dies**
`p2pgpu logs`. Usually the image is missing something or the port is taken.

**PyTorch dataloader crashes with a shared-memory error**
Shouldn't happen — `--shm-size=8g` is set. If you overrode the image, keep it.

**First `share` takes forever**
It's pulling a multi-GB PyTorch image. Once. Later shares start in seconds.
