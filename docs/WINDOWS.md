# Windows guide

You don't need WSL, a terminal, or any Python knowledge. Download the repo and
double-click things.

---

## Get the files

Either `git clone`, or on the GitHub page click **Code → Download ZIP** and
extract it somewhere permanent like `C:\p2pgpu` — not your Downloads folder,
because the tool keeps a workspace directory next to it.

---

## If you want to SHARE your GPU

You have the NVIDIA card. Your friend borrows it.

### 1. Double-click `Setup.bat`

It checks for Python, uv, Tailscale, Docker Desktop and your NVIDIA driver, and
offers to install whatever is missing. **Nothing installs without you saying
yes.** Safe to run again any time.

Docker Desktop usually wants a reboot. Run `Setup.bat` again afterwards.

### 2. Finish the Docker Desktop setup

Once installed, open Docker Desktop and:

- Complete the first-run wizard
- **Settings → General → tick "Use the WSL 2 based engine"**
- Leave it running (there should be a whale icon in your system tray)

> **The one mistake to avoid:** do *not* install Linux NVIDIA drivers inside
> WSL. Your normal Windows NVIDIA driver already exposes the GPU to WSL2 and
> to Docker. Installing Linux drivers on top breaks it.

### 3. Sign in to Tailscale

Open Tailscale from the Start menu and sign in. This is what lets the two
machines find each other through both home routers, with nothing exposed to the
internet — it works across cities and countries, not just your house.

You do **not** need to share a login with your friend. Best option: sign in with
your own account, then in the [admin console](https://login.tailscale.com/admin/machines)
find this PC and click **Share...**, and send the invite to your friend. They
accept, and they can reach this one machine — not the rest of your network.

### 4. Double-click `Check-Setup.bat`

This is the real test. It prints your GPU, its architecture, the driver, and
which container image it picked — then actually launches a container and runs
`nvidia-smi` inside it.

If it ends with **"GPU is visible inside Docker. Ready to share."** you're done.
Anything else prints exactly what to fix.

### 5. Double-click `Share-My-GPU.bat`

Asks how many hours, starts the container, and **copies the URL to your
clipboard**. Paste it to your friend.

The first run downloads a few GB of PyTorch image. That happens once.

### While sharing

| File | Does |
|---|---|
| `Sharing-Status.bat` | Is it running, how long is left |
| `Stop-Sharing.bat` | Stop right now |

It also stops by itself when the time runs out — that's deliberate, so a share
you forgot about can't hold your GPU for a week.

---

## If you only want to USE a friend's GPU

You don't need Docker or an NVIDIA card at all.

### 1. Double-click `Setup-Guest-Only.bat`

Skips Docker entirely. Installs Python, uv and Tailscale.

### 2. Sign in to Tailscale

Use your own account — you do not need your friend's login. Ask them to go to
their Tailscale admin console, find their GPU machine, click **Share...** and
send you the invite. Accept it, and that machine shows up in your own device
list.

### 3. Copy the URL your friend sent you

### 4. Double-click `Connect-To-GPU.bat`

It notices a share URL on your clipboard and offers to use it. Confirm, and
JupyterLab opens in your browser.

Check you really have the GPU:

```python
import torch
print(torch.cuda.get_device_name(0))
```

Save anything you want to keep into `/workspace` — that folder lives on your
friend's machine and survives after the session ends.

---

## What each file does

| File | Who runs it |
|---|---|
| `Setup.bat` | GPU owner — first-time setup |
| `Setup-Guest-Only.bat` | Borrower — first-time setup, no Docker |
| `Check-Setup.bat` | GPU owner — verify everything works |
| `Share-My-GPU.bat` | GPU owner — start sharing |
| `Sharing-Status.bat` | GPU owner — check on it |
| `Stop-Sharing.bat` | GPU owner — stop now |
| `Connect-To-GPU.bat` | Borrower — connect |

---

## Troubleshooting

**"running scripts is disabled on this system"**
Use the `.bat` files, not the `.ps1` files directly. The `.bat` wrappers set the
execution policy for that one run only, so nothing about your system changes.

**`Check-Setup.bat` says Docker is installed but not running**
Start Docker Desktop and wait for the whale icon to stop animating.

**"Docker does not list an NVIDIA runtime"**
On Windows this is usually just a warning, because Docker Desktop reaches the
GPU through WSL2 without advertising a separate runtime. `Check-Setup.bat` runs
a real passthrough test — trust that result over the warning.

**GPU test fails but `nvidia-smi` works in Windows**
Almost always one of: Docker Desktop isn't using the WSL2 engine, the Windows
NVIDIA driver is old, or Linux NVIDIA drivers were installed inside WSL. Check
in that order.

**Friend can't reach the URL**
Run `Sharing-Status.bat` to confirm the share is alive. Then have both of you
check Tailscale is connected — and if you used node sharing, that the invite was
accepted. The URL contains a `100.x.y.z` address, which only works over
Tailscale; it is not reachable from the plain internet, by design.

**Everything worked, then stopped**
The share expired. Run `Share-My-GPU.bat` again — it mints a fresh URL, and the
old one stops working on purpose.

**PyTorch can't see the GPU inside the notebook**
Run `Check-Setup.bat` and look at the image it selected. If your card is very
new or very old, the automatic choice may be wrong — you can override it from a
terminal with `p2pgpu share --image <something-else>`.

---

## Security, briefly

Sharing your GPU means your friend can run code on your machine, inside a
container. The container cannot read your files — only the `p2pgpu-workspace`
folder is shared — and the connection is only reachable over Tailscale, not
from your home Wi-Fi or the internet.

But container isolation is good, not perfect. **Share with people you'd lend
your laptop to.** Full detail in [SECURITY.md](SECURITY.md).
