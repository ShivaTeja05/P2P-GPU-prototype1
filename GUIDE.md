# Guide: use your friend's GPU

Your friend has a gaming PC with a good NVIDIA card. You have a laptop that
doesn't. This lets you run code on their GPU from anywhere — different city,
different country, doesn't matter.

You'll be writing code in a normal Jupyter notebook in your browser. It just
happens to run on their hardware.

**Time needed:** about 20 minutes the first time, mostly waiting for downloads.
After that it's two clicks.

---

## First: which one are you?

Read only your half of this guide.

| | You are... | Go to |
|---|---|---|
| 🖥️ | The one **with** the NVIDIA GPU, lending it out | [Part A](#part-a--im-lending-my-gpu) |
| 💻 | The one **borrowing** a friend's GPU | [Part B](#part-b--im-borrowing-a-gpu) |

Send this guide to your friend so you can each do your half.

---

## What the pieces are (30 seconds)

You don't need to understand these, but people ask:

- **Tailscale** — creates a private encrypted tunnel between your two computers.
  Works through both home routers. **Nothing gets exposed to the internet** —
  no ports opened, no router settings changed.
- **Docker** — runs the shared session in a sealed box on the GPU machine, so
  the borrower gets the graphics card but *not* the owner's files.
- **Jupyter** — the notebook you type code into.

---

# Part A — I'm lending my GPU

You need: an NVIDIA graphics card, Windows or Linux.

*(A Mac cannot lend its GPU — Docker GPU passthrough is NVIDIA-only. Macs can
still borrow, see Part B.)*

## Step A1 — Install Tailscale and sign in

Download from **[tailscale.com/download](https://tailscale.com/download)**,
install, and sign in with any account you like (Google, GitHub, Microsoft).

**Use your own account.** You do *not* need to share a login with your friend.

✅ **Check:** the Tailscale icon appears in your system tray and says "Connected".

## Step A2 — Get this repo

Go to the repo page → green **Code** button → **Download ZIP**.

> ⚠️ **Before extracting, unblock the ZIP.** Right-click the downloaded file →
> **Properties** → tick **Unblock** at the bottom → OK.
>
> Windows marks anything downloaded from the internet, and that mark is copied
> to every extracted file. Skip this and the `.bat` files trigger a
> "Windows protected your PC" warning. Unblocking the ZIP first fixes all of
> them in one go.

Now extract to a permanent folder like `C:\p2pgpu`.

> ⚠️ Don't leave it in Downloads. The tool keeps a workspace folder next to it.

**If the repo is private,** you need to be signed in to GitHub in your browser
and have accepted the collaborator invite first — otherwise the page 404s. See
[Getting access to a private repo](#getting-access-to-a-private-repo) below.

Or, if you have git:

```bash
git clone https://github.com/ShivaTeja05/P2P-GPU-prototype1.git
```

## Step A3 — Run the setup

**Windows:** double-click **`Setup.bat`**

**Linux:**

```bash
uv venv --python 3.12 && uv pip install -e .
```

The Windows script checks for Python, Tailscale, Docker Desktop and your NVIDIA
driver, and offers to install anything missing. **Nothing installs without you
saying yes.** You can re-run it any time.

If it installs Docker Desktop, you'll probably need to reboot. Then:

- Open Docker Desktop and finish the first-run wizard
- **Settings → General → tick "Use the WSL 2 based engine"**
- Leave it running (whale icon in the tray)
- Run `Setup.bat` again

> ⚠️ **The one mistake that breaks everything:** do NOT install Linux NVIDIA
> drivers inside WSL. Your normal Windows NVIDIA driver already gives Docker
> access to the GPU. Installing Linux drivers on top of it breaks things in
> confusing ways.

## Step A4 — Check it actually works

**Windows:** double-click **`Check-Setup.bat`**

**Linux:**

```bash
p2pgpu doctor
```

This prints your GPU, then genuinely launches a container and runs `nvidia-smi`
inside it. It's the real test, not a guess.

✅ **You want to see:** `GPU is visible inside Docker. Ready to share.`

Anything else, it tells you exactly what to fix. See
[Common problems](#common-problems) below.

## Step A5 — Give your friend access to this machine

This is the part people miss.

1. Go to **[login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)**
2. Find this PC in the list
3. Click the **⋯** menu → **Share...**
4. Send the invite link to your friend

They accept, and now they can reach **this one machine** — not your laptop, not
your phone, not anything else on your network.

✅ **Check:** the machine shows a "Shared" label in the admin console.

## Step A6 — Start sharing

**Windows:** double-click **`Share-My-GPU.bat`**

**Linux:**

```bash
p2pgpu share --hours 4
```

It asks how many hours, starts the session, and **copies the connection URL to
your clipboard.** Paste that to your friend.

> ⏳ The first run downloads several GB of PyTorch image. Once only — later
> shares start in seconds.

> 🔑 **Treat that URL like a password.** Anyone who has it can use your GPU.
> Send it privately.

### While it's running

| Windows | Linux | Does |
|---|---|---|
| `Sharing-Status.bat` | `p2pgpu status` | Is it running, how long is left |
| `Stop-Sharing.bat` | `p2pgpu stop` | Stop right now |

It stops **automatically** when the time runs out. That's deliberate — a share
you forgot about can't quietly hold your GPU for a week.

---

# Part B — I'm borrowing a GPU

You need: nothing special. Any Mac, Windows or Linux machine. No GPU, no Docker.

## Step B1 — Install Tailscale and sign in

Download from **[tailscale.com/download](https://tailscale.com/download)** and
sign in **with your own account**.

## Step B2 — Accept your friend's invite

Your friend sends you a Tailscale share link (their Step A5). Click it and
accept.

✅ **Check:** their GPU machine now appears in your
[Machines list](https://login.tailscale.com/admin/machines) with a `100.x.y.z`
address.

## Step B3 — Get this repo and set it up

**Windows:** Download ZIP → right-click it → Properties → tick **Unblock** →
extract → double-click **`Setup-Guest-Only.bat`** *(skips Docker — you don't
need it)*

If the repo is private, accept the collaborator invite first —
[details below](#getting-access-to-a-private-repo).

**Mac / Linux:**

```bash
git clone https://github.com/ShivaTeja05/P2P-GPU-prototype1.git && cd P2P-GPU-prototype1
```

```bash
uv venv --python 3.12 && uv pip install -e .
```

No `uv`? Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step B4 — Connect

Copy the URL your friend sent you, then:

**Windows:** double-click **`Connect-To-GPU.bat`** — it spots the URL on your
clipboard and offers to use it.

**Mac / Linux:**

```bash
p2pgpu attach "PASTE_THE_URL_HERE"
```

JupyterLab opens in your browser. **You're on their GPU.**

## Step B5 — Confirm you really have the GPU

In a notebook cell:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

✅ **You want:** `True` and something like `NVIDIA GeForce RTX 4080`.

That's it. Train something.

> 💾 **Save your work into `/workspace`.** That folder lives on your friend's
> machine and survives after the session ends. Anything saved elsewhere in the
> container is deleted when the share stops.

---

## Getting access to a private repo

If the owner keeps the repo private, they add you as a collaborator:

1. Repo page → **Settings** → **Collaborators** → **Add people**
2. Type your GitHub username or email → **Add**
3. You get an email invite — **accept it** (or go to
   [github.com/notifications](https://github.com/notifications))

Once accepted, **Download ZIP** works normally in a browser where you're signed
in. No tokens, no git, no command line.

If you prefer git on Windows, [GitHub Desktop](https://desktop.github.com/) is
the easiest option — it handles the login for you. Plain `git clone` over HTTPS
will ask for a Personal Access Token, not your password.

---

## Common problems

### "GPU is visible inside Docker" never appears (owner)

Work through these in order:

1. **Is Docker Desktop actually running?** Whale icon in the tray, not animating.
2. **Is the WSL2 engine on?** Docker Desktop → Settings → General.
3. **Is your Windows NVIDIA driver current?** Update it from GeForce Experience.
4. **Did you install Linux NVIDIA drivers in WSL?** Remove them. The Windows
   driver is the one that matters.

### "Docker does not list an NVIDIA runtime" (owner, Windows)

Usually **just a warning** on Windows — Docker Desktop reaches the GPU through
WSL2 without advertising a separate runtime. `Check-Setup.bat` runs a real test;
trust that result over the warning.

### "Could not reach ..." when connecting (borrower)

1. Is Tailscale connected on **both** machines?
2. Did you accept the share invite (Step B2)?
3. Ask your friend to run `Sharing-Status.bat` — the share may have expired.
4. `tailscale status` should list their machine.

### "Windows protected your PC" when double-clicking a .bat

Windows blocked it because it came from a downloaded ZIP. Best fix: delete the
extracted folder, right-click the original **ZIP** → Properties → **Unblock** →
extract again. Or click **More info → Run anyway** on the warning.

### "running scripts is disabled on this system" (Windows)

Use the **`.bat`** files, not the `.ps1` files. The `.bat` wrappers handle this
for that single run, and change nothing about your system.

### It worked yesterday, not today

The share expired. The owner runs `Share-My-GPU.bat` again — it makes a **new**
URL, and the old one stops working on purpose.

### Everything is slow

Run `tailscale status` on either machine. If it says `relay` instead of
`direct`, your routers wouldn't allow a direct connection and traffic is going
through a relay. Still private and encrypted, just slower.

Also remember: **moving data is limited by home upload speed, not GPU speed.**
Pushing a 5 GB dataset over a 20 Mbps upload takes ~35 minutes no matter how
fast the graphics card is.

---

## Is this safe? (worth 60 seconds)

**For the person lending the GPU:**

✅ Your files are safe — the session runs in a sealed container, and only the
`p2pgpu-workspace` folder is shared.
✅ Nothing is exposed to the internet. The connection only exists inside your
Tailscale tunnel. No ports opened, no router changes.
✅ It stops on a timer.

⚠️ But your friend **can run code** on your machine inside that container, and
container isolation is good rather than perfect.

**The rule: only share with people you'd lend your actual laptop to.**

Don't use this to sell GPU time to strangers — it isn't built for that. Full
detail in [docs/SECURITY.md](docs/SECURITY.md).

---

## Where to go next

- [docs/SETUP.md](docs/SETUP.md) — more detail, including Tailscale account options
- [docs/WINDOWS.md](docs/WINDOWS.md) — Windows specifics and troubleshooting
- [docs/SECURITY.md](docs/SECURITY.md) — the full threat model
- [README.md](README.md) — what this project is

**Stuck?** Open an issue with what you ran and what it printed. The output from
`Check-Setup.bat` / `p2pgpu doctor` is the single most useful thing to include.
