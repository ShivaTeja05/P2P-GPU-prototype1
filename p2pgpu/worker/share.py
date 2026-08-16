"""Share this machine's GPU with one trusted person, for a fixed time.

The model is the same one vast.ai uses for its hosts: the GPU is handed to a
Docker container via the NVIDIA Container Toolkit, and the guest gets the
container -- not the machine. Two things differ here:

  * There is no marketplace and no billing. One friend, one GPU.
  * Reachability comes from the Tailscale overlay instead of forwarded ports,
    so nothing is exposed to the public internet.

The share always expires. A forgotten container that quietly holds someone's
GPU hostage is the failure mode most likely to end the friendship, so the
timeout is not optional.
"""

from __future__ import annotations

import json
import platform
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from p2pgpu.common.config import CONFIG_DIR
from p2pgpu.worker import gpu_compat

CONTAINER_NAME = "p2pgpu-share"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DEFAULT_PORT = 8888
SESSION_FILE = CONFIG_DIR / "share_session.json"

# Mounted at /workspace in the container. Deliberately a dedicated directory:
# the guest needs somewhere to leave checkpoints, and it must not be $HOME.
WORKSPACE = Path.home() / "p2pgpu-workspace"


DEFAULT_SSH_PORT = 2222

# A public key is one line: type, base64 blob, optional comment. Validated
# before it goes anywhere near a shell command -- this string is interpolated
# into the container's startup script, so a quote or newline in it would be a
# command injection on the owner's machine.
SSH_KEY_RE = re.compile(
    r"^(ssh-ed25519|ssh-rsa|ssh-dss|ecdsa-sha2-[a-z0-9-]+|sk-[a-z0-9@.-]+)"
    r"\s+[A-Za-z0-9+/]+={0,3}(\s+\S.*)?$"
)


class ShareError(RuntimeError):
    """Something in the host setup is not ready. Message is user-facing."""


def validate_ssh_key(key: str) -> str:
    """Return a cleaned single-line public key, or raise ShareError."""
    cleaned = " ".join(key.strip().split())
    if not cleaned:
        raise ShareError("SSH key is empty.")
    if "\n" in key.strip() or "\r" in key:
        raise ShareError("SSH key must be a single line.")
    if "'" in cleaned or '"' in cleaned or "\\" in cleaned:
        raise ShareError("SSH key contains quoting characters; that is not a valid key.")
    if cleaned.startswith("-----BEGIN"):
        raise ShareError(
            "That looks like a PRIVATE key. Send the .pub file instead -- never "
            "share a private key with anyone."
        )
    if not SSH_KEY_RE.match(cleaned):
        raise ShareError(
            "Not a valid SSH public key. Expected something starting with "
            "'ssh-ed25519' or 'ssh-rsa'. Ask them to run 'p2pgpu mykey'."
        )
    return cleaned


@dataclass
class ShareSession:
    url: str
    token: str
    bind_ip: str
    port: int
    hours: float
    started_at: float
    image: str
    workspace: str
    ssh_port: int | None = None
    ssh_command: str | None = None

    @property
    def expires_at(self) -> float:
        return self.started_at + self.hours * 3600

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.time())


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------
# Host preflight
# --------------------------------------------------------------------------


# The Windows installer does not put tailscale.exe on PATH, and the macOS app
# bundles its CLI inside the .app, so 'which' alone finds it on Linux only.
TAILSCALE_FALLBACKS = [
    r"C:\Program Files\Tailscale\tailscale.exe",
    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/bin/tailscale",
    "/usr/local/bin/tailscale",
]


def tailscale_exe() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in TAILSCALE_FALLBACKS:
        if Path(candidate).exists():
            return candidate
    return None


def tailscale_ip() -> str | None:
    """This machine's stable overlay address, or None if Tailscale is absent."""
    exe = tailscale_exe()
    if exe is None:
        return None
    try:
        result = _run([exe, "ip", "-4"], timeout=15)
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def is_ipv6(address: str) -> bool:
    """A bare IPv6 literal has multiple colons; IPv4 and hostnames have none."""
    return address.count(":") > 1


def bracket_host(address: str) -> str:
    """Wrap an IPv6 literal in brackets so it can sit next to a :port.

    Both Docker's -p flag and URL syntax parse on colons, so a bare
    'fd7a:115c::1' followed by ':8888' is ambiguous. RFC 3986 brackets resolve
    it, and Docker follows the same convention.
    """
    if is_ipv6(address) and not address.startswith("["):
        return f"[{address}]"
    return address


def docker_mount_path(path: Path) -> str:
    """Render a host path for 'docker -v'.

    Docker Desktop accepts Windows paths, but a backslash form next to the
    colon separator is easy to get wrong. Forward slashes work on every
    platform, so normalise rather than special-casing at the call site.
    """
    return str(path).replace("\\", "/")


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return _run(["docker", "info"], timeout=30).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def nvidia_runtime_registered() -> bool:
    """Cheap check that the NVIDIA Container Toolkit is wired into Docker.

    Cheap because it reads Docker's config rather than pulling a CUDA image.
    A false here is the single most common setup failure.
    """
    try:
        result = _run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    if result.returncode != 0:
        return False
    try:
        return "nvidia" in json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return "nvidia" in result.stdout


def verify_gpu_passthrough(image: str = "nvidia/cuda:12.4.0-base-ubuntu22.04") -> tuple[bool, str]:
    """Actually run nvidia-smi inside a container. Slow (pulls ~200 MB) but definitive."""
    try:
        result = _run(["docker", "run", "--rm", "--gpus", "all", image, "nvidia-smi"], timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timed out pulling or running the CUDA test image"
    except OSError as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, (result.stderr or result.stdout).strip()


def preflight(
    require_tailscale: bool = True,
    require_gpu: bool = True,
) -> tuple[list[str], list[str]]:
    """Check the host. Returns (blocking problems, non-blocking warnings).

    require_gpu is False when the caller asked for '--gpu none', where demanding
    an NVIDIA runtime would be nonsense.
    """
    problems: list[str] = []
    warnings: list[str] = []
    on_windows = platform.system() == "Windows"

    if not docker_available():
        problems.append(
            "Docker is not installed or not running. Install Docker Engine "
            "(Linux) or Docker Desktop with the WSL2 backend (Windows), and "
            "make sure it is actually started."
        )
    elif require_gpu and not nvidia_runtime_registered():
        message = (
            "Docker does not list an NVIDIA runtime. On Linux, install the "
            "NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/"
            "cloud-native/container-toolkit/latest/install-guide.html"
        )
        if on_windows:
            # Docker Desktop reaches the GPU through WSL2 paravirtualisation and
            # does not always advertise an 'nvidia' runtime even when --gpus
            # works. Blocking here would reject a perfectly good setup, so this
            # is a warning and the real passthrough test in 'doctor' decides.
            warnings.append(
                "Docker does not list an NVIDIA runtime. On Windows this is "
                "often fine -- Docker Desktop reaches the GPU through WSL2. "
                "Run 'p2pgpu doctor' to test it for real."
            )
        else:
            problems.append(message)

    if require_tailscale and tailscale_ip() is None:
        problems.append(
            "No Tailscale address. Install from https://tailscale.com/download, "
            "run 'tailscale up', or pass --bind-ip to use a LAN address."
        )
    return problems, warnings


# --------------------------------------------------------------------------
# Container lifecycle
# --------------------------------------------------------------------------


def container_running() -> bool:
    try:
        result = _run(
            ["docker", "ps", "--filter", f"name=^{CONTAINER_NAME}$", "--format", "{{.Names}}"],
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return CONTAINER_NAME in result.stdout


def load_session() -> ShareSession | None:
    if not SESSION_FILE.exists():
        return None
    try:
        return ShareSession(**json.loads(SESSION_FILE.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _save_session(session: ShareSession) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    SESSION_FILE.write_text(json.dumps(asdict(session), indent=2))
    SESSION_FILE.chmod(0o600)


def resolve_image(explicit: str | None, gpu_selection: str = "all") -> tuple[str, str]:
    """Pick the container image for whatever GPU this host actually has.

    An explicit --image always wins; otherwise we match the card generation and
    driver version, because one hardcoded CUDA version cannot serve a GTX 1080
    and an RTX 5090 at the same time.
    """
    if explicit:
        return explicit, "explicitly requested"

    profiles = gpu_compat.detect_gpus()
    if gpu_selection not in ("", "all"):
        wanted = {int(part) for part in gpu_selection.split(",") if part.strip().isdigit()}
        profiles = [p for p in profiles if p.index in wanted] or profiles
    return gpu_compat.recommend_image(profiles)


def image_present(image: str) -> bool:
    try:
        return _run(["docker", "image", "inspect", image], timeout=60).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def pull_image(image: str, timeout_s: int = 7200) -> None:
    """Fetch the image, streaming docker's own progress to the terminal.

    Pulling is deliberately separate from 'docker run'. Folding it into the run
    meant one timeout had to cover both, and the PyTorch CUDA images are 7-9 GB
    -- easily over 15 minutes on home internet. Worse, the pull is silent when
    the container is started detached, so the owner sees a frozen prompt with no
    indication that gigabytes are moving. Doing it here gives them docker's
    progress bars and a timeout sized for the actual job.
    """
    if image_present(image):
        return
    try:
        # No capture_output: docker's progress goes straight to the terminal.
        result = subprocess.run(["docker", "pull", image], timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise ShareError(
            f"Image pull exceeded {timeout_s // 60} minutes. The download may "
            f"still be running in the background -- try again shortly."
        ) from exc
    if result.returncode != 0:
        raise ShareError(f"Could not pull {image}. Check the image name and your connection.")


def start_share(
    hours: float = 4.0,
    image: str | None = None,
    port: int = DEFAULT_PORT,
    bind_ip: str | None = None,
    shm_size: str = "8g",
    gpus: str = "all",
    ssh_key: str | None = None,
    ssh_port: int = DEFAULT_SSH_PORT,
) -> ShareSession:
    """Launch the shared GPU container and return the session details."""
    if container_running():
        raise ShareError(
            f"A share is already running. Stop it first with 'p2pgpu stop'."
        )

    try:
        gpu_flag = gpu_compat.format_gpu_flag(gpus)
    except ValueError as exc:
        raise ShareError(str(exc)) from exc

    image, _reason = resolve_image(image, gpus)

    ip = bind_ip or tailscale_ip()
    if ip is None:
        raise ShareError("No bind address available. Start Tailscale or pass --bind-ip.")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    seconds = int(hours * 3600)

    # jupyterlab is installed at start rather than baked in, so the default
    # image stays a stock PyTorch one that most people already have cached.
    # 'timeout' makes the container self-terminate; combined with --rm that
    # means an expired share leaves nothing behind and needs no daemon.
    setup = (
        "exec 3>&1; { echo '[p2pgpu] installing jupyterlab...'; "
        "pip install --quiet --no-input jupyterlab; "
    )

    if ssh_key:
        key = validate_ssh_key(ssh_key)
        # Key-only auth. Root here is root *inside an unprivileged container*,
        # which is not root on the host -- but passwords are still disabled so a
        # leaked URL cannot become a shell.
        setup += (
            "echo '[p2pgpu] installing openssh-server (this takes a few minutes)...'; "
            "apt-get update -qq && apt-get install -y -qq openssh-server; "
            "mkdir -p /run/sshd /root/.ssh; "
            f"printf '%s\\n' '{key}' > /root/.ssh/authorized_keys; "
            "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
            "sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' "
            "/etc/ssh/sshd_config; "
            "sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' "
            "/etc/ssh/sshd_config; "
            "/usr/sbin/sshd && echo '[p2pgpu] sshd ready'; "
        )

    inner = (
        setup
        + "echo '[p2pgpu] starting notebook'; } 2>&1 | tee /tmp/p2pgpu-setup.log >&3; "
        + f"exec timeout {seconds}s jupyter lab "
        f"--ip=0.0.0.0 --port={port} --no-browser --allow-root "
        "--ServerApp.root_dir=/workspace"
    )

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        *(["--gpus", gpu_flag] if gpu_flag else []),
        "--shm-size", shm_size,          # PyTorch dataloaders die on the 64 MB default
        "-e", f"JUPYTER_TOKEN={token}",
        # Binding to the overlay IP, not 0.0.0.0, keeps this off the host's LAN
        # and off the public internet even if a router is misconfigured.
        "-p", f"{bracket_host(ip)}:{port}:{port}",
        *(["-p", f"{bracket_host(ip)}:{ssh_port}:22"] if ssh_key else []),
        "-v", f"{docker_mount_path(WORKSPACE)}:/workspace",
        "-w", "/workspace",
        image,
        "bash", "-lc", inner,
    ]

    pull_image(image)

    try:
        result = _run(cmd, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise ShareError("Timed out starting the container.") from exc

    if result.returncode != 0:
        raise ShareError(f"docker run failed:\n{(result.stderr or result.stdout).strip()}")

    session = ShareSession(
        url=f"http://{bracket_host(ip)}:{port}/lab?token={token}",
        token=token,
        bind_ip=ip,
        port=port,
        hours=hours,
        started_at=time.time(),
        image=image,
        workspace=str(WORKSPACE),
        ssh_port=ssh_port if ssh_key else None,
        ssh_command=(f"ssh -p {ssh_port} root@{ip}" if ssh_key else None),
    )
    _save_session(session)
    return session


def wait_until_ready(
    host: str,
    port: int,
    timeout_s: int = 900,
    poll_s: float = 3.0,
) -> bool:
    """Block until the notebook actually answers, or give up.

    The container installs jupyterlab (and optionally openssh-server) on start,
    which takes minutes on a first run. Handing the owner a URL before any of
    that finishes means their friend gets 'connection refused' and everyone
    assumes it is broken. So we wait and say so.
    """
    import socket

    deadline = time.monotonic() + timeout_s
    target = host.strip("[]")
    while time.monotonic() < deadline:
        if not container_running():
            return False
        try:
            with socket.create_connection((target, port), timeout=3) as sock:
                # Docker's proxy accepts the TCP connection whether or not
                # anything is listening inside, so a connect() alone proves
                # nothing. Send a request and require actual bytes back.
                sock.sendall(b"GET / HTTP/1.0\r\n\r\n")
                if sock.recv(16):
                    return True
        except OSError:
            pass
        time.sleep(poll_s)
    return False


def setup_log() -> str:
    """The container's own setup transcript, for when something fails."""
    try:
        result = _run(["docker", "exec", CONTAINER_NAME, "cat", "/tmp/p2pgpu-setup.log"], timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"could not read setup log: {exc}"
    return (result.stdout or result.stderr).strip()


def stop_share() -> bool:
    """Stop the share. Returns True if a container was actually stopped."""
    was_running = container_running()
    if was_running:
        _run(["docker", "stop", CONTAINER_NAME], timeout=60)
    SESSION_FILE.unlink(missing_ok=True)
    return was_running


def share_logs(lines: int = 50) -> str:
    try:
        result = _run(["docker", "logs", "--tail", str(lines), CONTAINER_NAME], timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"could not read logs: {exc}"
    return (result.stdout + result.stderr).strip()
