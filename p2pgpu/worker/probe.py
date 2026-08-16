"""Detect what compute this machine actually has.

Deliberately works without torch installed, because the Mac control node has no
reason to carry a 2 GB dependency just to list its peers. When torch is present
we prefer it -- it reports the memory the allocator can really reach, which is
what matters for placement.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import psutil

from p2pgpu import __version__
from p2pgpu.common.config import node_id
from p2pgpu.common.schema import Backend, GPUInfo, NodeCapabilities

MB = 1024 * 1024


def _nvidia_smi_gpus() -> list[GPUInfo]:
    """Query NVIDIA GPUs without needing torch. Returns [] if no driver."""
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,memory.total,memory.free,compute_cap,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, free, cap, driver = parts[:6]
        try:
            gpus.append(
                GPUInfo(
                    index=int(idx),
                    name=name,
                    vendor="nvidia",
                    total_memory_mb=int(float(total)),
                    free_memory_mb=int(float(free)),
                    compute_capability=cap,
                    driver_version=driver,
                )
            )
        except ValueError:
            continue
    return gpus


def _apple_gpu() -> list[GPUInfo]:
    """Apple Silicon: the GPU shares system RAM, so 'VRAM' is a policy number.

    Metal will not hand the whole 36 GB to one process. The recommended working
    set is roughly 75% of system memory, and that is the honest figure to
    advertise to a scheduler.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    total_ram_mb = psutil.virtual_memory().total // MB
    budget = int(total_ram_mb * 0.75)
    available = int(psutil.virtual_memory().available // MB * 0.75)
    return [
        GPUInfo(
            index=0,
            name=f"Apple {platform.processor() or 'Silicon'} (unified memory)",
            vendor="apple",
            total_memory_mb=budget,
            free_memory_mb=min(budget, available),
        )
    ]


def _cpu_model() -> str | None:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _torch_info() -> tuple[str | None, Backend | None, list[GPUInfo]]:
    try:
        import torch
    except ImportError:
        return None, None, []

    version = torch.__version__
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    vendor="nvidia",
                    total_memory_mb=total_b // MB,
                    free_memory_mb=free_b // MB,
                    compute_capability=f"{props.major}.{props.minor}",
                )
            )
        return version, "cuda", gpus

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return version, "mps", _apple_gpu()

    return version, "cpu", []


def probe() -> NodeCapabilities:
    """Build this node's advertised capability record."""
    torch_version, backend, gpus = _torch_info()

    # Fall back to vendor tools when torch is absent or saw nothing.
    if not gpus:
        gpus = _nvidia_smi_gpus()
        if gpus:
            backend = "cuda"
    if not gpus:
        gpus = _apple_gpu()
        if gpus:
            backend = backend or "mps"

    vm = psutil.virtual_memory()
    return NodeCapabilities(
        node_id=node_id(),
        hostname=platform.node(),
        platform=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_cores=psutil.cpu_count(logical=True),
        ram_total_mb=vm.total // MB,
        ram_free_mb=vm.available // MB,
        backend=backend or "cpu",
        gpus=gpus,
        torch_version=torch_version,
        agent_version=__version__,
    )
