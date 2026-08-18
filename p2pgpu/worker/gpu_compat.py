"""Work out which container image a given NVIDIA GPU can actually run.

Hardcoding one CUDA image breaks on real hardware in two directions:

  * Old driver, new image. A CUDA 12.x runtime needs driver >= 525.60.13 on
    Linux (>= 527.41 on Windows). Below that the container starts and then
    fails with an unhelpful CUDA error.
  * New card, old image. RTX 50-series is Blackwell / sm_120, and PyTorch only
    gained sm_120 kernels in 2.7 with CUDA 12.8. Anything older runs, sees the
    GPU, and then dies on the first kernel launch with "no kernel image is
    available for execution on the device".

Both failures look like a bug in this tool rather than a version mismatch, so
we detect and choose up front instead.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# CUDA 12.x minor-version compatibility floor (Linux). Windows is 527.41, but
# on Windows the GPU reaches the container through WSL2 and the driver that
# matters is the Windows one, which we cannot read from inside. 525 is the
# safe common floor to reason with.
CUDA12_MIN_DRIVER = (525, 60, 13)
CUDA11_MIN_DRIVER = (450, 80, 2)

# Compute capability -> (architecture, typical consumer cards).
ARCHITECTURES: list[tuple[float, str, str]] = [
    (6.0, "Pascal", "GTX 10-series"),
    (7.0, "Volta", "Titan V"),
    (7.5, "Turing", "RTX 20-series, GTX 16-series"),
    (8.0, "Ampere", "A100"),
    (8.6, "Ampere", "RTX 30-series"),
    (8.9, "Ada Lovelace", "RTX 40-series"),
    (9.0, "Hopper", "H100"),
    (10.0, "Blackwell", "B200"),
    (12.0, "Blackwell", "RTX 50-series"),
]

# The '-runtime' variants, not '-devel'. Devel carries nvcc and the CUDA
# headers -- roughly double the download -- and those are only needed to
# *compile* CUDA extensions, not to train. Anyone who needs them can pass
# --image. Halving a 7-9 GB first-run download matters more than the rare case.
#
# Ordered most-capable first. Each entry: (min_cc, min_driver, image, note).
IMAGE_MATRIX: list[tuple[float, tuple[int, int, int], str, str]] = [
    (
        12.0,
        CUDA12_MIN_DRIVER,
        "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
        "Blackwell (sm_120) needs PyTorch >= 2.7 built against CUDA 12.8",
    ),
    (
        7.0,
        CUDA12_MIN_DRIVER,
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        "modern default for Turing through Hopper",
    ),
    (
        6.0,
        CUDA11_MIN_DRIVER,
        "pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime",
        "older driver or Pascal card; last comfortable CUDA 11 line",
    ),
]

FALLBACK_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


@dataclass
class GpuProfile:
    index: int
    name: str
    compute_capability: str | None
    vram_mb: int
    driver_version: str | None

    @property
    def cc_float(self) -> float | None:
        if not self.compute_capability:
            return None
        try:
            return float(self.compute_capability)
        except ValueError:
            return None

    @property
    def architecture(self) -> str:
        cc = self.cc_float
        if cc is None:
            return "unknown"
        best = "unknown"
        for threshold, arch, _cards in ARCHITECTURES:
            if cc >= threshold:
                best = arch
        return best


def parse_driver(version: str | None) -> tuple[int, int, int] | None:
    """'550.54.14' -> (550, 54, 14). Tolerates two-part versions."""
    if not version:
        return None
    parts = version.strip().split(".")
    numbers: list[int] = []
    for part in parts[:3]:
        try:
            numbers.append(int(part))
        except ValueError:
            break
    if not numbers:
        return None
    while len(numbers) < 3:
        numbers.append(0)
    return (numbers[0], numbers[1], numbers[2])


def detect_gpus() -> list[GpuProfile]:
    """Read the host's GPUs via nvidia-smi. Empty list if there are none."""
    if not shutil.which("nvidia-smi"):
        return []
    query = "index,name,compute_cap,memory.total,driver_version"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []

    profiles: list[GpuProfile] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            profiles.append(
                GpuProfile(
                    index=int(parts[0]),
                    name=parts[1],
                    compute_capability=parts[2] or None,
                    vram_mb=int(float(parts[3])),
                    driver_version=parts[4] or None,
                )
            )
        except ValueError:
            continue
    return profiles


def recommend_image(profiles: list[GpuProfile]) -> tuple[str, str]:
    """Choose an image that every selected GPU can run. Returns (image, reason).

    Picks for the *least* capable card in the set, since one image has to serve
    them all, but still respects the floor imposed by the newest architecture.
    """
    if not profiles:
        return FALLBACK_IMAGE, "no GPU detected; using the default image"

    driver = parse_driver(profiles[0].driver_version)
    ccs = [p.cc_float for p in profiles if p.cc_float is not None]
    if not ccs:
        return FALLBACK_IMAGE, "could not read compute capability; using the default image"

    highest = max(ccs)
    lowest = min(ccs)

    # A Blackwell card in the set forces the newest image regardless of what
    # else is present -- older images have no sm_120 kernels at all.
    if highest >= 12.0:
        image = IMAGE_MATRIX[0][2]
        if lowest < 12.0:
            return image, (
                "mixed generations including Blackwell; using the CUDA 12.8 image, "
                "which is required for sm_120"
            )
        return image, IMAGE_MATRIX[0][3]

    for min_cc, min_driver, image, note in IMAGE_MATRIX:
        if lowest < min_cc:
            continue
        if driver is not None and driver < min_driver:
            continue
        return image, note

    # Driver too old for anything modern: fall back to the CUDA 11 line.
    return IMAGE_MATRIX[-1][2], "driver is old; falling back to the CUDA 11.8 image"


def compatibility_notes(profiles: list[GpuProfile]) -> list[str]:
    """Warnings a person should see before they try to share. Empty = fine."""
    notes: list[str] = []
    if not profiles:
        return ["No NVIDIA GPU detected by nvidia-smi."]

    driver = parse_driver(profiles[0].driver_version)
    if driver is None:
        notes.append("Could not read the driver version; image choice may be wrong.")
    elif driver < CUDA11_MIN_DRIVER:
        notes.append(
            f"Driver {profiles[0].driver_version} is very old (< 450.80.02). "
            "Update it before sharing."
        )
    elif driver < CUDA12_MIN_DRIVER:
        notes.append(
            f"Driver {profiles[0].driver_version} predates CUDA 12 "
            "(needs >= 525.60.13). Falling back to a CUDA 11.8 image; "
            "updating the driver unlocks newer PyTorch."
        )

    for profile in profiles:
        cc = profile.cc_float
        if cc is None:
            continue
        if cc < 6.0:
            notes.append(
                f"GPU {profile.index} ({profile.name}, cc {profile.compute_capability}) "
                "is older than Pascal and current PyTorch will not run on it."
            )
        elif cc < 7.0:
            notes.append(
                f"GPU {profile.index} ({profile.name}) is Pascal-era. It works, but "
                "expect no bf16 and no tensor cores."
            )

    distinct = {p.cc_float for p in profiles if p.cc_float is not None}
    if len(distinct) > 1:
        notes.append(
            "GPUs of different generations are present. One image must serve all "
            "of them -- consider --gpu <index> to share just one."
        )
    return notes


def format_gpu_flag(selection: str) -> str | None:
    """Turn a friendly selection into Docker's --gpus value.

    'all' -> all; '0' -> device 0; '0,1' -> devices 0 and 1.
    'none' -> None, meaning omit --gpus entirely (CPU-only container). That
    exists so the whole share/attach path can be exercised on a machine with no
    NVIDIA GPU -- a Mac, or CI.
    """
    selection = selection.strip().lower()
    if selection == "none":
        return None
    if not selection or selection == "all":
        return "all"
    indices = [part.strip() for part in selection.split(",") if part.strip()]
    if not all(part.isdigit() for part in indices):
        raise ValueError(
            f"invalid GPU selection: {selection!r} (use 'all', 'none', '0', or '0,1')"
        )
    return f'"device={",".join(indices)}"'
