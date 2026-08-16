"""Image selection across the GPU generations people actually own.

These are the cases that produce confusing runtime errors if we get them wrong,
so each one is pinned rather than left to a heuristic.
"""

from __future__ import annotations

import pytest

from p2pgpu.worker.gpu_compat import (
    GpuProfile,
    compatibility_notes,
    detect_gpus,
    format_gpu_flag,
    parse_driver,
    recommend_image,
)


def gpu(cc: str, driver: str = "550.54.14", index: int = 0, name: str = "GPU") -> GpuProfile:
    return GpuProfile(
        index=index, name=name, compute_capability=cc, vram_mb=16384, driver_version=driver
    )


# --- driver parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("550.54.14", (550, 54, 14)),
        ("525.60.13", (525, 60, 13)),
        ("470.82", (470, 82, 0)),
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_parse_driver(raw, expected):
    assert parse_driver(raw) == expected


# --- architecture naming ---------------------------------------------------


@pytest.mark.parametrize(
    "cc,arch",
    [
        ("6.1", "Pascal"),        # GTX 1080
        ("7.5", "Turing"),        # RTX 2080
        ("8.6", "Ampere"),        # RTX 3090
        ("8.9", "Ada Lovelace"),  # RTX 4080
        ("12.0", "Blackwell"),    # RTX 5090
    ],
)
def test_architecture_naming(cc, arch):
    assert gpu(cc).architecture == arch


# --- image selection -------------------------------------------------------


def test_rtx_40_series_gets_modern_cuda12():
    image, _ = recommend_image([gpu("8.9", name="RTX 4080")])
    assert "cuda12.4" in image


def test_rtx_50_series_requires_cuda128_and_torch27():
    """Blackwell is sm_120; older PyTorch has no kernels for it at all."""
    image, _ = recommend_image([gpu("12.0", driver="570.86.10", name="RTX 5090")])
    assert "cuda12.8" in image
    assert "2.7" in image


def test_old_driver_falls_back_to_cuda11():
    """CUDA 12.x needs driver >= 525.60.13; below that a 12.x image fails."""
    image, reason = recommend_image([gpu("8.6", driver="470.82.00", name="RTX 3060")])
    assert "cuda11.8" in image
    assert "driver" in reason.lower()


def test_driver_exactly_at_cuda12_floor_is_accepted():
    image, _ = recommend_image([gpu("8.6", driver="525.60.13")])
    assert "cuda12" in image


def test_pascal_card_gets_cuda11_line():
    image, _ = recommend_image([gpu("6.1", driver="470.82.00", name="GTX 1080")])
    assert "cuda11.8" in image


def test_mixed_generations_with_blackwell_forces_newest_image():
    """One image serves every shared GPU, and only 12.8 has sm_120 kernels."""
    image, reason = recommend_image(
        [gpu("8.9", index=0, name="RTX 4080"), gpu("12.0", index=1, name="RTX 5090")]
    )
    assert "cuda12.8" in image
    assert "blackwell" in reason.lower()


def test_mixed_older_generations_target_the_weakest_card():
    image, _ = recommend_image(
        [gpu("8.9", index=0, name="RTX 4080"), gpu("6.1", index=1, name="GTX 1080")]
    )
    assert "cuda11.8" in image


def test_no_gpu_returns_a_usable_default_rather_than_crashing():
    image, reason = recommend_image([])
    assert image
    assert "no gpu" in reason.lower()


# --- warnings --------------------------------------------------------------


def test_old_driver_is_flagged():
    notes = " ".join(compatibility_notes([gpu("8.6", driver="470.82.00")]))
    assert "525.60.13" in notes


def test_pascal_is_flagged_as_limited_not_broken():
    notes = " ".join(compatibility_notes([gpu("6.1", name="GTX 1080")])).lower()
    assert "pascal" in notes


def test_mixed_generations_are_flagged():
    notes = " ".join(compatibility_notes([gpu("8.9", index=0), gpu("7.5", index=1)])).lower()
    assert "different generations" in notes


def test_modern_card_with_modern_driver_has_nothing_to_warn_about():
    assert compatibility_notes([gpu("8.9", driver="550.54.14")]) == []


# --- docker --gpus flag ----------------------------------------------------


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("all", "all"),
        ("", "all"),
        ("0", '"device=0"'),
        ("0,1", '"device=0,1"'),
        (" 1 ", '"device=1"'),
    ],
)
def test_gpu_flag_formatting(selection, expected):
    assert format_gpu_flag(selection) == expected


def test_gpu_flag_rejects_nonsense():
    """A typo must fail loudly, not silently hand over every GPU."""
    with pytest.raises(ValueError):
        format_gpu_flag("gpu0")


def test_detect_gpus_is_safe_without_nvidia_smi():
    assert isinstance(detect_gpus(), list)
