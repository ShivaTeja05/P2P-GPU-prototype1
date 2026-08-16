"""Find GPUs on your tailnet without anyone pasting a URL.

Today's manual handoff -- owner starts a share, copies a URL, sends it over
chat, borrower pastes it -- exists only because the two machines have no way to
find each other. They do: Tailscale's local API already knows every peer, their
addresses and whether they are online.

So discovery is a local operation. No coordinator, no accounts, no server to
host or trust. The agent on each machine answers "am I sharing right now?" to
anyone holding the cluster token, and the borrower just asks everyone.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
from dataclasses import dataclass

import httpx

from p2pgpu.common.config import AUTH_HEADER

AGENT_PORT = 8777
PROBE_TIMEOUT_S = 4.0


@dataclass
class Peer:
    """A machine on the tailnet, as Tailscale sees it."""

    ip: str
    hostname: str
    os: str
    online: bool
    is_self: bool = False


@dataclass
class DiscoveredShare:
    """A peer that is running an agent, and what it currently offers."""

    peer: Peer
    agent_version: str
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    gpu_free_mb: int | None = None
    sharing: bool = False
    share_url: str | None = None
    ssh_command: str | None = None
    hours_left: float | None = None

    @property
    def label(self) -> str:
        gpu = self.gpu_name or "no GPU"
        if self.gpu_free_mb:
            gpu += f" ({self.gpu_free_mb / 1024:.1f} GB free)"
        return f"{self.peer.hostname} - {gpu}"


def _tailscale_exe() -> str | None:
    from p2pgpu.worker.share import tailscale_exe

    return tailscale_exe()


def list_peers(include_offline: bool = False) -> list[Peer]:
    """Every machine on this tailnet, from Tailscale's own local API."""
    exe = _tailscale_exe()
    if exe is None:
        return []
    try:
        result = subprocess.run(
            [exe, "status", "--json"], capture_output=True, text=True, timeout=20
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    peers: list[Peer] = []
    for entry in (data.get("Peer") or {}).values():
        ips = entry.get("TailscaleIPs") or []
        if not ips:
            continue
        online = bool(entry.get("Online"))
        if not online and not include_offline:
            continue
        peers.append(
            Peer(
                ip=ips[0],
                hostname=entry.get("HostName") or ips[0],
                os=entry.get("OS") or "?",
                online=online,
            )
        )
    return peers


def probe(peer: Peer, token: str | None, port: int = AGENT_PORT) -> DiscoveredShare | None:
    """Ask one peer whether it is running an agent, and what it is offering.

    Returns None for anything that is not a p2pgpu machine -- most peers on a
    tailnet are phones and laptops that have never heard of us.
    """
    base = f"http://{peer.ip}:{port}"
    headers = {AUTH_HEADER: token} if token else {}
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT_S) as client:
            health = client.get(f"{base}/health")
            if health.status_code != 200:
                return None
            found = DiscoveredShare(
                peer=peer,
                agent_version=health.json().get("agent_version", "?"),
            )

            caps = client.get(f"{base}/v1/capabilities", headers=headers)
            if caps.status_code == 200:
                payload = caps.json()
                gpus = payload.get("gpus") or []
                if gpus:
                    found.gpu_name = gpus[0].get("name")
                    found.gpu_vram_mb = gpus[0].get("total_memory_mb")
                    found.gpu_free_mb = gpus[0].get("free_memory_mb")

            share = client.get(f"{base}/v1/share", headers=headers)
            if share.status_code == 200:
                payload = share.json()
                found.sharing = bool(payload.get("sharing"))
                found.share_url = payload.get("url")
                found.ssh_command = payload.get("ssh_command")
                found.hours_left = payload.get("hours_left")
            return found
    except httpx.HTTPError:
        return None


def discover(token: str | None, port: int = AGENT_PORT) -> list[DiscoveredShare]:
    """Probe every online peer in parallel. Returns only p2pgpu machines.

    Parallel because a tailnet can hold dozens of devices and most will not
    answer -- serially that is a minute of timeouts.
    """
    peers = list_peers()
    if not peers:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(peers))) as pool:
        results = pool.map(lambda p: probe(p, token, port), peers)
    found = [r for r in results if r is not None]
    # Machines actively sharing float to the top; then most free VRAM.
    found.sort(key=lambda d: (not d.sharing, -(d.gpu_free_mb or 0)))
    return found
