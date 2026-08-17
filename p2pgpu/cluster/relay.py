"""A door from inside the session container out to the coordinator.

AVERAGE mode only ever dials *outward*: both GPU containers connect to the
coordinator, and the coordinator connects to nobody. That is what makes it work
across NAT without any hole punching. But "outward" still has to reach a
Tailscale address, and on one common setup it does not:

    Linux host          container -> host netns -> tailscale0    works
    Docker Desktop/Mac  container -> host netns -> tailscale0    works
    Windows + WSL2      container -> WSL2 VM -> ???              often fails

On Windows the container's host is not Windows -- it is the WSL2 virtual
machine, and the Windows Tailscale interface does not live there. So a request
to 100.x.y.z leaves the container, reaches WSL2, finds no route for the tailnet,
and dies. The GPU machine itself is on the tailnet the whole time; the container
just cannot see it.

This relay is the fix, and it is deliberately dumb: listen on an address the
container can always reach, and shovel bytes to the coordinator over the host's
own Tailscale connection. The host is already on the tailnet, so it is only ever
forwarding traffic that was going to be allowed anyway -- this adds a hop, not a
privilege.

Run it on the GPU machine, then point the training script at *this machine's
own* Tailscale IP -- not the coordinator's, and not `host.docker.internal`.

The Docker alias is the obvious choice and it is the wrong one: measured inside
a real session container, `host.docker.internal` did not resolve at all, while
the host's own Tailscale address answered in 3 ms. That asymmetry -- a container
reaching its own host but not necessarily other machines -- is what this relay
stands in for.

Before reaching for the relay, check the coordinator's own machine: a firewall
there blocks every node and produces exactly the same "cannot reach" symptom
from inside the container. `reachable()` below is that check.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

# One sync round is a whole state_dict, so chunks want to be big enough that a
# few hundred MB does not become a million tiny relayed writes.
CHUNK = 256 * 1024


@dataclass
class RelayStats:
    connections: int = 0
    bytes_out: int = 0
    bytes_in: int = 0


class RelayError(RuntimeError):
    """The relay could not start or could not reach the coordinator."""


def parse_target(url: str) -> tuple[str, int]:
    """Pull host and port out of a coordinator URL.

    IPv6 literals arrive bracketed (`http://[fd7a::1]:8899`) and `urlparse`
    strips the brackets for us -- which is what asyncio wants anyway. Getting
    this wrong was already a bug once, in the share URL.
    """
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        raise RelayError(f"could not read a host out of {url!r}")
    if parsed.scheme == "https":
        raise RelayError(
            "the relay forwards raw TCP and cannot terminate TLS. "
            "Point it at the coordinator's plain http:// URL."
        )
    return parsed.hostname, parsed.port or 8899


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> int:
    total = 0
    try:
        while True:
            chunk = await reader.read(CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
            total += len(chunk)
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass
    return total


async def _serve(
    listen_host: str, listen_port: int, target_host: str, target_port: int,
    stats: RelayStats, on_event=None,
) -> None:
    def say(message: str) -> None:
        if on_event is not None:
            on_event(message)

    async def handle(client_reader, client_writer):
        stats.connections += 1
        peer = client_writer.get_extra_info("peername")
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(target_host, target_port), timeout=30
            )
        except (OSError, asyncio.TimeoutError) as exc:
            say(f"[relay] {peer} -> coordinator unreachable: {exc}")
            client_writer.close()
            return

        # Both directions concurrently: a sync round uploads hundreds of MB and
        # then downloads the average, and a half-duplex relay would deadlock the
        # moment the coordinator started answering before we finished sending.
        sent, received = await asyncio.gather(
            _pump(client_reader, up_writer),
            _pump(up_reader, client_writer),
        )
        stats.bytes_out += sent
        stats.bytes_in += received

    server = await asyncio.start_server(handle, listen_host, listen_port)
    say(f"[relay] {listen_host}:{listen_port} -> {target_host}:{target_port}")
    async with server:
        await server.serve_forever()


def serve(
    coordinator_url: str,
    listen_host: str = "0.0.0.0",
    listen_port: int = 8899,
    on_event=None,
) -> None:
    """Forward every connection on `listen_port` to the coordinator. Blocks."""
    target_host, target_port = parse_target(coordinator_url)
    stats = RelayStats()
    try:
        asyncio.run(
            _serve(listen_host, listen_port, target_host, target_port, stats, on_event)
        )
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        raise RelayError(
            f"could not listen on {listen_host}:{listen_port}: {exc}\n"
            "Something else may already hold that port -- try --listen-port."
        ) from exc


def reachable(coordinator_url: str, token: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Can this process reach the coordinator directly? Decides if a relay is needed."""
    import httpx

    from p2pgpu.common.config import AUTH_HEADER

    try:
        resp = httpx.get(
            f"{coordinator_url.rstrip('/')}/health",
            headers={AUTH_HEADER: token},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return False, str(exc)
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}"
    role = resp.json().get("role")
    if role != "coordinator":
        return False, f"something is listening, but it is a {role!r}, not a coordinator"
    return True, "ok"
