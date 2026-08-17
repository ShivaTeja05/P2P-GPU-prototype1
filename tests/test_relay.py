"""The relay is what lets a container on Windows reach the coordinator.

Worth real tests rather than a read-through, because its two failure modes are
both silent: a half-duplex relay deadlocks only once a payload is large enough
that the far end answers before the near end finishes sending, and a mis-parsed
IPv6 target connects to the wrong place rather than erroring.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from p2pgpu.cluster.relay import RelayError, parse_target, serve


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- target parsing --------------------------------------------------------


def test_plain_host_and_port():
    assert parse_target("http://100.65.244.36:8899") == ("100.65.244.36", 8899)


def test_port_defaults_to_the_coordinators():
    assert parse_target("http://100.65.244.36") == ("100.65.244.36", 8899)


def test_bare_host_without_a_scheme_is_accepted():
    assert parse_target("100.65.244.36:8899") == ("100.65.244.36", 8899)


def test_ipv6_literal_loses_its_brackets():
    """asyncio wants the address unbracketed; getting this wrong bit us before."""
    host, port = parse_target("http://[fd7a:115c:a1e0::1]:8899")
    assert host == "fd7a:115c:a1e0::1"
    assert port == 8899


def test_https_is_refused_with_a_reason():
    with pytest.raises(RelayError, match="TLS"):
        parse_target("https://coordinator.example.com:8899")


def test_garbage_is_refused():
    with pytest.raises(RelayError, match="could not read a host"):
        parse_target("http://:8899")


# --- actually relaying bytes ----------------------------------------------


@pytest.fixture()
def echo_server():
    """Stands in for the coordinator: echoes everything back."""
    port = _free_port()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(8)

    stop = threading.Event()

    def run():
        server.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=_echo, args=(conn,), daemon=True).start()

    def _echo(conn):
        with conn:
            while True:
                try:
                    data = conn.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                conn.sendall(data)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    yield port
    stop.set()
    server.close()
    thread.join(timeout=5)


@pytest.fixture()
def relay(echo_server):
    """A relay pointed at the echo server, on its own port."""
    listen_port = _free_port()
    thread = threading.Thread(
        target=serve,
        kwargs={
            "coordinator_url": f"http://127.0.0.1:{echo_server}",
            "listen_host": "127.0.0.1",
            "listen_port": listen_port,
        },
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", listen_port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError("relay did not start")
    return listen_port


def test_bytes_survive_the_round_trip(relay):
    with socket.create_connection(("127.0.0.1", relay), timeout=10) as sock:
        sock.sendall(b"hello coordinator")
        assert sock.recv(1024) == b"hello coordinator"


def test_a_large_payload_does_not_deadlock(relay):
    """The real case: a state_dict big enough that the far end replies mid-send.

    A relay that finished reading the client before it started writing back
    would hang here rather than fail, which is the worst kind of bug to hit
    twenty minutes into a training round.
    """
    payload = bytes(range(256)) * 8192  # 2 MB, well past any socket buffer
    received = bytearray()

    with socket.create_connection(("127.0.0.1", relay), timeout=30) as sock:
        sender = threading.Thread(target=sock.sendall, args=(payload,), daemon=True)
        sender.start()
        while len(received) < len(payload):
            chunk = sock.recv(65536)
            if not chunk:
                break
            received.extend(chunk)
        sender.join(timeout=10)

    assert bytes(received) == payload


def test_several_connections_are_served(relay):
    """Two GPU nodes sync at the same time; one at a time would serialise them."""
    results = []

    def talk(tag):
        with socket.create_connection(("127.0.0.1", relay), timeout=10) as sock:
            sock.sendall(tag)
            results.append(sock.recv(1024))

    threads = [threading.Thread(target=talk, args=(f"node-{i}".encode(),)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert sorted(results) == [b"node-0", b"node-1", b"node-2", b"node-3"]


def test_a_dead_coordinator_closes_the_client_rather_than_hanging(echo_server):
    """If the far side is gone the client must find out, not wait forever."""
    listen_port = _free_port()
    dead_port = _free_port()  # nothing listening here
    threading.Thread(
        target=serve,
        kwargs={
            "coordinator_url": f"http://127.0.0.1:{dead_port}",
            "listen_host": "127.0.0.1",
            "listen_port": listen_port,
        },
        daemon=True,
    ).start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", listen_port), timeout=0.5) as sock:
                sock.sendall(b"anyone there?")
                assert sock.recv(1024) == b""  # closed, not hung
                return
        except (OSError, AssertionError):
            time.sleep(0.1)
    pytest.fail("relay never came up, or never closed the doomed connection")
