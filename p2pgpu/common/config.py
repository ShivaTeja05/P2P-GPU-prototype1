"""Node identity and the shared cluster secret.

The overlay network (Tailscale/WireGuard) already encrypts and authenticates at
the device level. The token here is defence in depth: it stops anything that
reaches the port -- a misconfigured ACL, a second process, a friend's roommate
on the same tailnet -- from driving the GPU.
"""

from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("P2PGPU_HOME", Path.home() / ".p2pgpu"))
NODE_ID_FILE = CONFIG_DIR / "node_id"
TOKEN_FILE = CONFIG_DIR / "cluster_token"

TOKEN_ENV = "P2PGPU_TOKEN"
AUTH_HEADER = "x-p2pgpu-token"


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)


def node_id() -> str:
    """Stable per-machine identifier, generated once and persisted."""
    _ensure_dir()
    if NODE_ID_FILE.exists():
        value = NODE_ID_FILE.read_text().strip()
        if value:
            return value
    value = uuid.uuid4().hex[:12]
    NODE_ID_FILE.write_text(value + "\n")
    return value


def cluster_token() -> str | None:
    """Shared secret. Env var wins so containers/CI can inject it."""
    env = os.environ.get(TOKEN_ENV)
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        value = TOKEN_FILE.read_text().strip()
        if value:
            return value
    return None


def write_cluster_token(token: str) -> Path:
    _ensure_dir()
    TOKEN_FILE.write_text(token.strip() + "\n")
    TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE


def generate_token() -> str:
    return secrets.token_urlsafe(32)
