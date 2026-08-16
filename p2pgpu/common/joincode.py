"""One paste to join a cluster.

Setting up the second machine used to be eight steps, and the one people
actually missed was buried in a web console: install Tailscale, sign in through
a browser, open the admin console, find the machine, share it, send the invite,
accept it, then separately exchange a cluster token.

All of that is really just two secrets moving between two people. A join code
carries both:

  * a Tailscale auth key, so the machine joins the tailnet with no browser and
    no admin console -- and because both machines then sit on the *same*
    tailnet, node sharing stops being necessary at all
  * the p2pgpu cluster token, so the agents trust each other

Treat a join code exactly like a password. Anyone holding one can put a device
on your tailnet.
"""

from __future__ import annotations

import base64
import json
import platform
from dataclasses import dataclass

PREFIX = "p2pgpu1-"
CODE_VERSION = 1

# Tailscale auth keys look like tskey-auth-XXXX-YYYY (and tskey-client-... for
# OAuth). Checking the shape up front means a mistyped code fails here with a
# clear message rather than deep inside 'tailscale up'.
TS_KEY_PREFIXES = ("tskey-auth-", "tskey-client-", "tskey-")


class JoinCodeError(ValueError):
    """Bad or unreadable join code. Message is user-facing."""


@dataclass
class JoinCode:
    tailscale_authkey: str
    cluster_token: str
    issued_by: str = ""
    version: int = CODE_VERSION

    def encode(self) -> str:
        payload = {
            "v": self.version,
            "ts": self.tailscale_authkey,
            "tok": self.cluster_token,
            "by": self.issued_by,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode()
        # urlsafe + stripped padding so the code survives chat apps, which love
        # to mangle '+' and '/' and sometimes swallow trailing '='.
        return PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def validate_authkey(key: str) -> str:
    key = key.strip()
    if not key:
        raise JoinCodeError("Tailscale auth key is empty.")
    if not key.startswith(TS_KEY_PREFIXES):
        raise JoinCodeError(
            "That does not look like a Tailscale auth key. They start with "
            "'tskey-auth-'. Create one at "
            "https://login.tailscale.com/admin/settings/keys"
        )
    return key


def decode(code: str) -> JoinCode:
    """Parse a join code. Raises JoinCodeError with something a person can act on."""
    code = code.strip()
    # Chat apps and terminals love to wrap long strings.
    code = "".join(code.split())

    if not code:
        raise JoinCodeError("No join code given.")
    if not code.startswith(PREFIX):
        raise JoinCodeError(
            f"A join code starts with '{PREFIX}'. Check you copied the whole thing."
        )

    body = code[len(PREFIX):]
    padding = "=" * (-len(body) % 4)
    try:
        raw = base64.urlsafe_b64decode(body + padding)
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise JoinCodeError("Join code is corrupted -- copy it again, whole.") from exc

    if not isinstance(payload, dict):
        raise JoinCodeError("Join code is corrupted.")

    version = payload.get("v")
    if version != CODE_VERSION:
        raise JoinCodeError(
            f"This join code is version {version}, this app understands "
            f"{CODE_VERSION}. One of you needs to update."
        )

    authkey = str(payload.get("ts") or "")
    token = str(payload.get("tok") or "")
    if not authkey or not token:
        raise JoinCodeError("Join code is missing part of its contents.")

    return JoinCode(
        tailscale_authkey=validate_authkey(authkey),
        cluster_token=token,
        issued_by=str(payload.get("by") or ""),
        version=version,
    )


def build(authkey: str, cluster_token: str) -> JoinCode:
    return JoinCode(
        tailscale_authkey=validate_authkey(authkey),
        cluster_token=cluster_token.strip(),
        issued_by=platform.node(),
    )
