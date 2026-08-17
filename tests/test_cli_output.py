"""IPv6 addresses must survive being printed.

Yesterday's fix bracketed IPv6 literals when *building* the share URL and the
docker -p flag. It did not cover *displaying* them, and rich parses square
brackets as markup tags -- so a correctly bracketed URL renders as
`http://:8888/lab?token=...`, with the host silently gone.

That is the worst shape a bug can take here: the URL in the panel is the exact
string the friend copies, so the failure lands on someone who has no way to
diagnose it and did nothing wrong.
"""

from __future__ import annotations

import re

import pytest
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from p2pgpu.worker.share import bracket_host

V6 = "fd7a:115c:a1e0::e801:f4bb"
V6_URL = f"http://[{V6}]:8888/lab?token=SECRET123"
V6_SSH = f"ssh -p 2222 root@[{V6}]"


def render(renderable) -> str:
    console = Console(width=200, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


# --- the bug itself --------------------------------------------------------


def test_unescaped_ipv6_really_does_vanish():
    """Pins the behaviour the escaping exists to defend against."""
    assert V6 not in render(f"[red]Could not reach {V6_URL}[/]")


def test_escaped_ipv6_survives_a_plain_print():
    assert V6 in render(f"[red]Could not reach {escape(V6_URL)}[/]")


def test_escaped_ipv6_survives_a_panel():
    """The share panel is the string the friend copies."""
    out = render(Panel(escape(V6_URL), title="send this to your friend"))
    assert V6 in out
    assert "SECRET123" in out


def test_escaped_ipv6_survives_a_table():
    table = Table(show_header=False)
    table.add_row("url", escape(V6_URL))
    table.add_row("ssh", escape(V6_SSH))
    out = render(table)
    assert out.count(V6) == 2


def test_escaped_ssh_command_keeps_its_host():
    out = render(f"[dim]ssh available: {escape(V6_SSH)}[/]")
    assert f"root@[{V6}]" in out


# --- the call sites --------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "session.url",
        "session.ssh_command",
        "target.ssh_command",
    ],
)
def test_no_call_site_prints_a_url_unescaped(source):
    """Grep the CLI so a new print of a user-facing address cannot regress this."""
    from pathlib import Path

    cli = Path(__file__).resolve().parent.parent / "p2pgpu" / "cli" / "main.py"
    text = cli.read_text()

    # Every occurrence inside a console.print / Panel / add_row must be wrapped.
    unescaped = [
        line.strip()
        for line in text.splitlines()
        if source in line
        and re.search(r"console\.print|Panel\(|add_row", line)
        and "escape(" not in line
    ]
    assert not unescaped, f"{source} printed without escape(): {unescaped}"


def test_bracket_host_still_brackets():
    """The construction half of the fix, which the display half depends on."""
    assert bracket_host(V6) == f"[{V6}]"
    assert bracket_host("100.65.244.36") == "100.65.244.36"
    assert bracket_host(f"[{V6}]") == f"[{V6}]"  # idempotent


def test_ipv4_is_unaffected_by_escaping():
    url = "http://100.65.244.36:8888/lab?token=abc"
    assert url in render(escape(url))
