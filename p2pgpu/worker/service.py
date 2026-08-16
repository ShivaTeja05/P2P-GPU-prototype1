"""Run the agent in the background, starting automatically at login.

Without this, discovery only works if someone remembered to leave a terminal
open running `p2pgpu agent` -- which is exactly the kind of invisible
prerequisite that makes a tool feel broken. "I can't see your GPU" and "you
forgot to start the agent" look identical from the other machine.

Every desktop OS already has a service manager for this, so we register with
the native one rather than inventing a daemon:

  macOS    launchd     a .plist in ~/Library/LaunchAgents, loaded by launchctl.
                       RunAtLoad starts it at login; KeepAlive restarts it if
                       it dies.
  Linux    systemd     a .service unit in ~/.config/systemd/user. 'enable'
                       makes it start at login; Restart=always covers crashes.
                       'loginctl enable-linger' additionally keeps it running
                       when nobody is logged in, which is what you want on a
                       headless GPU box.
  Windows  Task Sched  a scheduled task triggered ONLOGON. A true Windows
                       Service would need admin plus a wrapper like NSSM; a
                       user-level task needs neither and survives reboots.

All three are per-user, so none of this needs administrator rights.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "p2pgpu-agent"
MACOS_LABEL = "com.p2pgpu.agent"

LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{MACOS_LABEL}.plist"
SYSTEMD_UNIT = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"
LOG_DIR = Path.home() / ".p2pgpu" / "logs"


class ServiceError(RuntimeError):
    """Something went wrong registering the service. Message is user-facing."""


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def agent_command() -> list[str]:
    """How to invoke the agent, as an absolute command.

    Service managers do not inherit your shell's PATH, so 'p2pgpu' alone will
    not resolve. The console script lives next to the interpreter that is
    running us; if it is missing (an editable install run oddly), fall back to
    invoking the module through the interpreter directly.
    """
    exe = Path(sys.executable)
    script = exe.parent / ("p2pgpu.exe" if os.name == "nt" else "p2pgpu")
    if script.exists():
        return [str(script), "agent"]
    return [str(exe), "-m", "p2pgpu.cli.main", "agent"]


# --------------------------------------------------------------------------
# macOS -- launchd
# --------------------------------------------------------------------------


def _macos_install() -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    args = "".join(f"    <string>{a}</string>\n" for a in agent_command())
    LAUNCHD_PLIST.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{MACOS_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}  </array>
  <key>RunAtLoad</key><true/>
  <!-- Restart on crashes, but not after a clean exit. A plain <true/> here
       turns "an agent is already running" into an endless restart loop. -->
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>{LOG_DIR / "agent.log"}</string>
  <key>StandardErrorPath</key><string>{LOG_DIR / "agent.err"}</string>
</dict>
</plist>
'''
    )
    # bootout first so reinstalling over a running copy is not an error.
    _run(["launchctl", "bootout", f"gui/{os.getuid()}/{MACOS_LABEL}"])
    result = _run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(LAUNCHD_PLIST)])
    if result.returncode != 0:
        # Older macOS releases only understand the load/unload verbs.
        result = _run(["launchctl", "load", "-w", str(LAUNCHD_PLIST)])
        if result.returncode != 0:
            raise ServiceError((result.stderr or result.stdout).strip())
    return f"launchd agent registered at {LAUNCHD_PLIST}"


def _macos_uninstall() -> str:
    _run(["launchctl", "bootout", f"gui/{os.getuid()}/{MACOS_LABEL}"])
    _run(["launchctl", "unload", "-w", str(LAUNCHD_PLIST)])
    LAUNCHD_PLIST.unlink(missing_ok=True)
    return "launchd agent removed"


def _macos_running() -> bool:
    result = _run(["launchctl", "list"])
    return MACOS_LABEL in result.stdout


# --------------------------------------------------------------------------
# Linux -- systemd (user scope)
# --------------------------------------------------------------------------


def _linux_install() -> str:
    if not shutil.which("systemctl"):
        raise ServiceError("systemd not found. Start the agent manually with 'p2pgpu agent'.")
    SYSTEMD_UNIT.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(agent_command())
    SYSTEMD_UNIT.write_text(
        f"""[Unit]
Description=p2pgpu agent
After=network-online.target

[Service]
Type=simple
ExecStart={command}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""
    )
    _run(["systemctl", "--user", "daemon-reload"])
    result = _run(["systemctl", "--user", "enable", "--now", SERVICE_NAME])
    if result.returncode != 0:
        raise ServiceError((result.stderr or result.stdout).strip())

    # Without lingering, a user unit stops when the last session ends -- which
    # is precisely when a headless GPU host needs to stay reachable.
    if shutil.which("loginctl"):
        _run(["loginctl", "enable-linger", os.getlogin() if hasattr(os, "getlogin") else ""])
    return f"systemd user service enabled ({SYSTEMD_UNIT})"


def _linux_uninstall() -> str:
    _run(["systemctl", "--user", "disable", "--now", SERVICE_NAME])
    SYSTEMD_UNIT.unlink(missing_ok=True)
    _run(["systemctl", "--user", "daemon-reload"])
    return "systemd user service removed"


def _linux_running() -> bool:
    result = _run(["systemctl", "--user", "is-active", SERVICE_NAME])
    return result.stdout.strip() == "active"


# --------------------------------------------------------------------------
# Windows -- Task Scheduler
# --------------------------------------------------------------------------


def _windows_install() -> str:
    command = " ".join(f'"{a}"' if " " in a else a for a in agent_command())
    result = _run(
        [
            "schtasks", "/Create",
            "/TN", SERVICE_NAME,
            "/TR", command,
            "/SC", "ONLOGON",
            "/RL", "LIMITED",   # user-level: no administrator prompt
            "/F",               # overwrite an existing task
        ]
    )
    if result.returncode != 0:
        raise ServiceError((result.stderr or result.stdout).strip())
    _run(["schtasks", "/Run", "/TN", SERVICE_NAME])
    return f"scheduled task '{SERVICE_NAME}' created (runs at logon)"


def _windows_uninstall() -> str:
    _run(["schtasks", "/End", "/TN", SERVICE_NAME])
    result = _run(["schtasks", "/Delete", "/TN", SERVICE_NAME, "/F"])
    if result.returncode != 0:
        raise ServiceError((result.stderr or result.stdout).strip())
    return f"scheduled task '{SERVICE_NAME}' removed"


def _windows_running() -> bool:
    result = _run(["schtasks", "/Query", "/TN", SERVICE_NAME, "/FO", "LIST"])
    return "Running" in result.stdout


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_HANDLERS = {
    "Darwin": (_macos_install, _macos_uninstall, _macos_running),
    "Linux": (_linux_install, _linux_uninstall, _linux_running),
    "Windows": (_windows_install, _windows_uninstall, _windows_running),
}


def _handlers():
    try:
        return _HANDLERS[platform.system()]
    except KeyError as exc:
        raise ServiceError(
            f"No service support for {platform.system()}. "
            "Run 'p2pgpu agent' manually instead."
        ) from exc


def install() -> str:
    return _handlers()[0]()


def uninstall() -> str:
    return _handlers()[1]()


def is_running() -> bool:
    try:
        return _handlers()[2]()
    except ServiceError:
        return False


def manager_name() -> str:
    return {"Darwin": "launchd", "Linux": "systemd", "Windows": "Task Scheduler"}.get(
        platform.system(), "unsupported"
    )
