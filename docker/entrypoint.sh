#!/usr/bin/env bash
# Session entrypoint for the p2pgpu image.
#
# Everything here is configured by environment variables rather than by
# generating a shell command on the host. That was the previous design, and it
# meant an SSH public key got interpolated into a string that a shell would
# later evaluate -- one stray quote away from running arbitrary code on the
# owner's machine. Passing values as env vars removes that class of bug
# entirely.
#
#   P2PGPU_HOURS       how long before the session self-terminates (default 4)
#   P2PGPU_SSH_KEY     authorised public key; empty disables SSH
#   P2PGPU_PORT        notebook port (default 8888)
#   JUPYTER_TOKEN      notebook token (read by jupyter directly)

set -euo pipefail

HOURS="${P2PGPU_HOURS:-4}"
PORT="${P2PGPU_PORT:-8888}"
SECONDS_LEFT=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 }')

echo "[p2pgpu] session starting"
echo "[p2pgpu] expires in ${HOURS}h"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | sed 's/^/[p2pgpu] gpu: /' || true
else
    echo "[p2pgpu] no GPU visible in this container"
fi

if [ -n "${P2PGPU_SSH_KEY:-}" ]; then
    mkdir -p /root/.ssh
    # printf '%s' with the value already in a variable: never re-parsed as code.
    printf '%s\n' "$P2PGPU_SSH_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    if /usr/sbin/sshd; then
        echo "[p2pgpu] sshd ready on port 22"
    else
        echo "[p2pgpu] WARNING: sshd failed to start; notebook still works"
    fi
fi

echo "[p2pgpu] starting notebook on port ${PORT}"

# exec so the notebook is PID 1 and receives docker stop cleanly. timeout gives
# the hard expiry -- no daemon on the host, nothing to leak if the app crashes.
exec timeout "${SECONDS_LEFT}s" jupyter lab \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --no-browser \
    --allow-root \
    --ServerApp.root_dir=/workspace
