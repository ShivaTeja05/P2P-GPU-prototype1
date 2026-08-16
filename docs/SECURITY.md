# Security

Honest account of what sharing a GPU with this tool does and does not protect.
Read it before you run `p2pgpu share` on a machine you care about.

## The one-line version

**The GPU is shared. The machine is not — but the isolation is a container, not
a vault.** Share with people you would lend your laptop to.

## What protects the host

**Container isolation.** The guest gets a Docker container, not a shell on the
host. They cannot read the home directory, SSH keys, browser profiles, or any
other file on the machine.

**A single mounted directory.** Only `~/p2pgpu-workspace` is bind-mounted, at
`/workspace`. That is the sole path where guest and host filesystems meet. Don't
put anything sensitive in it.

**Overlay-only reachability.** The notebook port binds to the host's Tailscale
address specifically — `-p 100.x.y.z:8888:8888`, not `-p 8888:8888`. So it is
not reachable from the host's own home Wi-Fi, not from the public internet, and
not by anything without access to the tailnet. No router configuration is
involved, so there is no port-forwarding rule to leave open by mistake.

**Transport encryption.** Tailscale is WireGuard. Traffic between the two
machines is encrypted end to end.

**A random per-session token.** Each `share` mints a fresh 24-byte URL-safe
token. Old URLs stop working.

**A hard expiry.** The container runs the notebook under `timeout`, so it dies
on schedule with no daemon and no cron job. Combined with `--rm`, an expired
share leaves nothing running and nothing behind. This is deliberate: the failure
mode most likely to cause a real problem is a forgotten share quietly holding
someone's GPU for a week.

## What does not protect the host

**The guest runs arbitrary code.** A notebook is a code execution environment.
Anyone with the URL can run anything inside that container, with the GPU.

**Container escapes are real.** Docker isolation is good, not perfect. Kernel
vulnerabilities that let code break out of a container exist and are found
periodically. `--gpus all` also loads the NVIDIA driver stack into the
container, which is additional attack surface.

**No resource limits beyond the GPU.** The container can use as much CPU and
host RAM as it wants. It could fill the disk via `/workspace`. Add `--memory`
and `--cpus` to the Docker command if that matters to you.

**The token travels in the URL.** URLs end up in shell history and clipboards.
Treat a share URL as a password.

**Whoever holds the URL is the guest.** There is no identity check beyond the
token. If your friend forwards it, that person has the GPU.

## Threat model

This tool assumes **two people who already trust each other**. That assumption
is doing a lot of work, and it's what makes the design as simple as it is.

Note that this is the same tradeoff commercial GPU marketplaces make — they run
renters' code in Docker containers with GPU passthrough too. The difference is
they accept it with *strangers*, backed by billing, reputation and legal
recourse. Here, the safety comes from knowing the person.

**Do not use this to sell GPU time to strangers.** Making that safe needs
sandboxing, quotas, abuse handling and monitoring that this tool does not have.

## Reducing exposure

Ordered by how much they buy you:

1. **Share for hours, not days.** `--hours 2` beats `--hours 48`.
2. **Keep `~/p2pgpu-workspace` clean.** It is the one shared surface.
3. **Run it on a spare machine** if you have one. Best possible isolation.
4. **Add resource limits** — `--memory 32g --cpus 8` on the Docker command.
5. **`p2pgpu stop` when done** rather than waiting for expiry.
6. **Keep Docker and NVIDIA drivers updated.** Container escapes get patched.
7. **Check `p2pgpu status`** occasionally so nothing is running that you forgot.

## Reporting

Found a problem with this tool? Open an issue. It's a small project — no bounty,
but genuine fixes are welcome.
