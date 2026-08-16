# Running your own control plane

You do not have to depend on Tailscale's servers. The Tailscale *client* talks to
any compatible control plane, so self-hosting is a flag rather than a fork:

```bash
tailscale up --login-server=https://headscale.example.com --auth-key=...
```

p2pgpu passes that flag for you, and carries the URL inside the join code — so
switching control planes is invisible to whoever is joining. They paste a code
exactly as before and land on your server.

```bash
p2pgpu invite --login-server https://headscale.example.com
```

---

## What a control plane actually does

Worth understanding before you decide to build one, because "coordination
server" undersells it. Five jobs:

1. **Identity** — who is allowed on this network
2. **Key distribution** — every node's WireGuard public key and candidate
   endpoints, pushed to every other node
3. **NAT traversal coordination** — tells two peers about each other's
   addresses so they can hole-punch *simultaneously*, which is the only way it
   works
4. **Relay fallback** — when hole punching fails, traffic goes through relay
   servers instead
5. **Policy** — ACLs, tags, expiry

Jobs 1, 2, 3 and 5 are ordinary distributed-systems work. **Job 4 is the one
that will hurt you.** Roughly 10–20% of home-router pairs cannot hole-punch,
usually symmetric NAT on one side. Those connections need a relay, and a relay
is a real server carrying real bandwidth. One relay in one region means users on
the other side of the world get a slow, distant hop.

Tailscale runs a global DERP mesh for this. Reproducing it is not a coding
problem; it is a hosting bill.

---

## Three tiers, honestly

| | What you run | Cost | Effort | You control |
|---|---|---|---|---|
| **1. Tailscale** | nothing | free | none | nothing |
| **2. Headscale** | control plane on a VPS | ~$5/mo | a weekend | identity, keys, policy |
| **3. Your own protocol** | everything, incl. relays | hosting + time | months | everything |

**Tier 2 is the right next step.** It removes the third party from your trust
path and gives you unlimited devices and your own user management, while still
letting you borrow Tailscale's DERP relays until you outgrow them.

**Tier 3 is a research project.** Before starting it, read the NAT-traversal
literature — Ford, Srisuresh & Kegel's *Peer-to-Peer Communication Across
Network Address Translators*, RFC 8445 (ICE), and libp2p's DCUtR docs. If the
goal is truly open peer-to-peer between strangers, **libp2p** is a better
foundation than reimplementing a centralized control plane, because it was
designed for permissionless networks from the start.

---

## Setting up Headscale (tier 2)

You need a VPS with a public IP and a domain name. Any $5/month box is enough —
the control plane carries coordination traffic, not your data.

### 1. Install

```bash
wget https://github.com/juanfont/headscale/releases/latest/download/headscale_linux_amd64
sudo install -m 755 headscale_linux_amd64 /usr/local/bin/headscale
sudo mkdir -p /etc/headscale /var/lib/headscale
```

Write `/etc/headscale/config.yaml` — start from the sample in the repo and set
at minimum:

```yaml
server_url: https://headscale.example.com
listen_addr: 0.0.0.0:8080
private_key_path: /var/lib/headscale/private.key
noise:
  private_key_path: /var/lib/headscale/noise_private.key
prefixes:
  v4: 100.64.0.0/10
database:
  type: sqlite
  sqlite:
    path: /var/lib/headscale/db.sqlite
```

### 2. Put HTTPS in front of it

Headscale must be reachable over TLS — clients refuse plain HTTP. Caddy is the
least effort:

```
headscale.example.com {
    reverse_proxy localhost:8080
}
```

Caddy obtains and renews the certificate automatically.

### 3. Create a user and a pre-auth key

```bash
headscale users create friends
headscale preauthkeys create --user friends --reusable --expiration 90d
```

That prints a hex key. **This is what goes into your join code** — note it is
not in Tailscale's `tskey-auth-` format, which is why p2pgpu relaxes the key
shape check once a login server is present.

### 4. Make join codes against it

```bash
p2pgpu invite --login-server https://headscale.example.com
```

Paste the printed auth key when prompted. Everyone who uses that code now joins
**your** network. Nothing else about p2pgpu changes.

---

## What you gain and give up

**Gain:** no third party in the trust path, unlimited devices and users, your own
policy and branding, and a foundation you can point at strangers later.

**Give up:** uptime is yours. If your VPS goes down, new devices cannot join and
existing peers cannot learn about changes. Already-established direct
connections keep working — WireGuard does not need the control plane once a
tunnel is up — which softens the blow considerably.

**Still borrowed:** DERP relays, unless you run your own. Headscale defaults to
Tailscale's public DERP mesh. That is fine and normal; revisit it only when you
have users whose connections actually fall back to relay.

---

## Before opening this to strangers

Self-hosting the control plane is necessary for a public network but nowhere
near sufficient. The unsolved problems, in the order they will bite:

1. **Untrusted code on volunteers' machines.** A container is not a sandbox
   against a determined attacker, and consumer GPUs have no TEE — hardware
   attestation is H100-class only. This is genuinely open research.
2. **Verifying work was done.** A node can return garbage instead of computing.
   See Gensyn's Verde, and SENTINEL for pipeline-stage verification.
3. **Privacy of the workload.** A host can inspect anything running on its GPU.
4. **Abuse.** Someone will mine crypto, or worse, on donated hardware.

None of these block a private cluster among friends, which is why v1 is scoped
that way. All of them block a public one.
