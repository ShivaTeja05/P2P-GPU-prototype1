"""p2pgpu command line.

Two roles:
  * the person WITH the GPU runs  init / share / status / stop
  * the person BORROWING it runs  init / attach / probe / bench / inspect
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from p2pgpu import __version__
from p2pgpu.common.config import (
    AUTH_HEADER,
    TOKEN_FILE,
    cluster_token,
    generate_token,
    node_id,
    write_cluster_token,
)
from p2pgpu.common.netbench import benchmark_link, transfer_estimates
from p2pgpu.common.schema import NodeCapabilities

app = typer.Typer(
    help="Share one GPU between two trusted machines over Tailscale.",
    no_args_is_help=True,
)
console = Console()


def _fmt_gb(mb: int | None) -> str:
    return "-" if mb is None else f"{mb / 1024:.1f} GB"


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print version and this machine's identity."""
    console.print(f"p2pgpu {__version__}  node_id={node_id()}")


@app.command()
def init(
    token: str = typer.Option("", help="Join using the other machine's token."),
    force: bool = typer.Option(False, help="Overwrite an existing token."),
) -> None:
    """Create this machine's identity and shared token.

    Run bare on the first machine, then run with --token on the second.
    """
    if TOKEN_FILE.exists() and not force and not token:
        console.print(f"[yellow]Token already exists at {TOKEN_FILE}. Use --force to regenerate.[/]")
        raise typer.Exit(1)

    value = token.strip() or generate_token()
    path = write_cluster_token(value)
    console.print(f"node_id: [cyan]{node_id()}[/]")
    console.print(f"token written to [cyan]{path}[/] (mode 600)")
    if not token:
        console.print(
            Panel(value, title="shared token - send privately, never commit", border_style="red")
        )


@app.command()
def probe() -> None:
    """Show what compute this machine has."""
    from p2pgpu.worker.probe import probe as run_probe

    _render_caps(run_probe())


# ---------------------------------------------------------------------------
# GPU owner side
# ---------------------------------------------------------------------------


@app.command()
def prepare(
    image: str = typer.Option("", help="Image to pre-download (default: auto-detected)."),
) -> None:
    """Do the slow parts now, so sharing later is instant.

    Starts Docker if needed and downloads the session image. Run this once at
    install time -- otherwise the multi-GB download lands the moment someone
    actually wants to use the GPU, which is the worst possible moment.
    """
    from p2pgpu.worker import gpu_compat, share as sharing

    if not sharing.docker_available():
        console.print("Docker is not running. Starting it...")
        if sharing.start_docker_desktop():
            console.print("[green]Docker is up.[/]")
        else:
            console.print("[red]Could not start Docker. Start Docker Desktop manually.[/]")
            raise typer.Exit(1)
    else:
        console.print("[green]Docker is running.[/]")

    chosen, reason = sharing.resolve_image(image or None, "all")
    console.print(f"image: [cyan]{chosen}[/] [dim]({reason})[/]")

    if sharing.image_present(chosen):
        console.print("[green]Image already downloaded. Sharing will start in seconds.[/]")
        return

    console.print("\n[yellow]Downloading the session image. This is the one big wait.[/]")
    console.print("[dim]Several GB, once only. Progress below.[/]\n")
    try:
        sharing.pull_image(chosen)
    except sharing.ShareError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    console.print("\n[green]Ready. Future shares start in seconds.[/]")


@app.command()
def share(
    hours: float = typer.Option(4.0, help="Auto-stop after this many hours."),
    image: str = typer.Option("", help="Override the container image (default: auto-detected)."),
    port: int = typer.Option(8888, help="Port for the notebook server."),
    bind_ip: str = typer.Option("", help="Override the bind address (default: Tailscale IP)."),
    gpu: str = typer.Option("all", help="Which GPUs to share: 'all', '0', '0,1', or 'none'."),
    ssh_key: str = typer.Option("", help="Their SSH public key -- also enables SSH into the container."),
    ssh_key_file: str = typer.Option("", help="Read the public key from a file instead."),
    ssh_port: int = typer.Option(2222, help="Host port to expose container SSH on."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait until the notebook is actually up."),
    skip_checks: bool = typer.Option(False, help="Skip host preflight checks."),
) -> None:
    """Share this machine's GPU with your friend for a fixed time.

    Works with any CUDA GPU from Pascal (GTX 10-series) to Blackwell (RTX
    50-series) -- the container image is chosen automatically from the card's
    compute capability and the installed driver.

    Runs a container with the GPU passed through and a notebook server bound to
    your Tailscale address. Your files stay out of it -- only ~/p2pgpu-workspace
    is mounted. The share stops itself when the time is up.
    """
    from p2pgpu.worker import gpu_compat, share as sharing

    if not skip_checks and not sharing.docker_available():
        console.print("[dim]Docker is not running. Starting it...[/]")
        sharing.start_docker_desktop()

    if not skip_checks:
        problems, warnings = sharing.preflight(
            require_tailscale=not bind_ip,
            require_gpu=gpu.strip().lower() != "none",
        )
        for warning in warnings:
            console.print(f"[yellow]warning:[/] {warning}")
        if problems:
            console.print("[red]Host is not ready:[/]")
            for problem in problems:
                console.print(f"  [red]x[/] {problem}")
            raise typer.Exit(1)

    key = ssh_key.strip()
    if ssh_key_file:
        try:
            key = Path(ssh_key_file).expanduser().read_text().strip()
        except OSError as exc:
            console.print(f"[red]Could not read {ssh_key_file}: {exc}[/]")
            raise typer.Exit(1) from exc
    if key:
        try:
            sharing.validate_ssh_key(key)
        except sharing.ShareError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc

    chosen_image, reason = sharing.resolve_image(image or None, gpu)
    for note in gpu_compat.compatibility_notes(gpu_compat.detect_gpus()):
        console.print(f"[yellow]note:[/] {note}")
    console.print(f"image: [cyan]{chosen_image}[/] [dim]({reason})[/]")
    if not sharing.image_present(chosen_image):
        console.print(
            f"\n[yellow]The image is not downloaded yet ({chosen_image}).[/]\n"
            "[dim]This is 7-9 GB and only happens once. Progress below.[/]\n"
        )
    console.print(f"Starting share ({hours}h)...")
    try:
        session = sharing.start_share(
            hours=hours,
            image=image or None,
            port=port,
            bind_ip=bind_ip or None,
            gpus=gpu,
            ssh_key=key or None,
            ssh_port=ssh_port,
        )
    except sharing.ShareError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    if wait:
        console.print(
            "[dim]Container started. Installing packages inside it -- this takes a few\n"
            "minutes on a first run. Waiting until it actually answers...[/]"
        )
        with console.status("[cyan]waiting for the notebook to come up..."):
            ready = sharing.wait_until_ready(session.bind_ip, session.port)
        if not ready:
            console.print("[red]The notebook never came up. Setup log:[/]")
            console.print(sharing.setup_log() or "[dim](empty)[/]")
            console.print("\nThe container may still be running: [cyan]p2pgpu stop[/] to clean up.")
            raise typer.Exit(1)
        console.print("[green]Ready.[/]\n")

    console.print(
        Panel(
            session.url,
            title=f"send this to your friend - expires in {hours}h",
            border_style="green",
        )
    )
    if session.ssh_command:
        console.print(
            Panel(session.ssh_command, title="they can also SSH in", border_style="cyan")
        )
    console.print(f"workspace (shared with the container): [cyan]{session.workspace}[/]")
    console.print("stop early with [cyan]p2pgpu stop[/]  ·  check with [cyan]p2pgpu status[/]")


@app.command()
def status() -> None:
    """Show whether this machine is currently sharing its GPU."""
    from p2pgpu.worker import share as sharing

    running = sharing.container_running()
    session = sharing.load_session()

    if not running:
        console.print("[dim]Not sharing.[/]")
        if session:
            console.print("[yellow]A previous session expired or was stopped.[/]")
        return

    table = Table(title="Sharing active", show_header=False)
    if session:
        remaining = session.remaining_s
        table.add_row("url", session.url)
        table.add_row("bound to", f"{session.bind_ip}:{session.port}")
        table.add_row("time left", f"{remaining / 3600:.1f} h")
        if session.ssh_command:
            table.add_row("ssh", session.ssh_command)
        table.add_row("workspace", session.workspace)
        table.add_row("image", session.image)
    else:
        table.add_row("container", sharing.CONTAINER_NAME)
        table.add_row("note", "running, but no saved session details")
    console.print(table)


@app.command()
def mykey(
    generate: bool = typer.Option(True, "--generate/--no-generate", help="Create a key if none exists."),
) -> None:
    """Print your SSH public key, to send to whoever is lending you their GPU.

    Safe to share: this is the *public* half. Your private key never leaves this
    machine.
    """
    import subprocess

    ssh_dir = Path.home() / ".ssh"
    for name in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"):
        candidate = ssh_dir / name
        if candidate.exists():
            console.print(f"[dim]{candidate}[/]\n")
            print(candidate.read_text().strip())
            console.print("\n[dim]Send that whole line to your friend.[/]")
            return

    if not generate:
        console.print("[yellow]No SSH key found and --no-generate was passed.[/]")
        raise typer.Exit(1)

    console.print("No SSH key found. Creating an ed25519 key...")
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    target = ssh_dir / "id_ed25519"
    try:
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(target), "-N", "", "-q"],
            check=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        console.print(f"[red]Could not run ssh-keygen: {exc}[/]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Created {target}[/]\n")
    print(target.with_suffix(".pub").read_text().strip())
    console.print("\n[dim]Send that whole line to your friend.[/]")


@app.command()
def url() -> None:
    """Print just the share URL, with no formatting.

    For scripts and for piping to the clipboard.
    """
    from p2pgpu.worker import share as sharing

    session = sharing.load_session()
    if session is None or not sharing.container_running():
        raise typer.Exit(1)
    print(session.url)


@app.command()
def stop() -> None:
    """Stop sharing the GPU right now."""
    from p2pgpu.worker import share as sharing

    if sharing.stop_share():
        console.print("[green]Share stopped. GPU is yours again.[/]")
    else:
        console.print("[dim]Nothing was running.[/]")


@app.command()
def logs(lines: int = typer.Option(50, help="How many lines to show.")) -> None:
    """Show the shared container's output (useful when it won't start)."""
    from p2pgpu.worker import share as sharing

    setup = sharing.setup_log()
    if setup:
        console.print("[bold]container setup:[/]")
        console.print(setup)
        console.print()
    console.print("[bold]notebook output:[/]")
    console.print(sharing.share_logs(lines) or "[dim]no output[/]")


@app.command()
def doctor(
    quick: bool = typer.Option(False, help="Skip the container GPU test (no image pull)."),
) -> None:
    """Check whether this machine can share its GPU, and say what's missing."""
    from p2pgpu.worker import gpu_compat, share as sharing

    table = Table(title="Host readiness", show_header=False)
    ts_ip = sharing.tailscale_ip()
    checks = [
        ("Docker", sharing.docker_available()),
        ("NVIDIA container runtime", sharing.nvidia_runtime_registered()),
        ("Tailscale", ts_ip is not None),
    ]
    for name, ok in checks:
        table.add_row(name, "[green]ok[/]" if ok else "[red]missing[/]")
    if ts_ip:
        table.add_row("Tailscale IP", ts_ip)
    console.print(table)

    profiles = gpu_compat.detect_gpus()
    if profiles:
        gtable = Table(title="Detected GPUs")
        for column in ("#", "name", "arch", "cc", "VRAM", "driver"):
            gtable.add_column(column)
        for profile in profiles:
            gtable.add_row(
                str(profile.index),
                profile.name,
                profile.architecture,
                profile.compute_capability or "?",
                f"{profile.vram_mb / 1024:.0f} GB",
                profile.driver_version or "?",
            )
        console.print(gtable)

        image, reason = gpu_compat.recommend_image(profiles)
        console.print(f"selected image: [cyan]{image}[/]\n[dim]{reason}[/]")
        for note in gpu_compat.compatibility_notes(profiles):
            console.print(f"[yellow]note:[/] {note}")

    problems, warnings = sharing.preflight()
    for warning in warnings:
        console.print(f"[yellow]warning:[/] {warning}")
    if problems:
        console.print("\n[yellow]To fix:[/]")
        for problem in problems:
            console.print(f"  · {problem}")
        raise typer.Exit(1)

    if quick:
        console.print("\n[green]Prerequisites look good.[/] Re-run without --quick to test passthrough.")
        return

    console.print("\nRunning a real GPU passthrough test (may pull ~200 MB)...")
    ok, detail = sharing.verify_gpu_passthrough()
    if ok:
        console.print("[green]GPU is visible inside Docker. Ready to share.[/]")
    else:
        console.print(f"[red]GPU passthrough failed:[/]\n{detail}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Borrower side
# ---------------------------------------------------------------------------


@app.command()
def attach(
    url: str = typer.Argument(..., help="The URL your friend sent you."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open in a browser."),
) -> None:
    """Connect to a GPU your friend is sharing."""
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        reachable = resp.status_code < 400
    except httpx.HTTPError as exc:
        console.print(f"[red]Could not reach {url}[/]")
        console.print(f"[dim]{exc}[/]")
        console.print(
            "\nCheck: is Tailscale up on both machines "
            "([cyan]tailscale status[/]), and is your friend still sharing "
            "([cyan]p2pgpu status[/] on their machine)?"
        )
        raise typer.Exit(1) from exc

    if not reachable:
        console.print(f"[yellow]Reached the server but got HTTP {resp.status_code}.[/]")
        console.print("The token may be wrong or the share may have expired.")
        raise typer.Exit(1)

    console.print("[green]Connected.[/] The notebook runs on your friend's GPU.")
    console.print("Files you put in /workspace persist on their machine and survive the session.")
    if open_browser:
        webbrowser.open(url)
    else:
        console.print(url)


@app.command()
def discover() -> None:
    """Find every p2pgpu machine on your tailnet. No URLs, no copy-paste."""
    from p2pgpu.common.discovery import discover as run_discover

    token = cluster_token()
    console.print("[dim]scanning tailnet...[/]")
    found = run_discover(token)
    if not found:
        console.print("[yellow]No p2pgpu machines found.[/]")
        console.print("Is the other machine running [cyan]p2pgpu agent[/]? Check [cyan]tailscale status[/].")
        raise typer.Exit(1)

    table = Table(title="Machines on your tailnet")
    for col in ("host", "os", "GPU", "free", "status"):
        table.add_column(col)
    for item in found:
        if item.sharing:
            status = f"[green]sharing[/] ({item.hours_left}h left)"
        else:
            status = "[dim]idle[/]"
        table.add_row(
            item.peer.hostname,
            item.peer.os,
            item.gpu_name or "[dim]none[/]",
            f"{item.gpu_free_mb / 1024:.1f} GB" if item.gpu_free_mb else "-",
            status,
        )
    console.print(table)
    if any(f.sharing for f in found):
        console.print("\nConnect with [cyan]p2pgpu connect[/]")


@app.command()
def connect(
    host: str = typer.Option("", help="Hostname to connect to (default: the only active share)."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open in a browser."),
) -> None:
    """Connect to a shared GPU on your tailnet, without needing a URL."""
    from p2pgpu.common.discovery import discover as run_discover

    token = cluster_token()
    console.print("[dim]looking for shared GPUs...[/]")
    active = [f for f in run_discover(token) if f.sharing and f.share_url]

    if host:
        active = [f for f in active if f.peer.hostname.lower() == host.lower()]
    if not active:
        console.print("[yellow]No active shares found on your tailnet.[/]")
        console.print("Ask them to run [cyan]p2pgpu share[/] (or Share-My-GPU.bat).")
        raise typer.Exit(1)
    if len(active) > 1:
        console.print("Several machines are sharing. Pick one with [cyan]--host[/]:")
        for item in active:
            console.print(f"  · {item.label}")
        raise typer.Exit(1)

    target = active[0]
    console.print(f"[green]Found:[/] {target.label}")
    if target.ssh_command:
        console.print(f"[dim]ssh available: {target.ssh_command}[/]")
    attach(target.share_url, open_browser=open_browser)


@app.command()
def agent(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8777, help="Port to listen on."),
) -> None:
    """Run the always-on agent so peers can discover this machine."""
    serve(host=host, port=port)


@app.command()
def inspect(url: str = typer.Argument(..., help="Worker base URL, e.g. http://100.x.y.z:8777")) -> None:
    """Fetch a remote machine's capabilities (needs 'p2pgpu serve' running there)."""
    token = cluster_token()
    if token is None:
        console.print("[red]No token. Run 'p2pgpu init --token <value>' first.[/]")
        raise typer.Exit(1)
    try:
        resp = httpx.get(
            f"{url.rstrip('/')}/v1/capabilities", headers={AUTH_HEADER: token}, timeout=20.0
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[red]Could not reach worker: {exc}[/]")
        raise typer.Exit(1) from exc
    _render_caps(NodeCapabilities.model_validate(resp.json()))


@app.command()
def bench(
    url: str = typer.Argument(..., help="Worker base URL, e.g. http://100.x.y.z:8777"),
    samples: int = typer.Option(20, help="RTT probes to send."),
    payload_mb: int = typer.Option(8, help="Throughput payload size in MiB."),
) -> None:
    """Measure the link, and estimate how long real transfers will take."""
    token = cluster_token()
    console.print(f"Benchmarking [cyan]{url}[/] ...")
    try:
        link = benchmark_link(url, token, rtt_samples=samples, payload_mb=payload_mb)
    except httpx.HTTPError as exc:
        console.print(f"[red]Benchmark failed: {exc}[/]")
        raise typer.Exit(1) from exc

    table = Table(title="Link", show_header=False)
    table.add_row("peer", link.peer)
    table.add_row("RTT min / p50 / p95", f"{link.rtt_ms_min} / {link.rtt_ms_p50} / {link.rtt_ms_p95} ms")
    table.add_row("upload", f"{link.upload_mbps} Mbps" if link.upload_mbps else "[dim]skipped[/]")
    table.add_row("download", f"{link.download_mbps} Mbps" if link.download_mbps else "[dim]skipped[/]")
    console.print(table)

    ttable = Table(title="What that means in practice")
    ttable.add_column("transfer")
    ttable.add_column("time", justify="right")
    for label, _size, human in transfer_estimates(link.upload_mbps, link.download_mbps):
        ttable.add_row(label, human)
    console.print(ttable)

    if link.rtt_ms_p50 > 120:
        console.print("[yellow]High latency - an interactive notebook will feel laggy.[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8777, help="Port to listen on."),
) -> None:
    """Run the capability/benchmark agent (optional, for probe and bench)."""
    if cluster_token() is None:
        console.print("[red]No token. Run 'p2pgpu init' first.[/]")
        raise typer.Exit(1)
    from p2pgpu.worker.agent import serve as run_server

    console.print(f"agent [cyan]{node_id()}[/] listening on {host}:{port}")
    run_server(host=host, port=port)


def _render_caps(caps: NodeCapabilities) -> None:
    table = Table(title=f"{caps.hostname}  ({caps.node_id})", show_header=False)
    table.add_row("platform", f"{caps.platform} / {caps.arch}")
    table.add_row("cpu", f"{caps.cpu_model or '?'} ({caps.cpu_cores} threads)")
    table.add_row("ram", f"{_fmt_gb(caps.ram_free_mb)} free of {_fmt_gb(caps.ram_total_mb)}")
    table.add_row("backend", f"[bold]{caps.backend}[/]")
    table.add_row("torch", caps.torch_version or "[dim]not installed[/]")
    console.print(table)

    if not caps.gpus:
        console.print("[yellow]No GPU detected -- CPU only.[/]")
        return

    gtable = Table(title="GPUs")
    for column in ("#", "name", "vendor", "total", "free", "cc"):
        gtable.add_column(column)
    for gpu in caps.gpus:
        gtable.add_row(
            str(gpu.index),
            gpu.name,
            gpu.vendor,
            _fmt_gb(gpu.total_memory_mb),
            _fmt_gb(gpu.free_memory_mb),
            gpu.compute_capability or "-",
        )
    console.print(gtable)
    console.print(f"usable VRAM: [bold green]{_fmt_gb(caps.usable_vram_mb)}[/]")


if __name__ == "__main__":
    app()
