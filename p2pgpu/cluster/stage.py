"""One node's slice of a model, served over HTTP.

A stage holds a contiguous run of transformer blocks and nothing else. It
receives hidden states, runs them through its blocks, and returns hidden
states. Stage 0 additionally owns the embedding; the last stage owns the final
norm and the LM head.

Why HTTP rather than torch.distributed: gloo forms a full mesh on ephemeral
ports, so every rank must be directly reachable by every other rank. Our GPUs
live inside Docker containers behind NAT, and on Windows the container's host is
the WSL2 VM, which does not carry the Windows Tailscale interface. A pipeline
needs only point-to-point, forward, one known port per stage -- which a
container can publish on its Tailscale address exactly the way the notebook
port already is.

The cost of a stage boundary is one round trip carrying hidden_dim * 2 bytes
per token. At 110 ms that caps the pipeline near 9 tokens/second, which is the
honest price of a model that fits on neither card alone.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from p2pgpu import __version__
from p2pgpu.worker.agent import require_token

# Hidden states for a long prompt are large but bounded; anything past this is
# a bug or a misconfigured batch size, not a legitimate request.
MAX_ACTIVATION_BYTES = 512 * 1024 * 1024


class StageError(RuntimeError):
    """Something is wrong with this stage's model or request."""


def find_layer_list(model: Any) -> tuple[Any, str]:
    """Locate the ModuleList of transformer blocks.

    Architectures disagree about where they keep it, and there is no common
    interface. These four cover essentially every decoder people actually run.
    """
    candidates = [
        ("model.layers", lambda m: m.model.layers),          # Llama, Mistral, Qwen
        ("transformer.h", lambda m: m.transformer.h),        # GPT-2, Falcon
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),    # Pythia, NeoX
        ("model.decoder.layers", lambda m: m.model.decoder.layers),  # OPT
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers, path
    raise StageError(
        "Could not find the transformer block list on this model. "
        "Supported layouts: model.layers, transformer.h, gpt_neox.layers, "
        "model.decoder.layers."
    )


@dataclass
class StageConfig:
    model_id: str
    first_layer: int
    last_layer: int          # exclusive
    total_layers: int
    device: str = "cuda"
    dtype: str = "float16"

    @property
    def is_first(self) -> bool:
        return self.first_layer == 0

    @property
    def is_last(self) -> bool:
        return self.last_layer >= self.total_layers


@dataclass
class StageRuntime:
    config: StageConfig | None = None
    model: Any = None
    layers: Any = None
    loaded_at: float = 0.0
    forwards: int = 0
    total_forward_s: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)


RUNTIME = StageRuntime()
app = FastAPI(title="p2pgpu pipeline stage", version=__version__)


def block_kwargs(block: Any, hidden, position_ids, position_embeddings) -> dict:
    """Build only the arguments this block's forward actually accepts.

    Architectures disagree, and passing the wrong ones is silently wrong rather
    than loudly broken. GPT-2 blocks take no position argument at all -- their
    positional information was already added to the embedding. Llama-family
    blocks need `position_embeddings`, the precomputed rotary cos/sin pair, and
    produce garbage without it.
    """
    import inspect

    accepted = set(inspect.signature(block.forward).parameters)
    kwargs: dict[str, Any] = {}
    if "position_embeddings" in accepted and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    if "position_ids" in accepted:
        kwargs["position_ids"] = position_ids
    return kwargs


def embed_inputs(model: Any, ids):
    """Token ids -> hidden states, including any learned positional embedding.

    GPT-2 keeps a separate `wpe` table that the model adds before the first
    block. Skipping it costs no error and ruins every result, so it is handled
    explicitly rather than left to the block loop.
    """
    import torch

    hidden = model.get_input_embeddings()(ids.long())
    transformer = getattr(model, "transformer", None)
    wpe = getattr(transformer, "wpe", None) if transformer is not None else None
    if wpe is not None:
        seq = ids.shape[-1]
        positions = torch.arange(seq, device=ids.device).unsqueeze(0)
        hidden = hidden + wpe(positions)
    return hidden


def rotary_embeddings(model: Any, hidden, position_ids):
    """Precompute rotary cos/sin if this architecture uses them."""
    inner = getattr(model, "model", None)
    rotary = getattr(inner, "rotary_emb", None) if inner is not None else None
    if rotary is None:
        return None
    try:
        return rotary(hidden, position_ids)
    except (TypeError, RuntimeError):
        return None


def final_norm(model: Any):
    """The norm applied after the last block, wherever this family keeps it."""
    for owner, name in (
        (getattr(model, "model", None), "norm"),
        (getattr(model, "transformer", None), "ln_f"),
        (getattr(model, "gpt_neox", None), "final_layer_norm"),
    ):
        if owner is not None:
            layer = getattr(owner, name, None)
            if layer is not None:
                return layer
    return None


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}.get(name, torch.float16)


def load_stage(config: StageConfig) -> dict:
    """Load only this stage's layers onto the device.

    The whole model is materialised on CPU first, then everything outside our
    range is dropped *before* anything moves to the GPU. Loading the full model
    onto the GPU and then trimming would defeat the point -- the peak is what
    OOMs, not the steady state.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=_torch_dtype(config.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()

    layers, path = find_layer_list(model)
    total = len(layers)
    if config.last_layer > total:
        raise StageError(
            f"stage wants layers [{config.first_layer}, {config.last_layer}) but "
            f"the model has {total}"
        )

    # Replace out-of-range blocks with None so their parameters are freed.
    keep = range(config.first_layer, config.last_layer)
    for index in range(total):
        if index not in keep:
            layers[index] = None

    if not config.is_last:
        # Only the final stage needs the head; it is often the single largest
        # tensor in the model (vocab x hidden), so dropping it matters.
        if hasattr(model, "lm_head"):
            model.lm_head = torch.nn.Identity()

    model.to(config.device)

    RUNTIME.config = config
    RUNTIME.model = model
    RUNTIME.layers = layers
    RUNTIME.loaded_at = time.time()
    RUNTIME.error = None
    RUNTIME.meta = {
        "layer_path": path,
        "total_layers": total,
        "hidden_size": getattr(hf_config, "hidden_size", None),
    }

    allocated = None
    if config.device.startswith("cuda") and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3

    return {
        "model": config.model_id,
        "layers": [config.first_layer, config.last_layer],
        "of": total,
        "device": config.device,
        "vram_gb": round(allocated, 2) if allocated else None,
        "layer_path": path,
    }


def _serialise(tensor) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _deserialise(raw: bytes):
    import torch

    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


@app.get("/health")
def health() -> dict:
    cfg = RUNTIME.config
    avg = (RUNTIME.total_forward_s / RUNTIME.forwards) if RUNTIME.forwards else None
    return {
        "ok": True,
        "role": "stage",
        "version": __version__,
        "loaded": cfg is not None,
        "model": cfg.model_id if cfg else None,
        "layers": [cfg.first_layer, cfg.last_layer] if cfg else None,
        "is_first": cfg.is_first if cfg else None,
        "is_last": cfg.is_last if cfg else None,
        "forwards": RUNTIME.forwards,
        "avg_forward_ms": round(avg * 1000, 1) if avg else None,
        "error": RUNTIME.error,
    }


@app.post("/v1/stage/load", dependencies=[Depends(require_token)])
def load(
    model_id: str,
    first_layer: int,
    last_layer: int,
    total_layers: int = 0,
    device: str = "cuda",
    dtype: str = "float16",
) -> dict:
    try:
        return load_stage(
            StageConfig(
                model_id=model_id,
                first_layer=first_layer,
                last_layer=last_layer,
                total_layers=total_layers or last_layer,
                device=device,
                dtype=dtype,
            )
        )
    except (StageError, OSError, RuntimeError, ValueError) as exc:
        RUNTIME.error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/stage/forward", dependencies=[Depends(require_token)])
async def forward(request: Request) -> Response:
    """Run this stage's blocks over the incoming hidden states.

    Stage 0 receives token ids and embeds them; every other stage receives
    hidden states from the stage before it.
    """
    import torch

    if RUNTIME.config is None:
        raise HTTPException(status_code=409, detail="no model loaded on this stage")

    raw = await request.body()
    if len(raw) > MAX_ACTIVATION_BYTES:
        raise HTTPException(status_code=413, detail="activation payload too large")

    cfg = RUNTIME.config
    model = RUNTIME.model
    started = time.perf_counter()

    try:
        payload = _deserialise(raw)
        tensor = payload["tensor"].to(cfg.device)

        with torch.no_grad():
            if cfg.is_first:
                hidden = embed_inputs(model, tensor)
            else:
                hidden = tensor.to(_torch_dtype(cfg.dtype))

            batch, seq, _ = hidden.shape
            position_ids = torch.arange(seq, device=cfg.device).unsqueeze(0).expand(batch, -1)
            rotary = rotary_embeddings(model, hidden, position_ids)

            for index in range(cfg.first_layer, cfg.last_layer):
                block = RUNTIME.layers[index]
                out = block(hidden, **block_kwargs(block, hidden, position_ids, rotary))
                hidden = out[0] if isinstance(out, tuple) else out

            if cfg.is_last:
                norm = final_norm(model)
                if norm is not None:
                    hidden = norm(hidden)

        result = {"tensor": hidden.detach().to("cpu"), "stage": cfg.first_layer}
    except (RuntimeError, KeyError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=f"forward failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    RUNTIME.forwards += 1
    RUNTIME.total_forward_s += elapsed
    return Response(content=_serialise(result), media_type="application/octet-stream")


@app.post("/v1/stage/unload", dependencies=[Depends(require_token)])
def unload() -> dict:
    import torch

    RUNTIME.model = None
    RUNTIME.layers = None
    RUNTIME.config = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True}


def serve(host: str = "0.0.0.0", port: int = 8900) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
