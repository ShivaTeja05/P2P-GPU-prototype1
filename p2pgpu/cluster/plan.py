"""Decide which layers live on which GPU.

This is the part that turns "two GPUs" into "one bigger GPU". A transformer is a
stack of near-identical blocks, so it can be cut at any block boundary and the
halves placed on different machines. Cut it in the right place and a model that
fits on neither card fits across both.

The split is by *memory*, not by layer count. An RTX 4050 (6 GB) and an RTX 4080
(16 GB) should not get eight layers each -- the 4050 would OOM while the 4080 sat
two-thirds empty. Weighting by free VRAM is what makes heterogeneous hardware
usable, and heterogeneous hardware is the normal case when the GPUs belong to
different people.
"""

from __future__ import annotations

from dataclasses import dataclass

# Weights are not the only thing in VRAM. CUDA context, activations, the KV
# cache and allocator fragmentation all take a share, and running a card to
# 100% is how you get an OOM three hours into a job.
VRAM_HEADROOM = 0.85

BYTES_PER_PARAM = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}


@dataclass
class StageSpec:
    """One node's slice of the model."""

    rank: int
    node: str
    first_layer: int
    last_layer: int  # exclusive
    vram_free_mb: int
    estimated_mb: int

    @property
    def layer_count(self) -> int:
        return self.last_layer - self.first_layer

    @property
    def utilisation(self) -> float:
        budget = self.vram_free_mb * VRAM_HEADROOM
        return self.estimated_mb / budget if budget else float("inf")

    def __str__(self) -> str:
        return (
            f"rank {self.rank}  {self.node:<14} layers "
            f"[{self.first_layer:>3}, {self.last_layer:>3})  "
            f"{self.estimated_mb / 1024:5.1f} GB of "
            f"{self.vram_free_mb / 1024:5.1f} GB free  "
            f"({self.utilisation * 100:.0f}%)"
        )


class PlanError(ValueError):
    """The model cannot be placed on these GPUs. Message is user-facing."""


def layer_memory_mb(total_params: int, num_layers: int, dtype: str = "float16") -> float:
    """Rough MB per transformer block.

    Approximate on purpose. Embeddings and the LM head are not blocks and get
    charged to the end stages by the caller; this is the per-block cost that
    drives the split.
    """
    per_param = BYTES_PER_PARAM.get(dtype)
    if per_param is None:
        raise PlanError(f"unknown dtype {dtype!r}")
    if num_layers <= 0:
        raise PlanError("model reports no layers to split")
    return (total_params * per_param) / num_layers / (1024 * 1024)


def plan_split(
    nodes: list[tuple[str, int]],
    num_layers: int,
    total_params: int,
    dtype: str = "float16",
) -> list[StageSpec]:
    """Assign contiguous layer ranges to nodes, weighted by free VRAM.

    `nodes` is [(name, free_vram_mb), ...] in pipeline order. Contiguous
    because a pipeline stage must own a *run* of layers -- activations flow
    forward through the stack, and scattering layers would send the tensor back
    and forth across the internet several times per token.
    """
    if not nodes:
        raise PlanError("no nodes given")
    if num_layers < len(nodes):
        raise PlanError(
            f"{num_layers} layers cannot be split across {len(nodes)} nodes; "
            "use fewer nodes or a deeper model"
        )

    per_layer = layer_memory_mb(total_params, num_layers, dtype)
    budgets = [free * VRAM_HEADROOM for _name, free in nodes]
    total_budget = sum(budgets)
    if total_budget <= 0:
        raise PlanError("no free VRAM reported on any node")

    needed = per_layer * num_layers
    if needed > total_budget:
        raise PlanError(
            f"model needs ~{needed / 1024:.1f} GB but the cluster has "
            f"~{total_budget / 1024:.1f} GB usable. Try a smaller model, a "
            f"lower-precision dtype, or add a node."
        )

    # Hand out layers in proportion to each node's budget, then give the
    # remainder to whoever has the most spare room.
    shares = [b / total_budget for b in budgets]
    counts = [max(1, int(num_layers * s)) for s in shares]
    while sum(counts) > num_layers:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < num_layers:
        slack = [
            (budgets[i] - counts[i] * per_layer, i) for i in range(len(counts))
        ]
        counts[max(slack)[1]] += 1

    specs: list[StageSpec] = []
    cursor = 0
    for rank, ((name, free), count) in enumerate(zip(nodes, counts)):
        specs.append(
            StageSpec(
                rank=rank,
                node=name,
                first_layer=cursor,
                last_layer=cursor + count,
                vram_free_mb=free,
                estimated_mb=int(count * per_layer),
            )
        )
        cursor += count

    over = [s for s in specs if s.utilisation > 1.0]
    if over:
        worst = max(over, key=lambda s: s.utilisation)
        raise PlanError(
            f"{worst.node} would need {worst.estimated_mb / 1024:.1f} GB but has "
            f"{worst.vram_free_mb / 1024:.1f} GB free. The split cannot be balanced."
        )
    return specs


def describe(specs: list[StageSpec]) -> str:
    total_free = sum(s.vram_free_mb for s in specs)
    total_used = sum(s.estimated_mb for s in specs)
    lines = [str(s) for s in specs]
    lines.append(
        f"\ncombined: {total_used / 1024:.1f} GB of model across "
        f"{total_free / 1024:.1f} GB of VRAM in {len(specs)} machines"
    )
    return "\n".join(lines)
