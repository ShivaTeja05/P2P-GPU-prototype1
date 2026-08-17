"""Use several GPUs, in different houses, as one.

There are two genuinely different things people mean by "cluster", and they
solve different problems. Keeping them named apart matters, because picking the
wrong one wastes a lot of work:

  PIPELINE  (plan.py, stage.py, pipeline.py)   <- combines VRAM
      One model is cut into contiguous blocks of layers; each node holds a
      different block. A 4050 and a 4080 together hold a model neither could
      load alone. Activations flow forward through the stages, one small tensor
      per hop. This is what you want when the model does not fit.

  AVERAGE   (coordinator.py, trainer.py)       <- combines throughput
      Every node holds a *complete* copy of the model and trains on different
      data, syncing weights every few hundred steps (DiLoCo-style). Memory is
      not combined at all -- the model must already fit on each card. This is
      what you want when the model fits but training is slow.

Only AVERAGE can train today. PIPELINE runs a forward pass and nothing else:
training through it needs activations kept for the backward pass and gradients
sent back up the stages, which is a separate build.

Why not tensor parallelism, the thing datacenters actually use? It all-reduces
twice per layer, so an 80-layer model at 110 ms round-trip costs roughly 4.8
seconds per token. It needs NVLink-class latency and there is no way around
that.
"""

from p2pgpu.cluster.plan import PlanError, StageSpec, describe, plan_split
from p2pgpu.cluster.trainer import (
    ClusterTrainer,
    RoundResult,
    TrainerError,
    estimate_sync_cost,
    shard,
    sharded_batches,
)

__all__ = [
    "ClusterTrainer",
    "PlanError",
    "RoundResult",
    "StageSpec",
    "TrainerError",
    "describe",
    "estimate_sync_cost",
    "plan_split",
    "shard",
    "sharded_batches",
]
