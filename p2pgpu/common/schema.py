"""Wire types shared by worker, coordinator and CLI.

Every field here is part of the network contract. Changing one means bumping
PROTOCOL_VERSION so mismatched nodes fail loudly instead of silently
misinterpreting each other's bytes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

Backend = Literal["cuda", "mps", "rocm", "cpu"]


class GPUInfo(BaseModel):
    index: int
    name: str
    vendor: str
    total_memory_mb: int
    # Free VRAM is the number that actually decides placement: a 4080 with a
    # browser and a game open is not a 16 GB GPU.
    free_memory_mb: int | None = None
    compute_capability: str | None = None
    driver_version: str | None = None


class NodeCapabilities(BaseModel):
    """What a worker advertises about itself. The scheduler plans against this."""

    protocol_version: int = PROTOCOL_VERSION
    node_id: str
    hostname: str
    platform: str
    arch: str
    cpu_model: str | None = None
    cpu_cores: int | None = None
    ram_total_mb: int
    ram_free_mb: int
    backend: Backend
    gpus: list[GPUInfo] = Field(default_factory=list)
    torch_version: str | None = None
    agent_version: str

    @property
    def total_vram_mb(self) -> int:
        return sum(g.total_memory_mb for g in self.gpus)

    @property
    def usable_vram_mb(self) -> int:
        """VRAM we can actually plan against right now."""
        return sum(g.free_memory_mb or g.total_memory_mb for g in self.gpus)


class LinkMeasurement(BaseModel):
    """Result of benchmarking the path between two nodes.

    The whole feasibility of pipeline parallelism rests on these numbers, so
    they are first-class data, not a log line.
    """

    peer: str
    rtt_ms_p50: float
    rtt_ms_p95: float
    rtt_ms_min: float
    upload_mbps: float | None = None
    download_mbps: float | None = None
    samples: int = 0


class HealthResponse(BaseModel):
    ok: bool = True
    node_id: str
    agent_version: str
    protocol_version: int = PROTOCOL_VERSION
    uptime_s: float
