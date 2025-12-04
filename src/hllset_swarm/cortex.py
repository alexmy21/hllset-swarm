"""
Git-like commits of SwarmState + lattice matrices.
Also implements meta-swarm: each particle = previous s vector.
"""
from __future__ import annotations
import hashlib, json, io, torch, zstandard as zstd
from typing import Optional, Dict
from .lattice import Lattice
from .kernel import get_kernel

__all__ = ["CortexLayer", "MetaSwarm"]

class CortexLayer:
    """Single commit = (s, Wτ, Wρ, meta)."""
    __slots__ = ("sha", "s", "lattice", "meta")

    def __init__(self, s: torch.Tensor, lattice: Lattice, meta: Dict):
        self.s = s.detach().cpu().half()          # 80 k float16
        self.lattice = lattice
        self.meta = meta
        self.sha = self._compute_sha()

    def _compute_sha(self) -> str:
        buf = io.BytesIO()
        torch.save({"s": self.s, "Wτ": self.lattice.Wτ, "meta": self.meta}, buf)
        return hashlib.sha256(zstd.compress(buf.getvalue())).hexdigest()[:16]

    def pack(self) -> bytes:
        buf = io.BytesIO()
        torch.save({
            "s": self.s,
            "Wτ_coo": self.lattice.Wτ.coalesce(),
            "Wρ_coo": self.lattice.Wρ.coalesce(),
            "meta": self.meta
        }, buf)
        return zstd.compress(buf.getvalue())

    @staticmethod
    def unpack(blob: bytes, device: str = "cuda") -> "CortexLayer":
        d = torch.load(io.BytesIO(zstd.decompress(blob)), map_location=device)
        lat = Lattice(d["Wτ_coo"], d["Wρ_coo"], 0.35, 0.15, get_kernel().idx)
        return CortexLayer(d["s"], lat, d["meta"])

# --------------------------------------
class MetaSwarm:
    """Swarm of swarms – each particle = previous CortexLayer.s"""
    def __init__(self, layers: list[CortexLayer], alpha: float = 0.2, beta: float = 0.15):
        self.particles = torch.stack([L.s for L in layers])          # (P, 80000)
        self.velocity = torch.zeros_like(self.particles)
        self.alpha, self.beta = alpha, beta

    def step(self, teacher: Optional[torch.Tensor] = None):
        """PSO update in particle space (each particle is an 80 k vector)."""
        cognitive = self.alpha * (self.particles - self.velocity)
        self.velocity += cognitive
        if teacher is not None:
            social = self.beta * (teacher - self.particles)
            self.velocity += social
        self.particles += self.velocity
        self.particles = torch.clamp(self.particles, 0.0, 1.0)

    def centroid(self) -> torch.Tensor:
        return self.particles.mean(dim=0)