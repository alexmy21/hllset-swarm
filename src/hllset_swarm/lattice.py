"""
Build / store / update sparse Wτ (coverage) and Wρ (exclusion) matrices
from kernel HLLSets.
"""
from __future__ import annotations
import torch
from typing import Tuple
from .kernel import get_kernel

__all__ = ["Lattice", "build_matrices"]

class Lattice:
    """Container for sparse COO matrices + τ-ρ thresholds."""
    __slots__ = ("Wτ", "Wρ", "tau0", "rho0", "idx")

    def __init__(self, Wτ: torch.Tensor, Wρ: torch.Tensor, tau0: float, rho0: float, idx: dict[str, int]):
        self.Wτ: torch.Tensor = Wτ.coalesce()          # sparse COO float32
        self.Wρ: torch.Tensor = Wρ.coalesce()
        self.tau0, self.rho0 = tau0, rho0
        self.idx = idx                                 # char → row/col

    # ---- Hebbian update after swarm ----
    def update_edge(self, u: int, v: int, delta: float, rho_ratio: float = 0.3):
        """in-place update of a single edge"""
        self.Wτ += torch.sparse_coo_tensor([[u], [v]], [delta], size=self.Wτ.shape, device=self.Wτ.device)
        self.Wρ += torch.sparse_coo_tensor([[u], [v]], [delta * rho_ratio], size=self.Wρ.shape, device=self.Wρ.device)
        self.Wτ = self.Wτ.coalesce()
        self.Wρ = self.Wρ.coalesce()

def build_matrices(tau0: float = 0.35, rho0: float = 0.15, device: str = "cuda") -> Lattice:
    """Build initial Wτ, Wρ from kernel BSS."""
    kernel = get_kernel()
    idx = {c: i for i, c in enumerate(kernel.keys())}
    n = len(kernel)
    rows, cols, tau_vals, rho_vals = [], [], [], []

    for u_ch, u_hll in kernel.items():
        u = idx[u_ch]
        for v_ch, v_hll in kernel.items():
            v = idx[v_ch]
            if u == v:
                continue
            metrics = u_hll.calculate_bss_to(v_hll)
            if metrics.tau >= tau0 and metrics.rho <= rho0:
                rows.append(u)
                cols.append(v)
                tau_vals.append(metrics.tau)
                rho_vals.append(metrics.rho)

    Wτ = torch.sparse_coo_tensor([rows, cols], tau_vals, size=(n, n), device=device)
    Wρ = torch.sparse_coo_tensor([rows, cols], rho_vals, size=(n, n), device=device)
    return Lattice(Wτ, Wρ, tau0, rho0, idx)