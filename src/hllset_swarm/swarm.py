# src/hllset_swarm/swarm.py
import torch
from .lattice import Lattice

class SwarmState:
    """GPU resident vector + sparse matrices"""
    def __init__(self, n: int, alpha: float, beta: float, gamma: float):
        self.s = torch.full((n,), 0.5, device="cuda")
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def step(self, teacher: torch.Tensor | None = None):
        cognitive = torch.sparse.mm(Lattice.Wτ, self.s.unsqueeze(1)).squeeze()
        exclusion = torch.sparse.mm(Lattice.Wρ, self.s.unsqueeze(1)).squeeze()
        self.s += self.alpha * cognitive - self.beta * exclusion
        if teacher is not None:
            social = torch.sparse.mm(Lattice.Wτ, teacher.unsqueeze(1)).squeeze()
            self.s += self.gamma * social
        self.s = torch.clamp(self.s, 0.0, 1.0)