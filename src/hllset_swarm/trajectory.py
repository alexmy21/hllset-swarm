"""
YAML → GPU kernel compiler + high-level SwarmProgram API
"""
from __future__ import annotations
import yaml, torch, pathlib
from typing import Dict, Any, Callable, Optional
from .swarm import SwarmState
from .lattice import Lattice, build_matrices
from .kernel import get_kernel
from .io.env import Environment
from .io.github import github_upload

__all__ = ["SwarmProgram"]

# ---------- YAML DSL ----------
class SwarmProgram:
    """
    High-level object that:
    1. loads a YAML trajectory script,
    2. compiles it to a list of GPU kernel calls,
    3. owns the SwarmState and Lattice,
    4. commits the final layer to Github.
    """
    def __init__(self, cfg: Dict[str, Any], lattice: Lattice, kernel_idx: Dict[str, int]):
        self.cfg = cfg
        self.lattice = lattice
        self.kernel_idx = kernel_idx
        self.swarm = SwarmState(
            n=len(kernel_idx),
            alpha=cfg["params"]["alpha"],
            beta=cfg["params"]["beta"],
            gamma=cfg["params"]["gamma"],
        )
        self.trajectory = TrajectoryCompiler(cfg["trajectory"]).compile()
        self.parent_sha: Optional[str] = None

    # ---------- factory ----------
    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "SwarmProgram":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        lattice = build_matrices(
            tau0=cfg.get("tau0", 0.35),
            rho0=cfg.get("rho0", 0.15),
            device="cuda",
        )
        return cls(cfg, lattice, lattice.idx)

    # ---------- run ----------
    def run(self, env: Environment) -> torch.Tensor:
        """Execute compiled trajectory and return final s vector."""
        teacher = None
        for kernel in self.trajectory:
            teacher = kernel(self, env, teacher)   # compiled op
        return self.swarm.s

    # ---------- commit ----------
    def commit_to_github(self, repo: str, token: str, meta: Optional[Dict] = None) -> str:
        from .cortex import CortexLayer
        layer = CortexLayer(self.swarm.s, self.lattice, meta or {"entry": env.text})
        github_upload(repo, f"layers/{layer.sha}.pt.zst", layer.pack(), token)
        self.parent_sha = layer.sha
        return layer.sha


# ---------- internal compiler ----------
class TrajectoryCompiler:
    def __init__(self, steps: list[Dict[str, Any]]):
        self.steps = steps

    def compile(self) -> list[Callable]:
        return [self._build_op(op) for op in self.steps]

    def _build_op(self, op: Dict[str, Any]) -> Callable:
        match op["op"]:
            case "reset":
                return lambda prog, env, t: prog.swarm.s.fill_(op["value"])
            case "cover":
                return lambda prog, env, t: env.cover_mask()
            case "teacher":
                return lambda prog, env, t: env.cover_mask() if op.get("source") == "env.revision" else t
            case "converge":
                return lambda prog, env, t: self._converge(prog, t, **op)
            case "feedback":
                return lambda prog, env, t: prog._feedback(env)
            case _:
                raise ValueError(f"Unknown op: {op['op']}")

    # ---------- converge loop ----------
    def _converge(self, prog: SwarmProgram, teacher: torch.Tensor, max_steps: int, tol: float):
        for _ in range(max_steps):
            old = prog.swarm.s.clone()
            prog.swarm.step(teacher)
            if torch.linalg.norm(prog.swarm.s - old) < tol:
                break
        return teacher

    # ---------- write back ----------
    def _feedback(self, prog: SwarmProgram, env: Environment):
        env.write_back(prog.swarm.s)
        return None