from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol
    
    class Adapter(Protocol):
        """Protocol for environment adapters (LLM, DB, robot, etc.)"""
        def update_embedding(self, embedding: Any) -> None:
            ...

from ..kernel import KERNEL

__all__ = ["Environment", "Adapter"]
class Environment:
    def __init__(self, text: str, adapter: Adapter):
        self.text = text
        self.adapter = adapter          # LLM, DB, robot, etc.

    def cover_mask(self) -> torch.Tensor:
        chars = {c for c in self.text if c in kernel}
        t = torch.zeros(len(kernel), device="cuda")
        t[[idx[c] for c in chars]] = 1.0
        return t

    def write_back(self, s: torch.Tensor):
        """adapter-specific: REST, SQL, ROS, …"""
        self.adapter.update_embedding(s.cpu().numpy())