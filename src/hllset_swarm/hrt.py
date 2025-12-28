import torch
from typing import List, Dict, Optional

class HashRelationTensor:
    """
    Binary relation tensor r(h_i, h_j) = v  addressed only by hashes.
    Shape grows automatically; all updates are one tensor op.
    """
    def __init__(self, init_cap=16, dtype=torch.float32, device='cpu'):
        self.device   = device
        self.dtype    = dtype
        self.capacity = init_cap
        self.R        = torch.zeros(init_cap, init_cap, dtype=dtype, device=device)

        self.h2i      = {}          # hash -> internal index
        self.i2h      = {}          # internal index -> hash
        self.next_idx = 0           # first free internal index

    # ---------- public ----------
    def update(self, h_i: int, h_j: int, value):
        """h_i, h_j are hash values (python ints)"""
        # obtain (or create) internal indices
        i = self._index_for_hash(h_i)
        j = self._index_for_hash(h_j)
        # single in-place tensor write
        self.R[i, j] = value

    def get(self, h_i: int, h_j: int):
        if h_i not in self.h2i or h_j not in self.h2i:
            raise KeyError("Relation not stored")
        return self.R[self.h2i[h_i], self.h2i[h_j]]

    def get_dense(self) -> torch.Tensor:
        """Return the dense sub-matrix actually in use (shape next_idx × next_idx)"""
        return self.R[:self.next_idx, :self.next_idx].clone()
    
    def prune(self, keep: List[int]) -> torch.Tensor:
        """
        Drop every token whose hash is NOT in `keep`.
        Rebuild h2i/i2h and shrink self.R to |keep|×|keep| in-place.
        Returns the dense pruned adjacency matrix.
        """
        keep_set = set(keep)
        # 1. build new contiguous index mapping
        new_h2i = {h: idx for idx, h in enumerate(h for h in keep if h in self.h2i)}
        n = len(new_h2i)
        if n == 0:                       # nothing left
            self.h2i.clear()
            self.i2h.clear()
            self.next_idx = 0
            self.capacity = 0
            self.R = torch.empty(0, 0, dtype=self.dtype, device=self.device)
            return self.R

        # 2. allocate compact matrix
        new_R = torch.zeros(n, n, dtype=self.dtype, device=self.device)

        # 3. copy old values into new positions (vectorised)
        old_rows, old_cols = [], []
        new_rows, new_cols = [], []
        vals = []
        for h_row in new_h2i:
            for h_col in new_h2i:
                old_rows.append(self.h2i[h_row])
                old_cols.append(self.h2i[h_col])
                new_rows.append(new_h2i[h_row])   # ✅ use NEW indices
                new_cols.append(new_h2i[h_col])
                vals.append(self.R[self.h2i[h_row], self.h2i[h_col]])

        if vals:
            new_R[torch.tensor(new_rows, device=self.device),
                torch.tensor(new_cols, device=self.device)] = torch.tensor(vals, device=self.device)

        # 4. overwrite internal state
        self.h2i = new_h2i
        self.i2h = {idx: h for h, idx in new_h2i.items()}
        self.R = new_R
        self.capacity = n
        self.next_idx = n
        return new_R
    
    # ====== history (incoming edges) ======
    def get_history(self, h: int, horizon: int = 1) -> Dict[int, float]:
        """Return {hash_of_past_token : edge_strength} within horizon steps."""
        if h not in self.h2i:
            return {}
        idx = self.h2i[h]
        # 1-step predecessors
        mask = self.R[:, idx] != 0          # (capacity,)
        preds = mask.nonzero(as_tuple=False).squeeze(1)  # (<=capacity,)
        strengths = self.R[preds, idx]      # (<=capacity,)
        # truncate to horizon closest (here 1 step – see multi-step below)
        topk = min(horizon, len(preds))
        if topk == 0:
            return {}
        top_vals, top_idx = torch.topk(strengths, k=topk)
        return {self.i2h[int(i)]: float(v) for i, v in zip(preds[top_idx], top_vals)}

    def get_history_batch(self, hs: List[int], horizon: int = 1) -> List[Dict[int, float]]:
        return [self.get_history(h, horizon) for h in hs]

    # ====== future (outgoing edges) ======
    def get_future(self, h: int, horizon: int = 1) -> Dict[int, float]:
        """Return {hash_of_future_token : edge_strength} within horizon steps."""
        if h not in self.h2i:
            return {}
        idx = self.h2i[h]
        mask = self.R[idx, :] != 0
        succs = mask.nonzero(as_tuple=False).squeeze(1)
        strengths = self.R[idx, succs]
        topk = min(horizon, len(succs))
        if topk == 0:
            return {}
        top_vals, top_idx = torch.topk(strengths, k=topk)
        return {self.i2h[int(i)]: float(v) for i, v in zip(succs[top_idx], top_vals)}

    def get_future_batch(self, hs: List[int], horizon: int = 1) -> List[Dict[int, float]]:
        return [self.get_future(h, horizon) for h in hs]

    # ====== projection (dense sub-matrix for a given token set) ======
    def project(self, hs: List[int]) -> torch.Tensor:
        """Return dense |hs|×|hs| relation matrix for the supplied hashes."""
        idx = [self.h2i[h] for h in hs if h in self.h2i]
        if not idx:
            return torch.empty(0, 0, dtype=self.dtype, device=self.device)
        idx_t = torch.tensor(idx, device=self.device)
        return self.R[idx_t][:, idx_t]   # elegant batched indexing

    # ---------- internal ----------
    def _index_for_hash(self, h):
        """Return internal index for hash h, growing tensor if necessary."""
        if h in self.h2i:
            return self.h2i[h]
        # new hash → assign next internal index
        if self.next_idx >= self.capacity:
            self._grow()
        idx = self.next_idx
        self.h2i[h] = idx
        self.i2h[idx] = h
        self.next_idx += 1
        return idx

    def _grow(self):
        old_cap = self.capacity
        new_cap = old_cap * 2
        new_R = torch.zeros(new_cap, new_cap, dtype=self.dtype, device=self.device)
        # copy old block in one tensor op
        new_R[:old_cap, :old_cap] = self.R
        self.R = new_R
        self.capacity = new_cap