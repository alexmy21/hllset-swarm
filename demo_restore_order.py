#!/usr/bin/env python3
"""
Demo: restore order from random HLLSet (tiny Chinese corpus)
- ingest 3 short texts
- build HLLSets + inverted indices + lean adjacency
- pick a random bit-vector
- restore 5 candidate orderings
"""
import random, torch, sys
from typing import List, Dict

# ---------- CONFIG ----------
P              = 10            # 1024 registers
BATCH_SIZE     = 1000
BEAM           = 5
MAX_LEN        = 50
START_ID       = 0
END_ID         = 1

# ---------- UTILS ----------
def hash32(x: str) -> int:
    return abs(hash(x)) & ((1 << 32) - 1)

def slot(h: int) -> tuple[int, int]:
    return h % (1 << P), (h >> P) & 31

# ---------- 1.  INGESTION (raw stream, per-occurrence) ----------
# def ingest_corpus(corpus: List[str]):
#     """returns: adjacency, tok_occ_id, indices"""
#     adj_u, adj_v, adj_w = [], [], []
#     tok_occ_id = {}          # (token, gram, occurrence) → ID
#     occ_counter = 0
#     window = []

#     for line in corpus:
#         for ch in line.strip():
#             window.append(ch)
#             if len(window) == 3:
#                 g1, g2, g3 = window[0], ''.join(window[0:2]), ''.join(window)

#                 # ---- store IDs *before* using them ----
#                 for g, gram in [(g1, "1g"), (g2, "2g"), (g3, "3g")]:
#                     key = (g, gram, occ_counter)
#                     if key not in tok_occ_id:
#                         reg, run = slot(hash32(g))
#                         tok_occ_id[key] = len(tok_occ_id)
#                     occ_counter += 1

#                 # ---- 3 edges per window ----
#                 u1 = tok_occ_id[(g1, "1g", occ_counter - 3)]
#                 u2 = tok_occ_id[(g2, "2g", occ_counter - 2)]
#                 u3 = tok_occ_id[(g3, "3g", occ_counter - 1)]
#                 adj_u.extend([u1, u2, u3])
#                 adj_v.extend([u2, u3, tok_occ_id[(window[2], "1g", occ_counter - 1)]])  # collapse to last
#                 adj_w.extend([1.0, 1.0, 1.0])
#                 window.pop(0)

#     # ---- virtual start/end ----
#     start_id = len(tok_occ_id)
#     end_id   = start_id + 1
#     for (tok, gram, occ), uid in tok_occ_id.items():
#         if gram == "1g":
#             adj_u.extend([start_id, uid])
#             adj_v.extend([uid, end_id])
#             adj_w.extend([1.0, 1.0])

#     adj = torch.sparse_coo_tensor(
#         indices=torch.stack([torch.tensor(adj_u), torch.tensor(adj_v)]),
#         values=torch.tensor(adj_w), size=(len(tok_occ_id) + 2, len(tok_occ_id) + 2), dtype=torch.float32)
#     return adj, tok_occ_id

def ingest_corpus(corpus: List[str]):
    """returns: adjacency, tok_occ_id, indices"""
    adj_u, adj_v, adj_w = [], [], []
    tok_occ_id = {}          # (token, gram, occurrence) → ID
    global_occ = 0           # global occurrence counter
    window = []

    for line in corpus:
        for ch in line.strip():
            # Store 1-gram with current occurrence
            key_1g = (ch, "1g", global_occ)
            if key_1g not in tok_occ_id:
                tok_occ_id[key_1g] = len(tok_occ_id)
            
            window.append((ch, global_occ))
            global_occ += 1
            
            if len(window) >= 3:
                # Get the three characters and their occurrences
                ch1, occ1 = window[-3]
                ch2, occ2 = window[-2]
                ch3, occ3 = window[-1]
                
                # Create gram keys
                g2 = ch1 + ch2
                g3 = ch1 + ch2 + ch3
                
                # Store 2-gram and 3-gram with their occurrence
                key_2g = (g2, "2g", occ2)
                key_3g = (g3, "3g", occ3)
                
                if key_2g not in tok_occ_id:
                    tok_occ_id[key_2g] = len(tok_occ_id)
                if key_3g not in tok_occ_id:
                    tok_occ_id[key_3g] = len(tok_occ_id)
                
                # Add edges: 1g→2g, 2g→3g, 3g→1g
                u1 = tok_occ_id[(ch1, "1g", occ1)]
                u2 = tok_occ_id[key_2g]
                u3 = tok_occ_id[key_3g]
                u4 = tok_occ_id[(ch3, "1g", occ3)]
                
                adj_u.extend([u1, u2, u3])
                adj_v.extend([u2, u3, u4])
                adj_w.extend([1.0, 1.0, 1.0])
                
                window.pop(0)

    # ---- virtual start/end ----
    start_id = len(tok_occ_id)
    end_id   = start_id + 1
    
    for (tok, gram, occ), uid in tok_occ_id.items():
        if gram == "1g":
            adj_u.extend([start_id, uid])
            adj_v.extend([uid, end_id])
            adj_w.extend([1.0, 1.0])

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([torch.tensor(adj_u), torch.tensor(adj_v)]),
        values=torch.tensor(adj_w), 
        size=(len(tok_occ_id) + 2, len(tok_occ_id) + 2), 
        dtype=torch.float32
    )
    return adj, tok_occ_id

# ---------- 2.  BUILD HLLSet FROM CORPUS ----------
def build_hllset_from_corpus(corpus: List[str]) -> List[int]:
    bits = [0] * (1 << P)
    for line in corpus:
        for ch in line.strip():
            h = hash32(ch)
            reg, run = slot(h)
            bits[reg] |= 1 << run
    return bits

# ---------- 3.  COVER FROM BIT-VECTOR ----------
def cover_from_bits(bit_vec: List[int], tok_occ_id: dict) -> List[int]:
    """return *occurrence* token IDs whose (reg,run) bit is 1"""
    cover_ids = []
    for (tok, gram, occ), uid in tok_occ_id.items():
        if gram != "1g":
            continue
        reg, run = slot(hash32(tok))
        if bit_vec[reg] & (1 << run):
            cover_ids.append(uid)
    return cover_ids

# ---------- 4.  PRUNE + BEAM ----------
def prune_with_virtuals(cover_ids: List[int], full_adj: torch.Tensor, tok_occ_id: dict) -> tuple[torch.Tensor, int, int]:
    """Prune adjacency to keep only covered nodes plus virtual start/end"""
    start_id = len(tok_occ_id)
    end_id   = start_id + 1
    keep_ids = set(cover_ids + [start_id, end_id])
    
    # Extract edges
    indices = full_adj.coalesce().indices()
    values = full_adj.coalesce().values()
    
    # Filter edges where both endpoints are in keep_ids
    mask = torch.tensor([u.item() in keep_ids and v.item() in keep_ids 
                        for u, v in zip(indices[0], indices[1])], dtype=torch.bool)
    
    new_indices = indices[:, mask]
    new_values = values[mask]
    
    # Create mapping from old IDs to new compact IDs
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_ids))}
    
    # Remap indices
    remapped_indices = torch.stack([
        torch.tensor([old_to_new[idx.item()] for idx in new_indices[0]]),
        torch.tensor([old_to_new[idx.item()] for idx in new_indices[1]])
    ])
    
    pruned = torch.sparse_coo_tensor(
        remapped_indices, 
        new_values, 
        size=(len(keep_ids), len(keep_ids)),
        dtype=torch.float32
    )
    
    return pruned.coalesce(), old_to_new[start_id], old_to_new[end_id]

def generate_orderings(pruned_adj: torch.Tensor, start_id: int, end_id: int, id_to_tok: dict, beam: int = 5, max_len: int = 50) -> List[List[str]]:
    """Generate candidate orderings via beam search"""
    indices = pruned_adj.coalesce().indices()
    values = pruned_adj.coalesce().values()
    
    # Build adjacency dict
    adj_dict = {}
    for i in range(indices.shape[1]):
        u, v, w = int(indices[0, i]), int(indices[1, i]), float(values[i])
        adj_dict.setdefault(u, []).append((v, w))
    
    candidates = [([start_id], 0.0)]
    
    for step in range(max_len):
        next_cand = []
        for partial, score in candidates:
            last_id = partial[-1]
            if last_id == end_id:
                next_cand.append((partial, score))
                continue
                
            for next_id, weight in adj_dict.get(last_id, []):
                next_partial = partial + [next_id]
                next_score = score + weight
                next_cand.append((next_partial, next_score))
        
        if not next_cand:
            break
            
        candidates = sorted(next_cand, key=lambda x: x[1], reverse=True)[:beam]
        
        if all(p[-1] == end_id for p, _ in candidates):
            break
    
    # Convert IDs back to tokens
    cleaned = []
    for partial, _ in candidates:
        tokens = [id_to_tok.get(i) for i in partial if i not in (start_id, end_id)]
        tokens = [t for t in tokens if t is not None]
        if tokens:
            cleaned.append(tokens)
    
    return cleaned

# ---------- 5.  DEMO ----------
CORPUS = [
    "人工智能正在改变世界",
    "机器学习让代码更聪明",
    "深度学习是未来的钥匙"
]

if __name__ == "__main__":
    print("=== DEMO: restore order from random HLLSet ===")
    # ---- 1. ingest ----
    adj, tok_occ_id = ingest_corpus(CORPUS)
    print(f"Ingested: {len(tok_occ_id)} token occurrences, {adj._nnz()} edges")
    
    # ---- 2. build HLLSet ----
    original_bits = build_hllset_from_corpus(CORPUS)
    print(f"Original HLLSet: {sum(bin(b).count('1') for b in original_bits)} bits set / {1 << P}")
    
    # ---- 3. random HLLSet (use original for testing) ----
    test_bits = original_bits  # Use original instead of random for testing
    print(f"Test HLLSet: {sum(bin(b).count('1') for b in test_bits)} bits set")
    
    # ---- 4. restore order ----
    cover_ids = cover_from_bits(test_bits, tok_occ_id)
    print(f"Cover: {len(cover_ids)} token IDs")
    
    if len(cover_ids) == 0:
        print("No tokens in cover! Check hash function.")
        sys.exit(1)
    
    # Create reverse mapping
    id_to_tok = {uid: tok for (tok, gram, occ), uid in tok_occ_id.items() if gram == "1g"}
    
    pruned_adj, start_id, end_id = prune_with_virtuals(cover_ids, adj, tok_occ_id)
    candidates = generate_orderings(pruned_adj, start_id, end_id, id_to_tok, beam=5, max_len=20)
    
    print("Candidate orderings (highest weight first):")
    for i, cand in enumerate(candidates, 1):
        print(f"{i}. {''.join(cand)}")
    print("=== DEMO END ===")