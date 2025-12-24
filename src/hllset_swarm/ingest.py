import torch
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from hllset_swarm.hllset_wrapper import HllSet, HllHashInfo

class LookupTable:
    """
    Hash-based lookup table for tokens with collision handling.
    """
    def __init__(self):
        self.START = "⊢"
        self.END = "⊣"
        self.table: Dict[Tuple[int, int], Dict] = {}
        self.token_to_pair: Dict[str, Tuple[int, int]] = {}
        self.token_frequency: Dict[str, int] = defaultdict(int)  # Track token frequencies
    
    def add_token(self, token: str, hash_info: HllHashInfo) -> Tuple[int, int]:
        """Add token to LUT and update frequency"""
        pair = hash_info.hll_pair
        
        if pair not in self.table:
            self.table[pair] = {
                'tokens': set(),
                'hashes': set()
            }
        
        self.table[pair]['tokens'].add(token)
        self.table[pair]['hashes'].add(hash_info.hash_value)
        self.token_to_pair[token] = pair
        self.token_frequency[token] += 1  # Increment frequency
        
        return pair
    
    def merge_from(self, other_lut: 'LookupTable'):
        """Merge another LookupTable into this one"""
        # Merge tables
        for pair, data in other_lut.table.items():
            if pair not in self.table:
                self.table[pair] = {
                    'tokens': set(),
                    'hashes': set()
                }
            self.table[pair]['tokens'].update(data['tokens'])
            self.table[pair]['hashes'].update(data['hashes'])
        
        # Merge token mappings
        self.token_to_pair.update(other_lut.token_to_pair)
        
        # Merge frequencies
        for token, freq in other_lut.token_frequency.items():
            self.token_frequency[token] += freq
    
    def get_tokens_by_hll(self, reg: int, run: int) -> List[str]:
        """Get tokens ordered by frequency"""
        pair = (reg, run)
        if pair in self.table:
            tokens = list(self.table[pair]['tokens'])
            # Sort by frequency (descending)
            tokens.sort(key=lambda t: self.token_frequency[t], reverse=True)
            return tokens
        return []
    
    def get_hll_pair(self, token: str) -> Optional[Tuple[int, int]]:
        return self.token_to_pair.get(token)
    
    def get_collision_count(self) -> int:
        return sum(1 for data in self.table.values() if len(data['tokens']) > 1)
    
    def get_collisions(self) -> Dict[Tuple[int, int], set]:
        return {
            pair: data['tokens'] 
            for pair, data in self.table.items() 
            if len(data['tokens']) > 1
        }
    
class CorpusState:
    """
    Maintains global state across multiple corpus ingestions.
    
    Architecture:
    - self.hllsets: List[HllSet] - One HLLSet per text (for self-generation)
    - self.master_hll: HllSet - Master vocabulary for consistent hash_idx
    - self.token_to_idx: Dict - Global token → hash_idx mapping
    - self.edge_freq: Dict - Accumulated edge frequencies
    """
    def __init__(self, P: int = 10):
        self.P = P
        self.lut = LookupTable()
        
        # Global mappings for adjacency matrix
        self.token_to_idx: Dict[str, int] = {}
        self.edge_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Collection of HLLSets (one per text) - THIS IS THE KEY
        self.hllsets: List[HllSet] = []
        
        # Master HLLSet for ensuring consistent hash_idx across all texts
        self.master_hll = HllSet(P=P)
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize START and END tokens in master HLL"""
        start_info = self.master_hll.add(self.lut.START)
        end_info = self.master_hll.add(self.lut.END)
        
        self.lut.add_token(self.lut.START, start_info)
        self.lut.add_token(self.lut.END, end_info)
        
        self.token_to_idx[self.lut.START] = start_info.idx
        self.token_to_idx[self.lut.END] = end_info.idx
    
    def ingest_corpus(self, corpus: List[str]):
        """
        Ingest corpus: create one HLLSet per text.
        """
        print(f"\n=== Ingesting {len(corpus)} texts ===")
        
        for text_idx, text in enumerate(corpus):
            text = text.strip()
            if len(text) == 0:
                continue
            
            # Create NEW HLLSet for THIS text (not global!)
            text_hll = HllSet(P=self.P)
            
            # Add special tokens to this text's HLLSet
            text_hll.add(self.lut.START)
            text_hll.add(self.lut.END)
            
            # Create token sequence
            tokens = [self.lut.START] + list(text) + [self.lut.END]
            
            print(f"Text {text_idx + 1}: {''.join(tokens)}")
            
            # Slide 3-token window
            for i in range(len(tokens) - 2):
                tok_a = tokens[i]
                tok_b = tokens[i + 1]
                tok_c = tokens[i + 2]
                
                # Decompose into n-grams
                unigram = tok_a
                bigram = tok_a + tok_b
                trigram = tok_a + tok_b + tok_c
                
                # Process each n-gram
                hash_indices = []
                for ngram in [unigram, bigram, trigram]:
                    # Add to THIS text's HLLSet (for self-generation loop)
                    text_hash_info = text_hll.add(ngram)
                    
                    # Also add to master HLL (for consistent hash_idx)
                    master_hash_info = self.master_hll.add(ngram)
                    
                    # Use MASTER hash_idx for adjacency matrix
                    if ngram not in self.token_to_idx:
                        self.token_to_idx[ngram] = master_hash_info.idx
                        self.lut.add_token(ngram, master_hash_info)
                    else:
                        # Token seen before - just update frequency
                        self.lut.token_frequency[ngram] += 1
                    
                    hash_idx = self.token_to_idx[ngram]
                    hash_indices.append(hash_idx)
                
                # Build edges using stable hash indices
                idx_1, idx_2, idx_3 = hash_indices
                
                # Accumulate edge frequencies
                self.edge_freq[(idx_1, idx_2)] += 1
                self.edge_freq[(idx_2, idx_3)] += 1
            
            # Store this text's HLLSet
            self.hllsets.append(text_hll)
            
            print(f"  Text HLL cardinality: {text_hll.count():.0f}")
        
        print(f"\nGlobal state:")
        print(f"  Total unique tokens: {len(self.token_to_idx)}")
        print(f"  Total HLLSets: {len(self.hllsets)}")
        print(f"  Total edges: {len(self.edge_freq)}")
        print(f"  Master HLL cardinality: {self.master_hll.count():.0f}")
    
    def get_hllset_for_text(self, text_idx: int) -> Optional[HllSet]:
        """Retrieve the HLLSet for a specific text by index"""
        if 0 <= text_idx < len(self.hllsets):
            return self.hllsets[text_idx]
        return None
    
    def get_all_hllsets(self) -> List[HllSet]:
        """Get all HLLSets (one per text)"""
        return self.hllsets.copy()
    
    def get_hllset_count(self) -> int:
        """Get total number of HLLSets (= number of texts processed)"""
        return len(self.hllsets)
    
    def get_master_hll(self) -> HllSet:
        """Get the master HLLSet (contains all unique tokens)"""
        return self.master_hll
    
    def get_adjacency_matrix(self) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Build unified adjacency matrix from accumulated edge frequencies"""
        if not self.edge_freq:
            return torch.sparse_coo_tensor(size=(0, 0)), {}
        
        all_indices = set()
        for (u, v) in self.edge_freq.keys():
            all_indices.add(u)
            all_indices.add(v)
        
        max_idx = max(all_indices)
        N = max_idx + 1
        
        adj_u, adj_v, adj_w = [], [], []
        for (u, v), freq in self.edge_freq.items():
            adj_u.append(u)
            adj_v.append(v)
            adj_w.append(float(freq))
        
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([adj_u, adj_v], dtype=torch.long),
            values=torch.tensor(adj_w, dtype=torch.float32),
            size=(N, N)
        ).coalesce()
        
        return adj, self.token_to_idx
    
    def get_stats(self) -> Dict:
        """Get statistics about current state"""
        return {
            'total_texts': len(self.hllsets),
            'total_tokens': len(self.token_to_idx),
            'total_edges': len(self.edge_freq),
            'master_hll_cardinality': self.master_hll.count(),
            'collision_count': self.lut.get_collision_count(),
            'avg_edge_frequency': sum(self.edge_freq.values()) / len(self.edge_freq) if self.edge_freq else 0,
            'avg_text_hll_cardinality': sum(h.count() for h in self.hllsets) / len(self.hllsets) if self.hllsets else 0
        }
    