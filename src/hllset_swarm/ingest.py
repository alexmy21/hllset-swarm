import torch
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from hllset_swarm.hllset_wrapper import HllSet, HllHashInfo


class LookupTable:
    """
    Hash-based lookup table for tokens with collision handling.
    Maps hash_value → token metadata
    """
    def __init__(self):
        self.START = "⊢"
        self.END = "⊣"
        
        # Core mapping: hash_value → token metadata
        self.hash_to_token: Dict[int, str] = {}
        self.token_to_hash: Dict[str, int] = {}
        
        # HLL pair mapping: hash_value → (reg, run)
        self.hash_to_pair: Dict[int, Tuple[int, int]] = {}
        
        # Collision tracking: (reg, run) → set of hash_values
        self.pair_to_hashes: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        
        # Token frequencies
        self.token_frequency: Dict[str, int] = defaultdict(int)
    
    def add_token(self, token: str, hash_info: HllHashInfo):
        """
        Add token to LUT using hash_value as stable ID.
        
        Args:
            token: Token string
            hash_info: HllHashInfo with hash_value, hll_pair, idx
        """
        hash_val = hash_info.hash_value
        pair = hash_info.hll_pair
        
        # Store hash → token mapping
        if hash_val not in self.hash_to_token:
            self.hash_to_token[hash_val] = token
            self.token_to_hash[token] = hash_val
            self.hash_to_pair[hash_val] = pair
        
        # Track HLL collisions
        self.pair_to_hashes[pair].add(hash_val)
        
        # Update frequency
        self.token_frequency[token] += 1
    
    def get_token_by_hash(self, hash_val: int) -> Optional[str]:
        """Get token by its hash value"""
        return self.hash_to_token.get(hash_val)
    
    def get_hash_by_token(self, token: str) -> Optional[int]:
        """Get hash value by token"""
        return self.token_to_hash.get(token)
    
    def get_tokens_by_pair(self, reg: int, run: int) -> List[str]:
        """Get all tokens that map to (reg, run) pair, sorted by frequency"""
        pair = (reg, run)
        if pair not in self.pair_to_hashes:
            return []
        
        tokens = [
            self.hash_to_token[h] 
            for h in self.pair_to_hashes[pair]
            if h in self.hash_to_token
        ]
        
        # Sort by frequency (descending)
        tokens.sort(key=lambda t: self.token_frequency[t], reverse=True)
        return tokens
    
    def get_hll_pair(self, token: str) -> Optional[Tuple[int, int]]:
        """Get (reg, run) pair for token"""
        hash_val = self.token_to_hash.get(token)
        if hash_val is not None:
            return self.hash_to_pair.get(hash_val)
        return None
    
    def get_collision_count(self) -> int:
        """Count HLL pairs with multiple hash values (collisions)"""
        return sum(1 for hashes in self.pair_to_hashes.values() if len(hashes) > 1)
    
    def get_collisions(self) -> Dict[Tuple[int, int], Set[str]]:
        """Get all colliding tokens grouped by HLL pair"""
        collisions = {}
        for pair, hashes in self.pair_to_hashes.items():
            if len(hashes) > 1:
                tokens = {self.hash_to_token[h] for h in hashes if h in self.hash_to_token}
                collisions[pair] = tokens
        return collisions
    
    def merge_from(self, other_lut: 'LookupTable'):
        """Merge another LookupTable into this one"""
        # Merge hash mappings
        self.hash_to_token.update(other_lut.hash_to_token)
        self.token_to_hash.update(other_lut.token_to_hash)
        self.hash_to_pair.update(other_lut.hash_to_pair)
        
        # Merge collision tracking
        for pair, hashes in other_lut.pair_to_hashes.items():
            self.pair_to_hashes[pair].update(hashes)
        
        # Merge frequencies
        for token, freq in other_lut.token_frequency.items():
            self.token_frequency[token] += freq


class CorpusState:
    """
    Maintains global state across multiple corpus ingestions.
    
    Architecture (Hash-Based):
    - self.hash_to_token: Dict[int, str] - hash_value → token (STABLE across sessions)
    - self.token_to_hash: Dict[str, int] - token → hash_value
    - self.edge_freq: Dict[Tuple[int, int], int] - (hash_u, hash_v) → frequency
    - self.hllsets: List[HllSet] - One HLLSet per text
    - self.master_hll: HllSet - Master vocabulary
    
    Key Insight: Hash values are STABLE identifiers that enable:
    1. Consistent indexing across sessions
    2. Simple merging (union of hash-based edges)
    3. Direct DuckDB integration (hash as primary key)
    """
    def __init__(self, P: int = 10):
        self.P = P
        self.lut = LookupTable()
        
        # Use hash values as stable node IDs in adjacency matrix
        self.hash_to_token: Dict[int, str] = {}
        self.token_to_hash: Dict[str, int] = {}
        
        # Edges use hash values as node IDs
        self.edge_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # HLLSets
        self.hllsets: List[HllSet] = []
        self.master_hll = HllSet(P=P)
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize START and END tokens"""
        start_info = self.master_hll.add(self.lut.START)
        end_info = self.master_hll.add(self.lut.END)
        
        self.lut.add_token(self.lut.START, start_info)
        self.lut.add_token(self.lut.END, end_info)
        
        # Store hash mappings
        self.hash_to_token[start_info.hash_value] = self.lut.START
        self.hash_to_token[end_info.hash_value] = self.lut.END
        self.token_to_hash[self.lut.START] = start_info.hash_value
        self.token_to_hash[self.lut.END] = end_info.hash_value
    
    def ingest_corpus(self, corpus: List[str]):
        """
        Ingest corpus: create one HLLSet per text.
        Uses hash values as stable IDs for adjacency matrix.
        """
        print(f"\n=== Ingesting {len(corpus)} texts ===")
        
        for text_idx, text in enumerate(corpus):
            text = text.strip()
            if len(text) == 0:
                continue
            
            # Create NEW HLLSet for THIS text
            text_hll = HllSet(P=self.P)
            text_hll.add(self.lut.START)
            text_hll.add(self.lut.END)
            
            # Create token sequence
            tokens = [self.lut.START] + list(text) + [self.lut.END]
            print(f"Text {text_idx + 1}: {''.join(tokens)}")
            
            # Track all hash values for this text (for n-gram edge creation)
            unigram_hashes = []  # For sequential edges
            
            # Slide 3-token window
            for i in range(len(tokens) - 2):
                print(f"\n{'='*50}")
                print(f"Processing window: tokens[{i}:{i+3}] → '{tokens[i]}', '{tokens[i+1]}', '{tokens[i+2]}'")

                tok_a = tokens[i]
                tok_b = tokens[i + 1]
                tok_c = tokens[i + 2]
                
                # Decompose into n-grams
                unigram = tok_a
                bigram = tok_a + tok_b
                trigram = tok_a + tok_b + tok_c
                
                # Process each n-gram and collect hash values
                for ngram in [unigram, bigram, trigram]:
                    # Add to text HLL
                    text_hash_info = text_hll.add(ngram)
                    print(f"Added to text HLL: '{ngram}' → hash {text_hash_info.hash_value}")
                    
                    # Add to master HLL
                    master_hash_info = self.master_hll.add(ngram)
                    
                    # Use HASH VALUE as stable ID
                    hash_val = master_hash_info.hash_value
                    print(f"Master HLL: '{ngram}' → hash {hash_val}")
                    
                    if hash_val not in self.hash_to_token:
                        self.hash_to_token[hash_val] = ngram
                        self.token_to_hash[ngram] = hash_val
                        self.lut.add_token(ngram, master_hash_info)
                    else:
                        # Token seen before - update frequency
                        self.lut.token_frequency[ngram] += 1
                
                # Store unigram hash for sequential edge creation
                if i == 0 or len(unigram_hashes) <= i:
                    unigram_hash = self.token_to_hash[unigram]
                    unigram_hashes.append(unigram_hash)
            
            # Add the last two unigrams
            if len(tokens) >= 2:
                second_last = tokens[-2]
                last = tokens[-1]
                
                for tok in [second_last, last]:
                    if tok not in self.token_to_hash:
                        info = self.master_hll.add(tok)
                        self.hash_to_token[info.hash_value] = tok
                        self.token_to_hash[tok] = info.hash_value
                        self.lut.add_token(tok, info)
                    
                    unigram_hashes.append(self.token_to_hash[tok])
            
            # BUILD EDGES: Create edges between CONSECUTIVE UNIGRAMS
            # This creates the sequential structure: ⊢ → 人 → 工 → 智 → 能 → ⊣
            for i in range(len(unigram_hashes) - 1):
                hash_u = unigram_hashes[i]
                hash_v = unigram_hashes[i + 1]
                self.edge_freq[(hash_u, hash_v)] += 1
            
            # ALSO BUILD EDGES: Between n-grams of same window (for disambiguation)
            # For each window (a, b, c):
            #   - Create edges: unigram(a) → bigram(a+b) → trigram(a+b+c)
            for i in range(len(tokens) - 2):
                tok_a = tokens[i]
                tok_b = tokens[i + 1]
                tok_c = tokens[i + 2]
                
                unigram = tok_a
                bigram = tok_a + tok_b
                trigram = tok_a + tok_b + tok_c
                
                # Get hash values
                hash_1 = self.token_to_hash[unigram]
                hash_2 = self.token_to_hash[bigram]
                hash_3 = self.token_to_hash[trigram]
                
                # Create hierarchical edges for disambiguation
                self.edge_freq[(hash_1, hash_2)] += 1  # unigram → bigram
                self.edge_freq[(hash_2, hash_3)] += 1  # bigram → trigram
            
            # Store this text's HLLSet
            self.hllsets.append(text_hll)
            print(f"  Text HLL cardinality: {text_hll.count():.0f}")
        
        print(f"\nGlobal state:")
        print(f"  Total unique tokens: {len(self.hash_to_token)}")
        print(f"  Total HLLSets: {len(self.hllsets)}")
        print(f"  Total edges: {len(self.edge_freq)}")
        print(f"  Master HLL cardinality: {self.master_hll.count():.0f}")
    
    def get_hash_to_compact_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create compact index mapping for sparse matrix storage.
        
        Hash values can be large (up to 2^64), so we create:
        - hash_to_compact: hash_value → compact_idx (0, 1, 2, ...)
        - compact_to_hash: compact_idx → hash_value
        
        Returns:
            (hash_to_compact, compact_to_hash)
        """
        unique_hashes = sorted(set(self.hash_to_token.keys()))
        hash_to_compact = {h: i for i, h in enumerate(unique_hashes)}
        compact_to_hash = {i: h for h, i in hash_to_compact.items()}
        return hash_to_compact, compact_to_hash
    
    def get_adjacency_matrix(self) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, int], Dict[int, int]]:
        """
        Build adjacency matrix with compact indices.
        
        Returns:
            (adj, token_to_compact, hash_to_compact, compact_to_hash)
            
            - adj: Sparse tensor with compact indices
            - token_to_compact: token → compact_idx (for disambiguation)
            - hash_to_compact: hash_value → compact_idx (for persistence)
            - compact_to_hash: compact_idx → hash_value (for restoration)
        """
        if not self.edge_freq:
            return (
                torch.sparse_coo_tensor(size=(0, 0)),
                {},
                {},
                {}
            )
        
        # Get all unique hash values involved in edges
        unique_hashes = sorted(set(
            hash_val 
            for (u, v) in self.edge_freq.keys() 
            for hash_val in (u, v)
        ))
        
        # Create compact mapping
        hash_to_compact = {h: i for i, h in enumerate(unique_hashes)}
        compact_to_hash = {i: h for h, i in hash_to_compact.items()}
        
        N = len(unique_hashes)
        
        # Build adjacency matrix with compact indices
        adj_u, adj_v, adj_w = [], [], []
        for (hash_u, hash_v), freq in self.edge_freq.items():
            compact_u = hash_to_compact[hash_u]
            compact_v = hash_to_compact[hash_v]
            adj_u.append(compact_u)
            adj_v.append(compact_v)
            adj_w.append(float(freq))
        
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([adj_u, adj_v], dtype=torch.long),
            values=torch.tensor(adj_w, dtype=torch.float32),
            size=(N, N)
        ).coalesce()
        
        # Create token → compact_idx mapping (for disambiguation)
        token_to_compact = {}
        for token, hash_val in self.token_to_hash.items():
            if hash_val in hash_to_compact:
                token_to_compact[token] = hash_to_compact[hash_val]
        
        return adj, token_to_compact, hash_to_compact, compact_to_hash
    
    def merge_edges_from(self, other_state: 'CorpusState'):
        """
        Merge edges from another CorpusState.
        
        Because we use hash values as IDs, merging is simple:
        just union the edge frequencies!
        """
        for (hash_u, hash_v), freq in other_state.edge_freq.items():
            self.edge_freq[(hash_u, hash_v)] += freq
        
        # Merge token mappings
        self.hash_to_token.update(other_state.hash_to_token)
        self.token_to_hash.update(other_state.token_to_hash)
        
        # Merge LUT
        self.lut.merge_from(other_state.lut)
        
        print(f"Merged {len(other_state.edge_freq)} edges from other state")
    
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
    
    def get_stats(self) -> Dict:
        """Get statistics about current state"""
        return {
            'total_texts': len(self.hllsets),
            'total_tokens': len(self.hash_to_token),
            'total_edges': len(self.edge_freq),
            'master_hll_cardinality': self.master_hll.count(),
            'collision_count': self.lut.get_collision_count(),
            'avg_edge_frequency': sum(self.edge_freq.values()) / len(self.edge_freq) if self.edge_freq else 0,
            'avg_text_hll_cardinality': sum(h.count() for h in self.hllsets) / len(self.hllsets) if self.hllsets else 0,
            'hash_value_range': (min(self.hash_to_token.keys()), max(self.hash_to_token.keys())) if self.hash_to_token else (0, 0)
        }


def ingest_corpus(
    corpus: List[str],
    state: Optional[CorpusState] = None,
    P: int = 10
) -> Tuple[torch.Tensor, Dict[str, int], LookupTable, List[HllSet], CorpusState]:
    """
    Convenience function for corpus ingestion.
    
    Args:
        corpus: List of text strings
        state: Existing CorpusState (for incremental ingestion), or None
        P: HLL precision parameter
    
    Returns:
        (adj, token_to_compact, lut, hllsets, state)
    """
    if state is None:
        state = CorpusState(P=P)
    
    state.ingest_corpus(corpus)
    
    adj, token_to_compact, _, _ = state.get_adjacency_matrix()
    
    return adj, token_to_compact, state.lut, state.hllsets, state