import torch
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from hllset_swarm.hllset_wrapper import HllSet, HllHashInfo


class LookupTable:

    """
    Hash-based lookup table for tokens with collision handling.
    
    Structure:
        hll_pair (reg, run) -> {
            'tokens': set of tokens with this (reg, run),
            'hashes': set of original hash values
        }
    """
    def __init__(self):
        self.table: Dict[Tuple[int, int], Dict] = {}
        self.token_to_pair: Dict[str, Tuple[int, int]] = {}
    
    def add_token(self, token: str, hash_info: HllHashInfo) -> Tuple[int, int]:
        """
        Add token to LUT using HllHashInfo.
        
        Args:
            token: Token string
            hash_info: HllHashInfo from HllSet.add()
        
        Returns:
            (reg, run) pair
        """
        pair = hash_info.hll_pair  # (bin, idx)
        
        if pair not in self.table:
            self.table[pair] = {
                'tokens': set(),  # Changed to set
                'hashes': set()   # Changed to set
            }
        
        # Set.add() is idempotent, no need to check membership
        self.table[pair]['tokens'].add(token)
        self.table[pair]['hashes'].add(hash_info.hash_value)
        self.token_to_pair[token] = pair
        
        return pair
    
    def get_tokens_by_hll(self, reg: int, run: int) -> List[str]:
        """Get all tokens that map to this (reg, run) pair"""
        pair = (reg, run)
        if pair in self.table:
            return list(self.table[pair]['tokens'])  # Convert set to list for return
        return []
    
    def get_hll_pair(self, token: str) -> Optional[Tuple[int, int]]:
        """Get HLLSet representation for token"""
        return self.token_to_pair.get(token)
    
    def get_collision_count(self) -> int:
        """Return number of (reg, run) pairs with multiple tokens"""
        return sum(1 for data in self.table.values() if len(data['tokens']) > 1)
    
    def get_collisions(self) -> Dict[Tuple[int, int], set]:
        """Return all (reg, run) pairs that have collisions"""
        return {
            pair: data['tokens'] 
            for pair, data in self.table.items() 
            if len(data['tokens']) > 1
        }

# HllHashInfo and HllSet are assumed to be defined elsewhere

def ingest_corpus(corpus: List[str], hll: HllSet) -> Tuple[torch.Tensor, Dict[str, int], LookupTable]:
    """
    Ingest corpus with proper START/END handling and sliding 3-token window.
    
    Algorithm:
    1. For each text: wrap with START+text+END (START and END are single tokens)
    2. Slide 3-token window over extended text with step=1
    3. Each 3-token window (a,b,c) decomposes into: {(a), (a,b), (a,b,c)}
       - 1-token (unigram): a
       - 2-token (bigram): ab
       - 3-token (trigram): abc
    4. Build adjacency matrix and lookup table
    
    Args:
        corpus: List of text strings (each text is a sequence of tokens/characters)
        hll: HllSet instance for encoding tokens
    
    Returns:
        adj: Sparse adjacency matrix with frequency edges
        tok_id: token_sequence → AM node ID mapping
        lut: LookupTable with token_sequence → HLL mappings
    """
    START = "⊢"  # Unicode start symbol (single token)
    END = "⊣"    # Unicode end symbol (single token)
    
    lut = LookupTable()
    tok_id = {}
    edge_freq = defaultdict(int)
    
    # Register special tokens (they are single tokens like any other character)
    start_info = hll.add(START)
    end_info = hll.add(END)
    lut.add_token(START, start_info)
    lut.add_token(END, end_info)
    tok_id[START] = 0
    tok_id[END] = 1
    
    for text in corpus:
        text = text.strip()
        if len(text) == 0:
            continue
        
        # Create token sequence: [START, tok1, tok2, ..., tokN, END]
        tokens = [START] + list(text) + [END]
        
        # Slide 3-token window over token sequence (step=1)
        for i in range(len(tokens) - 2):
            # Extract 3-token window
            tok_a = tokens[i]      # Token at position i
            tok_b = tokens[i + 1]  # Token at position i+1
            tok_c = tokens[i + 2]  # Token at position i+2
            
            # Decompose into n-token sequences (n-grams)
            unigram = tok_a              # 1-token sequence
            bigram = tok_a + tok_b       # 2-token sequence
            trigram = tok_a + tok_b + tok_c  # 3-token sequence
            
            # Register all n-token sequences in HLL, LUT, and AM
            for ngram in [unigram, bigram, trigram]:
                if ngram not in tok_id:
                    # Add to HLL and get hash info
                    hash_info = hll.add(ngram)
                    # Register in LUT
                    lut.add_token(ngram, hash_info)
                    # Assign AM node ID
                    tok_id[ngram] = len(tok_id)
            
            # Build edges: unigram → bigram → trigram
            id_1 = tok_id[unigram]
            id_2 = tok_id[bigram]
            id_3 = tok_id[trigram]
            
            edge_freq[(id_1, id_2)] += 1
            edge_freq[(id_2, id_3)] += 1
    
    # Build sparse adjacency matrix
    N = len(tok_id)
    adj_u, adj_v, adj_w = [], [], []
    
    for (u, v), freq in edge_freq.items():
        adj_u.append(u)
        adj_v.append(v)
        adj_w.append(float(freq))
    
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([adj_u, adj_v], dtype=torch.long),
        values=torch.tensor(adj_w, dtype=torch.float32),
        size=(N, N)
    ).coalesce()
    
    return adj, tok_id, lut