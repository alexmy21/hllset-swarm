import torch
from typing import List, Dict, Tuple, Set, Optional
from collections import deque
from hllset_swarm.hllset_wrapper import HllSet
from hllset_swarm.ingest import LookupTable


class AMDisambiguator:
    """
    Adjacency Matrix-guided disambiguation using sliding window algorithm.
    
    Key insight: Use AM structure to guide token extraction, avoiding
    expensive full collection intersections.
    
    Algorithm:
    1. Start from START symbol
    2. Find valid 2-grams: START + {a : a ∈ hll(1-gram) AND START+a ∈ hll(2-gram)}
    3. Extend to 3-grams: 2-gram + {b : b ∈ hll(1-gram) AND 2-gram+b ∈ hll(3-gram)}
    4. Disambiguate via intersection of 1-grams extracted from 2-gram and 3-gram
    5. Shift window and repeat until END
    """
    
    def __init__(self, 
                 hll: HllSet,
                 adj: torch.Tensor,
                 token_to_compact: Dict[str, int],
                 compact_to_hash: Dict[int, int],
                 hash_to_token: Dict[int, str],
                 lut: LookupTable):
        """
        Initialize disambiguator with hash-based mappings.
        
        Args:
            hll: HLLSet to disambiguate
            adj: Adjacency matrix (sparse COO tensor with compact indices)
            token_to_compact: Token → compact_idx mapping
            compact_to_hash: compact_idx → hash_value mapping
            hash_to_token: hash_value → token mapping
            lut: LookupTable for HLL pair lookups
        """
        self.hll = hll
        self.adj = adj.coalesce()  # Ensure coalesced
        self.token_to_compact = token_to_compact
        self.compact_to_hash = compact_to_hash
        self.hash_to_token = hash_to_token
        self.lut = lut
        
        # Build reverse mapping: compact_idx → tokens
        self.compact_to_tokens = {}
        for token, compact_idx in token_to_compact.items():
            if compact_idx not in self.compact_to_tokens:
                self.compact_to_tokens[compact_idx] = set()
            self.compact_to_tokens[compact_idx].add(token)
        
        # Build adjacency lookup for fast edge queries
        self._build_adjacency_lookup()
    
    def _build_adjacency_lookup(self):
        """Build hash map: source_compact_idx -> {target_compact_idx: frequency}"""
        self.adj_lookup = {}
        
        indices = self.adj.indices()
        values = self.adj.values()
        
        for i in range(indices.shape[1]):
            src = indices[0, i].item()
            tgt = indices[1, i].item()
            freq = values[i].item()
            
            if src not in self.adj_lookup:
                self.adj_lookup[src] = {}
            self.adj_lookup[src][tgt] = freq
    
    def in_hll(self, token: str) -> bool:
        """
        Check if token is in HLL.
        
        Uses HLL register/run bits to verify membership.
        """
        pair = self.lut.get_hll_pair(token)
        if not pair:
            return False
        
        reg, run = pair
        
        # Check if this (reg, run) is set in the HLL
        counts = self.hll.hll.counts
        if reg >= len(counts):
            return False
        
        bits = int(counts[reg])
        # run is 1-indexed, so check bit at position (run - 1)
        return bool(bits & (1 << (run - 1)))
    
    def get_neighbors(self, token: str) -> List[Tuple[str, float]]:
        """
        Get all tokens that follow the given token in AM.
        
        Returns:
            List of (neighbor_token, edge_frequency) tuples
        """
        # FIXED: Use token_to_compact instead of token_to_idx
        if token not in self.token_to_compact:
            return []
        
        src_compact_idx = self.token_to_compact[token]
        
        if src_compact_idx not in self.adj_lookup:
            return []
        
        neighbors = []
        for tgt_compact_idx, freq in self.adj_lookup[src_compact_idx].items():
            if tgt_compact_idx in self.compact_to_tokens:
                for tgt_token in self.compact_to_tokens[tgt_compact_idx]:
                    neighbors.append((tgt_token, freq))
        
        return neighbors
    
    def find_valid_2grams(self, curr_1gram: str) -> List[Tuple[str, str]]:
        """
        Find valid 2-grams starting from curr_1gram.
        
        Returns:
            List of (2-gram, next_1gram) tuples
        
        Algorithm:
            {x = curr + a : a ∈ hll(1-gram) AND curr+a ∈ hll(2-gram)}
        """
        valid_2grams = []
        
        # Get all neighbors from AM
        neighbors = self.get_neighbors(curr_1gram)
        
        for neighbor_token, freq in neighbors:
            # Check if neighbor is a 2-gram
            if len(neighbor_token) == 2:
                # Extract the next 1-gram
                if neighbor_token[0] == curr_1gram:
                    next_1gram = neighbor_token[1]
                    
                    # Verify: neighbor_token ∈ hll(2-gram) AND next_1gram ∈ hll(1-gram)
                    if self.in_hll(neighbor_token) and self.in_hll(next_1gram):
                        valid_2grams.append((neighbor_token, next_1gram))
        
        return valid_2grams
    
    def find_valid_3grams(self, bigram: str) -> List[Tuple[str, str]]:
        """
        Find valid 3-grams extending the given 2-gram.
        
        Returns:
            List of (3-gram, last_1gram) tuples
        
        Algorithm:
            {y = bigram + b : b ∈ hll(1-gram) AND bigram+b ∈ hll(3-gram)}
        """
        valid_3grams = []
        
        # Get all neighbors from AM
        neighbors = self.get_neighbors(bigram)
        
        for neighbor_token, freq in neighbors:
            # Check if neighbor is a 3-gram
            if len(neighbor_token) == 3:
                # Extract the last 1-gram
                if neighbor_token[:2] == bigram:
                    last_1gram = neighbor_token[2]
                    
                    # Verify: neighbor_token ∈ hll(3-gram) AND last_1gram ∈ hll(1-gram)
                    if self.in_hll(neighbor_token) and self.in_hll(last_1gram):
                        valid_3grams.append((neighbor_token, last_1gram))
        
        return valid_3grams
    
    def disambiguate(self, max_paths: int = 10) -> List[List[str]]:
        """
        Disambiguate HLL using AM-guided sliding window.
        
        Args:
            max_paths: Maximum number of paths to explore (prevent explosion)
        
        Returns:
            List of possible 1-gram sequences (without START/END symbols)
        """
        START = self.lut.START
        END = self.lut.END
        
        # BFS to handle multiple paths
        # Queue: (current_1gram, collected_1grams, path_trace)
        queue = deque([(START, [], [])])
        valid_sequences = []
        explored_paths = 0
        
        print(f"\n=== Starting AM-guided disambiguation ===")
        print(f"START symbol: '{START}'")
        print(f"END symbol: '{END}'")
        
        while queue and explored_paths < max_paths:
            curr_1gram, collected, path_trace = queue.popleft()
            explored_paths += 1
            
            print(f"\n[Path {explored_paths}] Current: '{curr_1gram}', Collected: {collected}")
            
            # Terminal condition
            if curr_1gram == END:
                print(f"  ✓ Reached END, sequence: {collected}")
                valid_sequences.append(collected)
                continue
            
            # Step 1: Find valid 2-grams
            valid_2grams = self.find_valid_2grams(curr_1gram)
            print(f"  Valid 2-grams: {[bg for bg, _ in valid_2grams]}")
            
            if not valid_2grams:
                print(f"  ✗ Dead end (no valid 2-grams)")
                # Dead end - save if we have collected something
                if collected and curr_1gram != START:
                    valid_sequences.append(collected)
                continue
            
            # Step 2: For each valid 2-gram, find valid 3-grams and disambiguate
            for bigram, next_1gram in valid_2grams:
                print(f"    Exploring 2-gram: '{bigram}' -> next: '{next_1gram}'")
                
                # Find 3-grams
                valid_3grams = self.find_valid_3grams(bigram)
                print(f"      Valid 3-grams: {[tg for tg, _ in valid_3grams]}")
                
                if valid_3grams:
                    # Step 3: Disambiguate via intersection
                    # 1-grams from 2-gram: {next_1gram}
                    bigram_1grams = {next_1gram}
                    
                    # 1-grams from 3-grams: {last_1gram for each trigram}
                    trigram_1grams = {last_1g for _, last_1g in valid_3grams}
                    
                    # Intersection
                    disambiguated = bigram_1grams & trigram_1grams
                    
                    print(f"      2-gram 1-grams: {bigram_1grams}")
                    print(f"      3-gram 1-grams: {trigram_1grams}")
                    print(f"      Intersection: {disambiguated}")
                    
                    if disambiguated:
                        # Take each disambiguated 1-gram (usually single)
                        for selected_1gram in disambiguated:
                            new_collected = collected + [selected_1gram]
                            new_trace = path_trace + [(bigram, [tg for tg, lg in valid_3grams if lg == selected_1gram])]
                            
                            # Step 4: Shift window
                            print(f"      → Shift to '{selected_1gram}'")
                            queue.append((selected_1gram, new_collected, new_trace))
                    else:
                        print(f"      ✗ Empty intersection")
                else:
                    # No 3-gram extension, accept 2-gram result
                    print(f"      No 3-grams, accepting 2-gram result")
                    new_collected = collected + [next_1gram]
                    new_trace = path_trace + [(bigram, [])]
                    
                    queue.append((next_1gram, new_collected, new_trace))
        
        print(f"\n=== Disambiguation complete ===")
        print(f"Explored {explored_paths} paths")
        print(f"Found {len(valid_sequences)} valid sequences")
        
        return valid_sequences
    
    def get_best_sequence(self, sequences: List[List[str]]) -> Optional[List[str]]:
        """
        Select best sequence from multiple candidates.
        
        Strategy: Prefer longer sequences with higher edge frequencies.
        """
        if not sequences:
            return None
        
        # Score each sequence by length and edge frequencies
        scored = []
        for seq in sequences:
            score = len(seq)  # Base score: length
            
            # Add edge frequency bonuses
            for i in range(len(seq) - 1):
                if seq[i] in self.token_to_compact and seq[i+1] in self.token_to_compact:
                    src_compact_idx = self.token_to_compact[seq[i]]
                    tgt_compact_idx = self.token_to_compact[seq[i+1]]
                    
                    if src_compact_idx in self.adj_lookup and tgt_compact_idx in self.adj_lookup[src_compact_idx]:
                        score += self.adj_lookup[src_compact_idx][tgt_compact_idx]
            
            scored.append((score, seq))
        
        # Return highest scoring sequence
        scored.sort(reverse=True)
        return scored[0][1]


def disambiguate_with_am(
    hll: HllSet,
    adj: torch.Tensor,
    token_to_compact: Dict[str, int],
    compact_to_hash: Dict[int, int],
    hash_to_token: Dict[int, str],
    lut: LookupTable,
    max_paths: int = 10
) -> Tuple[List[List[str]], Optional[List[str]]]:
    """
    Convenience function for AM-guided disambiguation with hash-based indices.
    
    Args:
        hll: HLLSet to disambiguate
        adj: Adjacency matrix (sparse COO with compact indices)
        token_to_compact: Token → compact_idx mapping
        compact_to_hash: compact_idx → hash_value mapping
        hash_to_token: hash_value → token mapping
        lut: LookupTable for HLL pair lookups
        max_paths: Maximum paths to explore
    
    Returns:
        (all_sequences, best_sequence)
    """
    disambiguator = AMDisambiguator(
        hll, adj, token_to_compact, compact_to_hash, hash_to_token, lut
    )
    sequences = disambiguator.disambiguate(max_paths=max_paths)
    best = disambiguator.get_best_sequence(sequences)
    
    return sequences, best