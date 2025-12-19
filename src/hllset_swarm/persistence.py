import torch
import duckdb
import pickle
import zstandard as zstd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from hllset_swarm.hllset_wrapper import HllSet, HllHashInfo
from hllset_swarm.ingest import ingest_corpus, LookupTable

class PersistentLookupTable:

    """DuckDB-backed LookupTable with incremental merge support"""
    
    def __init__(self, db_path: str = "lut.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create tables for LUT storage"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lut (
                reg INTEGER,
                run INTEGER,
                token VARCHAR,
                hash_value BIGINT,
                token_length INTEGER,
                frequency INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (reg, run, token)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reg_run ON lut(reg, run)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token ON lut(token)
        """)
    
    def merge_from_memory(self, memory_lut: 'LookupTable'):
        """
        Merge in-memory LUT into persistent storage.
        Updates frequency counts for existing tokens.
        """
        batch_data = []
        
        for (reg, run), data in memory_lut.table.items():
            for token in data['tokens']:
                # Get corresponding hash
                hashes = list(data['hashes'])
                hash_value = hashes[0] if hashes else 0
                
                batch_data.append((
                    reg, run, token, hash_value, len(token)
                ))

        if not batch_data:
            print("No data to merge")
            return
        
        # Use ON CONFLICT to increment frequency
        self.conn.executemany("""
            INSERT INTO lut (reg, run, token, hash_value, token_length, frequency)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT (reg, run, token) 
            DO UPDATE SET 
                frequency = lut.frequency + 1,
                last_updated = NOW()
        """, batch_data)
        
        print(f"Merged {len(batch_data)} token entries into persistent LUT")
    
    def get_tokens_by_hll_pairs(self, hll_pairs: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], List[str]]:
        """
        Batch retrieval: get all tokens for multiple (reg, run) pairs.
        Critical for unambiguation step.
        """
        if not hll_pairs:
            return {}
        
        # Convert to list for SQL IN clause
        pairs_list = list(hll_pairs)
        
        # Create temporary table for efficient join
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_pairs (reg INTEGER, run INTEGER)")
        self.conn.execute("DELETE FROM temp_pairs")
        self.conn.executemany("INSERT INTO temp_pairs VALUES (?, ?)", pairs_list)
        
        result = self.conn.execute("""
            SELECT l.reg, l.run, l.token, l.frequency
            FROM lut l
            INNER JOIN temp_pairs t ON l.reg = t.reg AND l.run = t.run
            ORDER BY l.reg, l.run, l.frequency DESC
        """).fetchall()
        
        # Group by (reg, run)
        tokens_by_pair = defaultdict(list)
        for reg, run, token, freq in result:
            tokens_by_pair[(reg, run)].append(token)
        
        return dict(tokens_by_pair)
    
    def get_tokens_by_hll(self, reg: int, run: int) -> List[str]:
        """Get tokens for single (reg, run) pair, ordered by frequency"""
        result = self.conn.execute("""
            SELECT token FROM lut 
            WHERE reg = ? AND run = ?
            ORDER BY frequency DESC
        """, (reg, run)).fetchall()
        return [row[0] for row in result]
    
    def get_hll_pair(self, token: str) -> Optional[Tuple[int, int]]:
        """Get HLL pair for a token"""
        result = self.conn.execute("""
            SELECT reg, run FROM lut WHERE token = ? LIMIT 1
        """, (token,)).fetchone()
        return tuple(result) if result else None
    
    def get_collisions_stats(self) -> List[Tuple[int, int, int]]:
        """Get collision statistics: (reg, run, token_count)"""
        result = self.conn.execute("""
            SELECT reg, run, COUNT(*) as cnt
            FROM lut
            GROUP BY reg, run
            HAVING cnt > 1
            ORDER BY cnt DESC
        """).fetchall()
        return result
    
    def export_to_parquet(self, path: str):
        """Export LUT to Parquet for archival"""
        self.conn.execute(f"COPY lut TO '{path}' (FORMAT PARQUET)")
    
    def vacuum(self):
        """Optimize database after many merges"""
        self.conn.execute("VACUUM")
    
    def close(self):
        self.conn.close()

class PersistentAdjacencyMatrix:

    """
    Sparse adjacency matrix storage with incremental merge support.
    Uses PyTorch sparse format + Zstd compression.
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.adj_path = self.base_path / "adj_matrix.pt.zst"
        self.tok_id_path = self.base_path / "tok_id.pkl.zst"
    
    def save(self, adj: torch.Tensor, tok_id: Dict[str, int]):
        """Save adjacency matrix and token mapping with compression"""
        # Coalesce and prepare data
        adj = adj.coalesce()
        data = {
            'indices': adj.indices(),
            'values': adj.values(),
            'shape': adj.shape,
            'nnz': adj._nnz()
        }
        
        # Compress and save adjacency
        serialized = pickle.dumps(data)
        compressed = zstd.compress(serialized, level=3)
        with open(self.adj_path, 'wb') as f:
            f.write(compressed)
        
        # Compress and save token mapping
        tok_serialized = pickle.dumps(tok_id)
        tok_compressed = zstd.compress(tok_serialized, level=3)
        with open(self.tok_id_path, 'wb') as f:
            f.write(tok_compressed)
        
        print(f"Saved AM: {adj.shape}, {adj._nnz()} edges, "
              f"compressed to {len(compressed) / 1024:.1f} KB")
    
    def load(self) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, int]]]:
        """Load adjacency matrix and token mapping"""
        if not self.adj_path.exists():
            return None, None
        
        # Load and decompress adjacency
        with open(self.adj_path, 'rb') as f:
            compressed = f.read()
        decompressed = zstd.decompress(compressed)
        data = pickle.loads(decompressed)
        
        adj = torch.sparse_coo_tensor(
            indices=data['indices'],
            values=data['values'],
            size=data['shape']
        ).coalesce()
        
        # Load and decompress token mapping
        with open(self.tok_id_path, 'rb') as f:
            tok_compressed = f.read()
        tok_decompressed = zstd.decompress(tok_compressed)
        tok_id = pickle.loads(tok_decompressed)
        
        print(f"Loaded AM: {adj.shape}, {adj._nnz()} edges")
        
        return adj, tok_id
    
    def merge_from_memory(self, memory_adj: torch.Tensor, memory_tok_id: Dict[str, int]):
        """
        Merge in-memory adjacency matrix with persistent storage.
        
        Strategy:
        1. Load existing AM and tok_id
        2. Align token IDs (create unified mapping)
        3. Merge edge weights (sum frequencies)
        4. Save merged result
        """
        # Load existing data
        existing_adj, existing_tok_id = self.load()
        
        if existing_adj is None:
            # First save, no merge needed
            self.save(memory_adj, memory_tok_id)
            return
        
        print("Merging adjacency matrices...")
        
        # Create unified token mapping
        unified_tok_id = existing_tok_id.copy()
        next_id = len(unified_tok_id)
        
        # Map new tokens to unified IDs
        memory_to_unified = {}
        for token, mem_id in memory_tok_id.items():
            if token in unified_tok_id:
                memory_to_unified[mem_id] = unified_tok_id[token]
            else:
                unified_tok_id[token] = next_id
                memory_to_unified[mem_id] = next_id
                next_id += 1
        
        # Remap memory adjacency indices to unified IDs
        memory_adj = memory_adj.coalesce()
        mem_indices = memory_adj.indices()
        mem_values = memory_adj.values()
        
        remapped_indices = torch.stack([
            torch.tensor([memory_to_unified[idx.item()] for idx in mem_indices[0]]),
            torch.tensor([memory_to_unified[idx.item()] for idx in mem_indices[1]])
        ])
        
        # Combine existing and new adjacencies
        existing_adj = existing_adj.coalesce()
        
        combined_indices = torch.cat([existing_adj.indices(), remapped_indices], dim=1)
        combined_values = torch.cat([existing_adj.values(), mem_values])
        
        # Create merged sparse tensor and coalesce to sum duplicate edges
        merged_size = (len(unified_tok_id), len(unified_tok_id))
        merged_adj = torch.sparse_coo_tensor(
            indices=combined_indices,
            values=combined_values,
            size=merged_size
        ).coalesce()
        
        # Save merged result
        self.save(merged_adj, unified_tok_id)
        
        print(f"Merge complete: {len(unified_tok_id)} tokens, {merged_adj._nnz()} edges")

class CorpusStateManager:
    """
    High-level manager for corpus state persistence and retrieval.
    Handles both LUT and AM, optimized for real-time SGS.ai processing.
    """
    
    def __init__(self, storage_dir: str = "./corpus_state"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.lut = PersistentLookupTable(str(self.storage_dir / "lut.duckdb"))
        self.am = PersistentAdjacencyMatrix(str(self.storage_dir))
    
    def ingest_and_merge(self, corpus: List[str], hll: HllSet):
        """
        Ingest new corpus in memory, then merge with persistent storage.
        This is called after each new data arrival in SGS.ai.
        """
        print("\n=== Ingesting new corpus ===")
        
        # Process in memory (fast)
        memory_adj, memory_tok_id, memory_lut = ingest_corpus(corpus, hll)
        
        print(f"Memory: {len(memory_tok_id)} tokens, {memory_adj._nnz()} edges")
        
        # Merge into persistent storage
        print("\n=== Merging with persistent storage ===")
        self.lut.merge_from_memory(memory_lut)
        self.am.merge_from_memory(memory_adj, memory_tok_id)
        
        print("Merge complete!\n")
    
    def retrieve_for_restoration(self, hllset: HllSet) -> Tuple[torch.Tensor, Dict[str, int], Dict[Tuple[int, int], List[str]]]:
        """
        Retrieve data needed for HLLSet → text restoration.
        
        Steps:
        1. Extract (reg, run) pairs from HLLSet
        2. Batch retrieve tokens from LUT
        3. Unambiguate tokens
        4. Load and prune AM to relevant tokens
        
        Returns:
            pruned_adj: Adjacency matrix with only relevant tokens
            pruned_tok_id: Token ID mapping for pruned matrix
            tokens_by_pair: Token candidates for each (reg, run) pair
        """
        print("\n=== Retrieving data for restoration ===")
        
        # Extract HLLSet pairs (this happens in memory, very fast)
        hll_pairs = self._extract_hll_pairs(hllset)
        print(f"HLLSet contains {len(hll_pairs)} unique (reg, run) pairs")
        
        # Batch retrieve tokens from LUT
        tokens_by_pair = self.lut.get_tokens_by_hll_pairs(hll_pairs)
        print(f"Retrieved token candidates for {len(tokens_by_pair)} pairs")
        
        # Load full AM and tok_id
        full_adj, full_tok_id = self.am.load()
        
        if full_adj is None:
            raise ValueError("No adjacency matrix found in storage")
        
        # Unambiguate and get relevant tokens
        disambiguated_tokens = self._unambiguate(tokens_by_pair, hll_pairs)
        print(f"Disambiguated to {len(disambiguated_tokens)} tokens")
        
        # Prune AM to only relevant tokens
        pruned_adj, pruned_tok_id = self._prune_adjacency(
            full_adj, full_tok_id, disambiguated_tokens
        )
        print(f"Pruned AM: {pruned_adj._nnz()} edges, {len(pruned_tok_id)} nodes")
        
        return pruned_adj, pruned_tok_id, tokens_by_pair
    
    def _extract_hll_pairs(self, hllset: HllSet) -> Set[Tuple[int, int]]:
        """Extract all (reg, run) pairs from HLLSet"""
        pairs = set()
        
        # Access Julia HLL counts
        counts = hllset.hll.counts
        
        for reg in range(len(counts)):
            bits = int(counts[reg])
            if bits == 0:
                continue
            
            # Extract run positions from bits
            for run in range(32):
                if bits & (1 << run):
                    pairs.add((reg, run + 1))  # run is 1-indexed
        
        return pairs
    
    def _unambiguate(self, tokens_by_pair: Dict[Tuple[int, int], List[str]], 
                     hll_pairs: Set[Tuple[int, int]]) -> Set[str]:
        """
        Unambiguate tokens using n-gram decomposition intersection.
        
        T_1 ∩ T_2 ∩ T_3 approach
        """
        # Separate by n-gram length
        tokens_1g = set()
        tokens_2g = []
        tokens_3g = []
        
        for pair in hll_pairs:
            for token in tokens_by_pair.get(pair, []):
                if len(token) == 1:
                    tokens_1g.add(token)
                elif len(token) == 2:
                    tokens_2g.append(token)
                elif len(token) == 3:
                    tokens_3g.append(token)
        
        # Decompose n-grams
        T_1 = tokens_1g
        T_2 = set()
        for bigram in tokens_2g:
            T_2.update(bigram)
        
        T_3 = set()
        for trigram in tokens_3g:
            T_3.update(trigram)
        
        # Intersection
        if T_2 and T_3:
            disambiguated_1g = T_1 & T_2 & T_3
        elif T_2:
            disambiguated_1g = T_1 & T_2
        else:
            disambiguated_1g = T_1
        
        # Include valid n-grams composed of disambiguated 1-grams
        result = set(disambiguated_1g)
        
        for bigram in tokens_2g:
            if all(c in disambiguated_1g for c in bigram):
                result.add(bigram)
        
        for trigram in tokens_3g:
            if all(c in disambiguated_1g for c in trigram):
                result.add(trigram)
        
        # Always include special tokens
        result.add("⊢")
        result.add("⊣")
        
        return result
    
    def _prune_adjacency(self, full_adj: torch.Tensor, full_tok_id: Dict[str, int], 
                        keep_tokens: Set[str]) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Prune adjacency matrix to only keep specified tokens"""
        keep_ids = {full_tok_id[tok] for tok in keep_tokens if tok in full_tok_id}
        
        if not keep_ids:
            return torch.sparse_coo_tensor(size=(0, 0)), {}
        
        # Create new compact ID mapping
        keep_list = sorted(keep_ids)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_list)}
        
        pruned_tok_id = {tok: old_to_new[old_id] 
                        for tok, old_id in full_tok_id.items() 
                        if old_id in keep_ids}
        
        # Filter and remap edges
        full_adj = full_adj.coalesce()
        indices = full_adj.indices()
        values = full_adj.values()
        
        mask = torch.tensor([
            u.item() in keep_ids and v.item() in keep_ids
            for u, v in zip(indices[0], indices[1])
        ])
        
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]
        
        # Remap to compact IDs
        remapped_indices = torch.stack([
            torch.tensor([old_to_new[idx.item()] for idx in filtered_indices[0]]),
            torch.tensor([old_to_new[idx.item()] for idx in filtered_indices[1]])
        ])
        
        pruned_adj = torch.sparse_coo_tensor(
            indices=remapped_indices,
            values=filtered_values,
            size=(len(keep_ids), len(keep_ids))
        ).coalesce()
        
        return pruned_adj, pruned_tok_id
    
    def close(self):
        """Close all connections"""
        self.lut.close()