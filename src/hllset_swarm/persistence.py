import torch
import duckdb
import pickle
import zstandard as zstd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict
from hllset_swarm.hllset_wrapper import HllSet
from hllset_swarm.ingest import CorpusState, LookupTable


class PersistentLookupTable:
    """DuckDB-backed LookupTable with incremental append support"""
    
    def __init__(self, db_path: str = "lut.duckdb"):
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
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
                last_updated TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (reg, run, token)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reg_run ON lut(reg, run)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token ON lut(token)
        """)
    
    def save_from_state(self, corpus_state: CorpusState):
        """
        Save LUT from CorpusState to persistent storage.
        Simplified: just append/update from current state.
        """
        batch_data = []
        
        for (reg, run), data in corpus_state.lut.table.items():
            for token in data['tokens']:
                hashes = list(data['hashes'])
                hash_value = hashes[0] if hashes else 0
                frequency = corpus_state.lut.token_frequency.get(token, 1)
                
                batch_data.append((
                    reg, run, token, hash_value, len(token), frequency
                ))
        
        if not batch_data:
            print("No LUT data to save")
            return
        
        # Use ON CONFLICT to update frequency
        self.conn.executemany("""
            INSERT INTO lut (reg, run, token, hash_value, token_length, frequency)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (reg, run, token) 
            DO UPDATE SET 
                frequency = frequency + 1,
                last_updated = excluded.last_updated
        """, batch_data)
        
        print(f"Saved {len(batch_data)} token entries to LUT")
    
    def get_tokens_by_hll_pairs(self, hll_pairs: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], List[str]]:
        """Batch retrieval: get all tokens for multiple (reg, run) pairs"""
        if not hll_pairs:
            return {}
        
        pairs_list = list(hll_pairs)
        
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_pairs (reg INTEGER, run INTEGER)")
        self.conn.execute("DELETE FROM temp_pairs")
        self.conn.executemany("INSERT INTO temp_pairs VALUES (?, ?)", pairs_list)
        
        result = self.conn.execute("""
            SELECT l.reg, l.run, l.token, l.frequency
            FROM lut l
            INNER JOIN temp_pairs t ON l.reg = t.reg AND l.run = t.run
            ORDER BY l.reg, l.run, l.frequency DESC
        """).fetchall()
        
        tokens_by_pair = defaultdict(list)
        for reg, run, token, freq in result:
            tokens_by_pair[(reg, run)].append(token)
        
        return dict(tokens_by_pair)
    
    def get_stats(self) -> Dict:
        """Get LUT statistics"""
        stats = {}
        
        result = self.conn.execute("SELECT COUNT(*) FROM lut").fetchone()
        stats['total_entries'] = result[0] if result else 0
        
        result = self.conn.execute(
            "SELECT COUNT(DISTINCT reg || ',' || run) FROM lut"
        ).fetchone()
        stats['unique_pairs'] = result[0] if result else 0
        
        result = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT reg, run FROM lut GROUP BY reg, run HAVING COUNT(*) > 1
            )
        """).fetchone()
        stats['collision_count'] = result[0] if result else 0
        
        return stats
    
    def export_to_parquet(self, path: str):
        """Export LUT to Parquet for archival"""
        self.conn.execute(f"COPY lut TO '{path}' (FORMAT PARQUET)")
    
    def vacuum(self):
        """Optimize database"""
        self.conn.execute("VACUUM")
    
    def close(self):
        self.conn.close()


class PersistentAdjacencyMatrix:
    """Sparse adjacency matrix storage with Zstd compression"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.adj_path = self.base_path / "adj_matrix.pt.zst"
        self.tok_id_path = self.base_path / "token_to_idx.pkl.zst"
    
    def save_from_state(self, corpus_state: CorpusState):
        """
        Save adjacency matrix from CorpusState.
        Simplified: just save the current state's adjacency.
        """
        adj, token_to_idx = corpus_state.get_adjacency_matrix()
        
        if adj.shape[0] == 0:
            print("No adjacency matrix to save")
            return
        
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
        tok_serialized = pickle.dumps(token_to_idx)
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


class HLLSetArchive:
    """Archive for storing HLLSets (one per text)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.hllsets_path = self.base_path / "hllsets.pkl.zst"
    
    def save_from_state(self, corpus_state: CorpusState):
        """Save all HLLSets from CorpusState"""
        if not corpus_state.hllsets:
            print("No HLLSets to save")
            return
        
        # Serialize HLLSets
        # Note: HllSet needs to be serializable (Julia objects may need special handling)
        hllsets_data = {
            'count': len(corpus_state.hllsets),
            'P': corpus_state.P,
            # Store HLL counts arrays instead of full Julia objects
            'hll_counts': [hll.hll.counts for hll in corpus_state.hllsets]
        }
        
        serialized = pickle.dumps(hllsets_data)
        compressed = zstd.compress(serialized, level=3)
        
        with open(self.hllsets_path, 'wb') as f:
            f.write(compressed)
        
        print(f"Saved {len(corpus_state.hllsets)} HLLSets, "
              f"compressed to {len(compressed) / 1024:.1f} KB")
    
    def load(self, P: int = 10) -> Optional[List[HllSet]]:
        """Load HLLSets and reconstruct them"""
        if not self.hllsets_path.exists():
            return None
        
        with open(self.hllsets_path, 'rb') as f:
            compressed = f.read()
        
        decompressed = zstd.decompress(compressed)
        data = pickle.loads(decompressed)
        
        # Reconstruct HLLSets from counts
        hllsets = []
        for counts in data['hll_counts']:
            hll = HllSet(P=data['P'])
            # Restore counts (this may need Julia interop)
            hll.counts = counts
            hllsets.append(hll)
        
        print(f"Loaded {len(hllsets)} HLLSets")
        
        return hllsets


class PersistenceManager:
    """
    Simplified persistence manager for SGS.ai iterations.
    
    Usage after each iteration:
        pm = PersistenceManager("./sgs_state")
        pm.save(corpus_state)
        
    Usage for restoration:
        pm = PersistenceManager("./sgs_state")
        corpus_state = pm.load()
    """
    
    def __init__(self, storage_dir: str = "./sgs_state"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.lut = PersistentLookupTable(str(self.storage_dir / "lut.duckdb"))
        self.am = PersistentAdjacencyMatrix(str(self.storage_dir))
        self.hllsets = HLLSetArchive(str(self.storage_dir))
    
    def save(self, corpus_state: CorpusState):
        """
        Save complete corpus state after iteration.
        This is the only method you need to call!
        """
        print("\n=== Saving corpus state ===")
        
        # Save LUT
        self.lut.save_from_state(corpus_state)
        
        # Save adjacency matrix
        self.am.save_from_state(corpus_state)
        
        # Save HLLSets
        self.hllsets.save_from_state(corpus_state)
        
        # Save metadata
        self._save_metadata(corpus_state)
        
        print("Save complete!\n")
    
    def load(self, P: int = 10) -> Optional[CorpusState]:
        """
        Load complete corpus state for restoration.
        
        Returns:
            CorpusState or None if no saved state exists
        """
        print("\n=== Loading corpus state ===")
        
        # Check if state exists
        metadata = self._load_metadata()
        if metadata is None:
            print("No saved state found")
            return None
        
        # Create new CorpusState
        corpus_state = CorpusState(P=metadata['P'])
        
        # Load adjacency matrix
        adj, token_to_idx = self.am.load()
        if adj is not None:
            corpus_state.token_to_idx = token_to_idx
            # Rebuild edge_freq from adjacency matrix
            corpus_state.edge_freq = self._rebuild_edge_freq(adj, token_to_idx)
        
        # Load HLLSets
        hllsets = self.hllsets.load(P=metadata['P'])
        if hllsets is not None:
            corpus_state.hllsets = hllsets
        
        # Note: LUT is loaded on-demand from DuckDB, no need to load into memory
        
        print(f"Loaded state: {len(corpus_state.hllsets)} texts, "
              f"{len(corpus_state.token_to_idx)} tokens")
        print("Load complete!\n")
        
        return corpus_state
    
    def _save_metadata(self, corpus_state: CorpusState):
        """Save corpus state metadata"""
        metadata = {
            'P': corpus_state.P,
            'total_texts': len(corpus_state.hllsets),
            'total_tokens': len(corpus_state.token_to_idx),
            'total_edges': len(corpus_state.edge_freq),
            'master_hll_cardinality': corpus_state.master_hll.count()
        }
        
        with open(self.storage_dir / "metadata.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load corpus state metadata"""
        metadata_path = self.storage_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            import json
            return json.load(f)
    
    def _rebuild_edge_freq(self, adj: torch.Tensor, 
                          token_to_idx: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        """Rebuild edge_freq from adjacency matrix"""
        edge_freq = defaultdict(int)
        
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        
        for i in range(indices.shape[1]):
            u = indices[0, i].item()
            v = indices[1, i].item()
            freq = int(values[i].item())
            edge_freq[(u, v)] = freq
        
        return dict(edge_freq)
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {}
        
        # LUT stats
        stats['lut'] = self.lut.get_stats()
        
        # File sizes
        if self.am.adj_path.exists():
            stats['adj_size_kb'] = self.am.adj_path.stat().st_size / 1024
        if self.am.tok_id_path.exists():
            stats['tok_id_size_kb'] = self.am.tok_id_path.stat().st_size / 1024
        if self.hllsets.hllsets_path.exists():
            stats['hllsets_size_kb'] = self.hllsets.hllsets_path.stat().st_size / 1024
        
        # Metadata
        metadata = self._load_metadata()
        if metadata:
            stats['metadata'] = metadata
        
        return stats
    
    def close(self):
        """Close all connections"""
        self.lut.close()