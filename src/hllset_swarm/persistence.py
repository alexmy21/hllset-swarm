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
    """DuckDB-backed LookupTable with hash-based primary keys"""
    
    def __init__(self, db_path: str = "lut.duckdb"):
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create tables with hash_value as primary key"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lut (
                hash_value BIGINT PRIMARY KEY,
                token VARCHAR NOT NULL,
                reg INTEGER NOT NULL,
                run INTEGER NOT NULL,
                token_length INTEGER NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT current_timestamp
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reg_run ON lut(reg, run)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token ON lut(token)
        """)
        
        # Create edge table with hash-based IDs
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                hash_source BIGINT NOT NULL,
                hash_target BIGINT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_updated TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (hash_source, hash_target)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(hash_source)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(hash_target)
        """)
    
    def save_from_state(self, corpus_state: CorpusState):
        """
        Save LUT and edges from CorpusState using hash values as IDs.
        """
        # Save tokens
        token_batch = []
        for hash_val, token in corpus_state.hash_to_token.items():
            pair = corpus_state.lut.get_hll_pair(token)
            if pair:
                reg, run = pair
                frequency = corpus_state.lut.token_frequency.get(token, 1)
                
                token_batch.append((
                    hash_val, token, reg, run, len(token), frequency
                ))
        
        if token_batch:
            self.conn.executemany("""
                INSERT INTO lut (hash_value, token, reg, run, token_length, frequency)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (hash_value) 
                DO UPDATE SET 
                    frequency = excluded.frequency,
                    last_updated = current_timestamp
            """, token_batch)
            
            print(f"Saved {len(token_batch)} token entries to LUT")
        
        # Save edges
        edge_batch = []
        for (hash_u, hash_v), freq in corpus_state.edge_freq.items():
            edge_batch.append((hash_u, hash_v, freq))
        
        if edge_batch:
            self.conn.executemany("""
                INSERT INTO edges (hash_source, hash_target, frequency)
                VALUES (?, ?, ?)
                ON CONFLICT (hash_source, hash_target)
                DO UPDATE SET
                    frequency = edges.frequency + excluded.frequency,
                    last_updated = current_timestamp
            """, edge_batch)
            
            print(f"Saved {len(edge_batch)} edges to database")
    
    def load_tokens(self) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, Tuple[int, int]]]:
        """
        Load all tokens from database.
        
        Returns:
            (hash_to_token, token_to_hash, hash_to_pair)
        """
        result = self.conn.execute("""
            SELECT hash_value, token, reg, run
            FROM lut
            ORDER BY hash_value
        """).fetchall()
        
        hash_to_token = {}
        token_to_hash = {}
        hash_to_pair = {}
        
        for hash_val, token, reg, run in result:
            hash_to_token[hash_val] = token
            token_to_hash[token] = hash_val
            hash_to_pair[hash_val] = (reg, run)
        
        print(f"Loaded {len(hash_to_token)} tokens from LUT")
        
        return hash_to_token, token_to_hash, hash_to_pair
    
    def load_edges(self) -> Dict[Tuple[int, int], int]:
        """
        Load all edges from database.
        
        Returns:
            edge_freq: (hash_u, hash_v) → frequency
        """
        result = self.conn.execute("""
            SELECT hash_source, hash_target, frequency
            FROM edges
        """).fetchall()
        
        edge_freq = {}
        for hash_u, hash_v, freq in result:
            edge_freq[(hash_u, hash_v)] = freq
        
        print(f"Loaded {len(edge_freq)} edges from database")
        
        return edge_freq
    
    def get_stats(self) -> Dict:
        """Get LUT and edge statistics"""
        stats = {}
        
        result = self.conn.execute("SELECT COUNT(*) FROM lut").fetchone()
        stats['total_tokens'] = result[0] if result else 0
        
        result = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        stats['total_edges'] = result[0] if result else 0
        
        result = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT reg, run FROM lut GROUP BY reg, run HAVING COUNT(*) > 1
            )
        """).fetchone()
        stats['collision_count'] = result[0] if result else 0
        
        result = self.conn.execute("SELECT SUM(frequency) FROM edges").fetchone()
        stats['total_edge_weight'] = result[0] if result else 0
        
        return stats
    
    def export_to_parquet(self, output_dir: str):
        """Export tables to Parquet for archival"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.conn.execute(f"COPY lut TO '{output_path / 'lut.parquet'}' (FORMAT PARQUET)")
        self.conn.execute(f"COPY edges TO '{output_path / 'edges.parquet'}' (FORMAT PARQUET)")
        
        print(f"Exported to {output_path}")
    
    def vacuum(self):
        """Optimize database"""
        self.conn.execute("VACUUM")
    
    def close(self):
        self.conn.close()


class PersistentAdjacencyMatrix:
    """Sparse adjacency matrix storage with hash→compact mapping"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.adj_path = self.base_path / "adj_matrix.pt.zst"
        self.mapping_path = self.base_path / "hash_mapping.pkl.zst"
    
    def save_from_state(self, corpus_state: CorpusState):
        """
        Save adjacency matrix with hash→compact mapping.
        """
        adj, token_to_compact, hash_to_compact, compact_to_hash = corpus_state.get_adjacency_matrix()
        
        if adj.shape[0] == 0:
            print("No adjacency matrix to save")
            return
        
        # Coalesce and prepare adjacency data
        adj = adj.coalesce()
        adj_data = {
            'indices': adj.indices(),
            'values': adj.values(),
            'shape': adj.shape,
            'nnz': adj._nnz()
        }
        
        # Compress and save adjacency
        serialized = pickle.dumps(adj_data)
        compressed = zstd.compress(serialized, level=3)
        with open(self.adj_path, 'wb') as f:
            f.write(compressed)
        
        # Save hash mappings
        mapping_data = {
            'hash_to_compact': hash_to_compact,
            'compact_to_hash': compact_to_hash,
            'token_to_compact': token_to_compact
        }
        
        mapping_serialized = pickle.dumps(mapping_data)
        mapping_compressed = zstd.compress(mapping_serialized, level=3)
        with open(self.mapping_path, 'wb') as f:
            f.write(mapping_compressed)
        
        print(f"Saved AM: {adj.shape}, {adj._nnz()} edges, "
              f"compressed to {len(compressed) / 1024:.1f} KB")
    
    def load(self) -> Tuple[Optional[torch.Tensor], Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        Load adjacency matrix and mappings.
        
        Returns:
            (adj, token_to_compact, hash_to_compact, compact_to_hash)
        """
        if not self.adj_path.exists():
            return None, None, None, None
        
        # Load adjacency
        with open(self.adj_path, 'rb') as f:
            compressed = f.read()
        decompressed = zstd.decompress(compressed)
        adj_data = pickle.loads(decompressed)
        
        adj = torch.sparse_coo_tensor(
            indices=adj_data['indices'],
            values=adj_data['values'],
            size=adj_data['shape']
        ).coalesce()
        
        # Load mappings
        with open(self.mapping_path, 'rb') as f:
            mapping_compressed = f.read()
        mapping_decompressed = zstd.decompress(mapping_compressed)
        mapping_data = pickle.loads(mapping_decompressed)
        
        print(f"Loaded AM: {adj.shape}, {adj._nnz()} edges")
        
        return (
            adj,
            mapping_data['token_to_compact'],
            mapping_data['hash_to_compact'],
            mapping_data['compact_to_hash']
        )


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
        
        hllsets_data = {
            'count': len(corpus_state.hllsets),
            'P': corpus_state.P,
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
        
        hllsets = []
        for counts in data['hll_counts']:
            hll = HllSet(P=data['P'])
            hll.hll.counts = counts
            hllsets.append(hll)
        
        print(f"Loaded {len(hllsets)} HLLSets")
        
        return hllsets


class PersistenceManager:
    """
    Hash-based persistence manager for SGS.ai.
    
    Key features:
    - Hash values as stable IDs (consistent across sessions)
    - DuckDB stores both tokens and edges
    - Simple merge: just union hash-based edges
    """
    
    def __init__(self, storage_dir: str = "./sgs_state"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.lut = PersistentLookupTable(str(self.storage_dir / "lut.duckdb"))
        self.am = PersistentAdjacencyMatrix(str(self.storage_dir))
        self.hllsets = HLLSetArchive(str(self.storage_dir))
    
    def save(self, corpus_state: CorpusState):
        """Save complete corpus state"""
        print("\n=== Saving corpus state (hash-based) ===")
        
        self.lut.save_from_state(corpus_state)
        self.am.save_from_state(corpus_state)
        self.hllsets.save_from_state(corpus_state)
        self._save_metadata(corpus_state)
        
        print("Save complete!\n")
    
    def load(self, P: int = 10) -> Optional[CorpusState]:
        """Load complete corpus state"""
        print("\n=== Loading corpus state (hash-based) ===")
        
        metadata = self._load_metadata()
        if metadata is None:
            print("No saved state found")
            return None
        
        # Create new CorpusState
        corpus_state = CorpusState(P=metadata['P'])
        
        # Load tokens from DuckDB
        hash_to_token, token_to_hash, hash_to_pair = self.lut.load_tokens()
        corpus_state.hash_to_token = hash_to_token
        corpus_state.token_to_hash = token_to_hash
        
        # Reconstruct LUT
        corpus_state.lut.hash_to_token = hash_to_token
        corpus_state.lut.token_to_hash = token_to_hash
        corpus_state.lut.hash_to_pair = hash_to_pair
        
        # Load edges from DuckDB
        edge_freq = self.lut.load_edges()
        corpus_state.edge_freq = edge_freq
        
        # Load HLLSets
        hllsets = self.hllsets.load(P=metadata['P'])
        if hllsets is not None:
            corpus_state.hllsets = hllsets
        
        print(f"Loaded state: {len(corpus_state.hllsets)} texts, "
              f"{len(corpus_state.hash_to_token)} tokens, "
              f"{len(corpus_state.edge_freq)} edges")
        print("Load complete!\n")
        
        return corpus_state
    
    def _save_metadata(self, corpus_state: CorpusState):
        """Save metadata"""
        metadata = {
            'P': corpus_state.P,
            'total_texts': len(corpus_state.hllsets),
            'total_tokens': len(corpus_state.hash_to_token),
            'total_edges': len(corpus_state.edge_freq),
            'master_hll_cardinality': corpus_state.master_hll.count(),
            'hash_value_range': corpus_state.get_stats()['hash_value_range']
        }
        
        with open(self.storage_dir / "metadata.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata"""
        metadata_path = self.storage_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            import json
            return json.load(f)
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {}
        
        # DuckDB stats
        stats['duckdb'] = self.lut.get_stats()
        
        # File sizes
        if self.am.adj_path.exists():
            stats['adj_size_kb'] = self.am.adj_path.stat().st_size / 1024
        if self.am.mapping_path.exists():
            stats['mapping_size_kb'] = self.am.mapping_path.stat().st_size / 1024
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