"""
HllSet Wrapper for Julia HllSets integration.
Provides BSS_τ (coverage) and BSS_ρ (exclusion) metrics for set operations.
"""

from julia import Main
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

# Auto-detect HllSets.jl path if not set
hllsets_path = os.getenv("HLLSETS_PATH")

if not hllsets_path:
    # Try to find HllSets.jl relative to this file
    current_dir = Path(__file__).parent
    hllsets_jl = current_dir / "HllSets.jl"
    
    if hllsets_jl.exists():
        hllsets_path = str(hllsets_jl)
    else:
        raise EnvironmentError(
            f"HLLSETS_PATH environment variable is not set and HllSets.jl not found at {hllsets_jl}"
        )

# Load the HllSets.jl file
Main.include(hllsets_path)
Main.using(".HllSets")


@dataclass
class BSSMetrics:
    """BSS metrics for HllSet operations."""
    tau: float  # Coverage: |A∩B| / |B|
    rho: float  # Exclusion: |A∖B| / |B|


@dataclass
class HllHashInfo:
    """Hash information returned by add! operation."""
    token: Any       # Original element/token
    hash_value: int  # Full hash value
    bin: int         # Register/bin number (reg_num)
    idx: int         # Run of zeros (run_zero)
    
    @property
    def hll_pair(self) -> Tuple[int, int]:
        """Return (reg_num, run_zero) pair for LUT integration."""
        return (self.bin, self.idx)


class HllSet:
    """
    Python wrapper for Julia HllSet with BSS metrics support.
    
    BSS_τ (tau): Coverage metric = |A∩B| / |B|
    BSS_ρ (rho): Exclusion metric = |A∖B| / |B|
    """
    
    def __init__(self, P: int = 10, tau: float = 0.7, rho: float = 0.21, seed: int = 42):
        """
        Initialize an HllSet with precision P and BSS metrics.
        
        Args:
            P: Precision parameter (number of bits for indexing)
            tau: Initial coverage metric (default: 0.7)
            rho: Initial exclusion metric (default: 0.21)
            seed: Random seed for reproducibility
        """
        self.P = P
        self.tau = tau
        self.rho = rho
        self.seed = seed
        self.hll = Main.HllSet(P)  # Create a new HllSet in Julia

    def add(self, element: Any) -> HllHashInfo:
        """
        Add an element to the HllSet and return hash information.
        
        Args:
            element: Element to add
            
        Returns:
            HllHashInfo containing hash value, bin, and idx
        """
        add_func = getattr(Main, "add!")
        token, h, bin, idx = add_func(self.hll, element, seed=self.seed)
        return HllHashInfo(token=token, hash_value=int(h), bin=int(bin), idx=int(idx))

    def add_batch(self, elements: list) -> List[HllHashInfo]:
        """
        Add a batch of elements to the HllSet and return their hash information.
        
        Args:
            elements: List of elements to add
            
        Returns:
            List of HllHashInfo for each added element
        """
        add_func = getattr(Main, "add!")
        results = []
        for element in elements:
            token, h, bin, idx = add_func(self.hll, element, seed=self.seed)
            results.append(HllHashInfo(token=token, hash_value=int(h), bin=int(bin), idx=int(idx)))
        return results

    def add_with_hash(self, element: Any) -> Tuple[HllHashInfo, None]:
        """
        Add an element and return its hash info (alias for add).
        
        Deprecated: Use add() directly.
        """
        return self.add(element), None

    def count(self) -> float:
        """Estimate the cardinality of the HllSet."""
        return float(Main.count(self.hll))

    def _calculate_bss_metrics(self, other: 'HllSet') -> BSSMetrics:
        """
        Calculate BSS metrics for operations with another HllSet.
        
        BSS_τ(A→B) = |A∩B| / |B| (Coverage)
        BSS_ρ(A→B) = |A∖B| / |B| (Exclusion)
        
        Args:
            other: Another HllSet
            
        Returns:
            BSSMetrics with calculated tau and rho
        """
        count_b = float(other.count())
        
        if count_b == 0:
            return BSSMetrics(tau=0.0, rho=0.0)
        
        # Calculate intersection - intersect returns an HllSet, need to count it
        intersection_hll = Main.intersect(self.hll, other.hll)
        intersection_count = float(Main.count(intersection_hll))
        
        # Calculate difference - returns (deleted, retained, new)
        deleted_hll, retained_hll, new_hll = Main.diff(self.hll, other.hll)
        deleted_count = float(Main.count(deleted_hll))
        
        # BSS_τ(A→B) = |A∩B| / |B|
        tau = intersection_count / count_b
        
        # BSS_ρ(A→B) = |A∖B| / |B|
        rho = deleted_count / count_b
        
        return BSSMetrics(tau=tau, rho=rho)

    def union(self, other: 'HllSet') -> 'HllSet':
        """
        Perform union with another HllSet.
        
        Metrics: tau = min(tau_A, tau_B), rho = max(rho_A, rho_B)
        
        Args:
            other: Another HllSet
            
        Returns:
            New HllSet with union result and combined metrics
        """
        result = Main.union(self.hll, other.hll)
        union_set = HllSet.from_julia(result, self.P)
        
        # Calculate combined metrics
        union_set.tau = min(self.tau, other.tau)
        union_set.rho = max(self.rho, other.rho)
        
        return union_set

    def intersection(self, other: 'HllSet') -> 'HllSet':
        """
        Perform intersection with another HllSet.
        
        Metrics: tau = min(tau_A, tau_B), rho = max(rho_A, rho_B)
        
        Args:
            other: Another HllSet
            
        Returns:
            New HllSet with intersection result and combined metrics
        """
        result = Main.intersect(self.hll, other.hll)
        intersect_set = HllSet.from_julia(result, self.P)
        
        # Calculate combined metrics
        intersect_set.tau = min(self.tau, other.tau)
        intersect_set.rho = max(self.rho, other.rho)
        
        return intersect_set

    def difference(self, other: 'HllSet') -> Tuple['HllSet', 'HllSet', 'HllSet']:
        """
        Perform difference with another HllSet.
        
        Metrics for each result: tau = min(tau_A, tau_B), rho = max(rho_A, rho_B)
        
        Args:
            other: Another HllSet
            
        Returns:
            Tuple of (deleted, retained, new) HllSets with combined metrics
        """
        deleted, retained, new = Main.diff(self.hll, other.hll)
        
        # Create HllSets from Julia results
        deleted_set = HllSet.from_julia(deleted, self.P)
        retained_set = HllSet.from_julia(retained, self.P)
        new_set = HllSet.from_julia(new, self.P)
        
        # Apply combined metrics to all results
        combined_tau = min(self.tau, other.tau)
        combined_rho = max(self.rho, other.rho)
        
        for result_set in [deleted_set, retained_set, new_set]:
            result_set.tau = combined_tau
            result_set.rho = combined_rho
        
        return deleted_set, retained_set, new_set

    def complement(self, other: 'HllSet') -> 'HllSet':
        """
        Perform complement operation with another HllSet.
        
        Args:
            other: Another HllSet
            
        Returns:
            New HllSet with complement result
        """
        result = Main.set_comp(self.hll, other.hll)
        comp_set = HllSet.from_julia(result, self.P)
        
        # Calculate BSS metrics for complement
        metrics = self._calculate_bss_metrics(other)
        comp_set.tau = metrics.tau
        comp_set.rho = metrics.rho
        
        return comp_set

    def calculate_bss_to(self, other: 'HllSet') -> BSSMetrics:
        """
        Calculate BSS metrics from this set to another.
        
        BSS_τ(A→B) = |A∩B| / |B| (Coverage)
        BSS_ρ(A→B) = |A∖B| / |B| (Exclusion)
        
        Args:
            other: Target HllSet
            
        Returns:
            BSSMetrics with tau and rho values
        """
        return self._calculate_bss_metrics(other)

    def id(self) -> str:
        """Get SHA1 hash of the HllSet counts."""
        return Main.id(self.hll)

    def __eq__(self, other: Any) -> bool:
        """Compare two HllSets for equality."""
        if not isinstance(other, HllSet):
            return False
        return Main.isequal(self.hll, other.hll)

    def to_binary_tensor(self):
        """Convert the HllSet to a binary tensor."""
        return Main.to_binary_tensor(self.hll)

    @classmethod
    def from_dict(cls, redis_data: Dict[Any, Any], P: int = 10, 
                  tau: float = 0.7, rho: float = 0.21, seed: int = 42) -> 'HllSet':
        """
        Create an HllSet from Redis hash data.

        Args:
            redis_data: The dictionary returned by redis.hgetall(redis_key)
            P: The precision for the HllSet
            tau: Initial coverage metric
            rho: Initial exclusion metric
            seed: Random seed

        Returns:
            An HllSet object
        """
        if not redis_data:
            raise ValueError("Redis data is empty or invalid")

        # Initialize a new HllSet
        hll = cls(P, tau, rho, seed)

        # Add elements from Redis data to the HllSet
        for key, value in redis_data.items():
            hll.add(key.decode() if isinstance(key, bytes) else key)
            hll.add(value.decode() if isinstance(value, bytes) else value)

        return hll

    @classmethod
    def from_julia(cls, julia_hll, P: int = 10, tau: float = 0.7, 
                   rho: float = 0.21, seed: int = 42) -> 'HllSet':
        """
        Create a Python HllSet from a Julia HllSet.
        
        Args:
            julia_hll: Julia HllSet object
            P: Precision parameter
            tau: Initial coverage metric
            rho: Initial exclusion metric
            seed: Random seed
            
        Returns:
            HllSet wrapper around the Julia object
        """
        hll = cls(P, tau, rho, seed)
        hll.hll = julia_hll
        return hll

    def __repr__(self) -> str:
        return (f"HllSet(P={self.P}, count={self.count():.0f}, "
                f"tau={self.tau:.3f}, rho={self.rho:.3f})")