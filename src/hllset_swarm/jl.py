"""
Minimal Julia bridge (assumes HllSets.jl is loaded).
"""
from julia import Main as _J

def HllSet(P: int):
    return _J.HllSet(P)

# re-export BSS method
_J.eval("bss_tau(h1, h2) = h1.calculate_bss_to(h2).tau")
_J.eval("bss_rho(h1, h2) = h1.calculate_bss_to(h2).rho")