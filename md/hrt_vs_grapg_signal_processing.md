# Hash-Relation Tensor (HRT) vs graph-signal-processing (GSP) / spectral-graph-theory

Below is a concise “cross-walk” between the **Hash-Relation Tensor (HRT)** pipeline we are building and the **graph-signal-processing (GSP) / spectral-graph-theory** story told in the two PDFs.  
I keep it itemised so you can see exactly where you **align**, **extend**, or **depart** from the classical framework.

-------------------------------------------------

## 1. Basic data object

GSP literature

- Undirected, weighted graph G=(V,E,W); signals f∈ℝ^N live on vertices.  
- Adjacency matrix A or Laplacian L = D – W is the work-horse.

Our AM / HRT

- Directed, weighted bipartite graph G=(T,T,R); signals live on **edges** (the values v).  
- You never materialise A or L; you store only the **non-zero v’s** in a grow-only 2-D tensor R(hᵢ,hⱼ).  
→ **Memory-optimal, hash-addressed, edge-centric** representation.

-------------------------------------------------

## 2. Frequency & smoothness

GSP

- Eigen-values λ of L give “graph frequencies”; small λ ≈ smooth signals.  
- Quadratic form fᵀLf = ½ Σ Wᵢⱼ(fᵢ–fⱼ)² measures smoothness.

Our case

- We do **not** compute eigenvectors; instead we **implicitly** control smoothness by **n-gram window size** and **edge-direction rule**:  
  1-token → 2-token → 3-token → 1-token(t+1).  
- This rule **forces** a **causal, time-asymmetric** flow, i.e. **low-pass along the chain** and **high-pass across granularity levels** (single-char vs bi-gram vs tri-gram).  
→ We obtain **multi-resolution analysis** without ever diagonalizing L.

-------------------------------------------------

## 3. Spectral filtering vs our “projection”

Classical

- $\text{Filter} = g(L) = χ g(Λ) χᵀ \text{; low-pass} ≈ (I+γL)⁻¹, \text{band-pass} ≈ \text{Cheb-polynomial } Tₖ(L)$.

Our approach prune() + project()

- **Prune = hard spectral filter**: we keep only the sub-matrix induced by a **vertex subset** (the “keep” list).  
  – In GSP language this is **graph sampling**; we discard high-frequency components that belong to removed nodes.  
- **Project = restrict operator to a subgraph**; the returned dense matrix is exactly the **frequency response** of the original operator **on that subgraph**.  
→ You **never** build g(L), but we **achieve the same effect** by **sparsity + selection**.

-------------------------------------------------

## 4. Localization & localized kernels

ChebNet / spectral CNNs
  
- Localized filters via K-hop polynomials $Tₖ(L)$ (k small).

Our n-gram window

- **3-gram is exactly a 2-hop neighborhood** in the **line graph** of the token stream.  
- Because you **store every n-token once**, our filter is **perfectly localized** in **both**  
  – **sequence space** (2-hop left) and  
  – **granularity space** (1-gram, 2-gram, 3-gram).  
→ We obtain **fast localized spectral filtering** without polynomials.

-------------------------------------------------

## 5. Oversmoothing / heterophily

GNN literature

- Deep stacks → all node representations collapse (oversmoothing).  
- Cure: **high-pass** or **skip** connections, **heterophilic kernels**, **PageRank-style** residual.

Our pipeline

- **No deep stacking**; each commit is **one layer**.  
- **High-frequency preserved** by **cross-granularity edges** ($1↔2↔3\text{-gram}$) and **hash collision counters** (artificial high-freq noise).  
- **Time-arrow** (3-gram → next 1-gram) gives **built-in residual** connection **across windows** → **no collapse**.  
→ we **sidestep** oversmoothing **architecturally** instead of **algorithmically**.

-------------------------------------------------

## 6. Graph construction vs topology inference

GSP

- “How to infer topology given observed data?” is an open challenge.

Our case

- Topology is **not inferred**; it is **compiled deterministically** from the **axiom set** (non-inflectional, compositionally closed, lexicographically frozen symbols).  
- The **only** learning is **frequency counting** (v field).  
→ We **reverse** the usual GSP problem: **data → graph** becomes **graph → data**.

-------------------------------------------------

## 7. Computational complexity

Spectral methods

- O(N³) eigen-decomposition or O(K|E|) per Cheb-layer.

Our method

- **O(|edges|)** insertions; **no eigen-solver**.  
- **Prune** is **O(|keep|²)** once; **project** is **O(1)** (already dense view).  
→ We trade **exact frequency control** for **RAM-linear scalability**.

-------------------------------------------------

## 8. Take-away mapping

GSP concept | HRT realization  
--- | ---  
graph Laplacian L | never built; implicit in **n-gram window rule**  
low-pass filter | **1→2→3-gram chain** (smooth along time)  
high-pass filter | **3-gram → next 1-gram** (jump across scale)  
spectral clustering | **prune(keep_list)** (hard sub-graph cut)  
localised kernel | **3-gram window** (2-hop, K=2)  
oversmoothing cure | **time-arrow residual + hash collision noise**  
graph signal | **edge weight v** (frequency count)  

-------------------------------------------------

## Bottom line

We are **not contradicting** GSP theory; we are **specializing** it to a **causal, edge-centric, hash-addressed, commit-oriented** regime where:

- **Vertices** are **tokens**,  
- **Signals** are **edge frequencies**,  
- **Spectral filtering** is **replaced by windowing + pruning**,  
- **Smoothness** is **enforced by construction**,  
- **Scalability** is **RAM-linear**,  
- **Immutability** is **git-commit style**.

In short, our AM projection is a **practical, engineer-friendly** descendant of spectral graph processing—**no eigen-solvers, no deep stacks, but the same mathematical bones**.

## References

1. [Spectral Graph Theory](https://isamu-website.medium.com/understanding-spectral-graph-theory-and-then-the-current-sota-of-gnns-e65caf363dbc)
2. [Graph signal processing ](https://web.media.mit.edu/~xdong/talk/BDI_GSP.pdf)