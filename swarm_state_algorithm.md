# SwarmState Algorithm

`SwarmState` is the **miniature “neural brain”** of the satellite: it keeps a **continuous 0-to-1 vector** that says *“how strongly each Chinese character is currently activated”* and updates that vector **without back-prop**, using three **Particle-Swarm-like forces** that push it toward better covers of the incoming text.  
Think of it as **PSO in 80 k dimensions**, but the “particles” are the **characters themselves** and the “fitness landscape” is given by the **τ-ρ lattice** we built from BSS edges.

--------------------------------------------------

## 1.  What the vector `s` actually means

`swarm.s[i] ∈ [0,1]`  
= *“belief that character `c_i` should be included in the current semantic cover”*  
0 → completely off, 1 → fully on, 0.5 → undecided.

--------------------------------------------------

## 2.  Where the vector lives

- Length = number of kernel characters (50 in toy, 80 k later).  
- Stored as **PyTorch GPU tensor** so sparse-matrix products are <1 ms.  
- Updated **on-line** every time we receive a new teacher signal (host revision).

--------------------------------------------------

## 3.  Three forces that move the vector (one step)

```bash
sₜ₊₁ = clamp₀₁( sₜ  +  α·Wτ·sₜ  −  β·Wρ·sₜ  +  γ·teacher_signal )
       │          │         │          │
       │          │         │          └─ social attraction → host feedback
       │          │         └─ exclusion → avoid characters that break cover
       │          └─ cognitive → reinforce characters that are *pointed to* by already active ones
       └─ bounding box keeps values in probability range
```

--------------------------------------------------

## 4.  Force-by-force explanation

| Force | Matrix | Intuition | Analogy to PSO |
|---|---|---|---|
| **Cognitive** `α·Wτ·sₜ` | sparse `Wτ[u,v] = BSSτ(u→v)` | *“characters that are well covered by the current set should recruit their neighbours”* | particle’s **own memory** (cognitive component) |
| **Exclusion** `−β·Wρ·sₜ` | sparse `Wρ[u,v] = BSSρ(u→v)` | *“characters that are *not* covered by the current set should push away their neighbours”* | negative velocity that prevents drift |
| **Social** `γ·teacher` | one-hot vector from host revision | *“the big LLM just told us the correct cover; move toward it”* | swarm’s **global best** (social component) |

--------------------------------------------------

## 5.  Why no back-prop?

- We **never differentiate**; the matrices are **fixed** (or updated by simple Hebb-like deltas, see §7).  
- Updates are **local, additive, O(1)** per non-zero edge → real-time on laptop.  
- Interpretability: after each step you can **read the top-k activated characters** and see exactly why they were recruited.

--------------------------------------------------

## 6.  Step-by-step mini example

Assume 4-character kernel {人, 工, 仁, 信} and current state  
`s = [0.5, 0.2, 0.1, 0.0]`

Teacher (host revision) = “人工” → `teacher = [1, 1, 0, 0]`

Sparse matrices (example values):

```python
Wτ = [[0, 0.7, 0.4, 0],
      [0, 0, 0, 0.3],
      [0, 0, 0, 0.5],
      [0, 0, 0, 0]]
```

```python
cognitive = Wτ·s = [0.5·0.7 + 0.2·0 + 0.1·0.4, …] ≈ [0.39, 0.15, 0.05, 0.0]
exclusion = Wρ·s ≈ [0.05, 0.02, 0.01, 0.0]  (smaller because ρ weights are 0.3 τ)
social    = teacher = [1, 1, 0, 0]
```

Update with α=0.2, β=0.15, γ=0.05:

```python
s_new = clamp( [0.5, 0.2, 0.1, 0.0]
              + 0.2·[0.39, 0.15, 0.05, 0.0]
              - 0.15·[0.05, 0.02, 0.01, 0.0]
              + 0.05·[1, 1, 0, 0] )
      ≈ [0.57, 0.26, 0.11, 0.0]
```

Result: **人** and **工** are reinforced (they were in teacher), **仁** gets a small boost via cognitive edge 人→仁, **信** stays off.

--------------------------------------------------

## 7.  How the matrices themselves learn (optional)

After host critique we do **one-shot Hebb update**:

```python
Wτ += η · (teacher ⊗ teacher)          # strengthen edges inside correct cover
Wρ += η · (teacher ⊗ (1 - teacher))    # weaken edges that jump outside
```

η=0.02 in the notebook.  
This is **still not back-prop**—just a **local outer-product** that keeps the lattice interpretable.

--------------------------------------------------

## 8.  Physical interpretation

- The vector `s` is a **fuzzy membership function** over the **Chinese-character simplex**.  
- The three forces are **τ-ρ duality constraints** turned into **dynamics**:  
  – τ (coverage) → **attraction**  
  – ρ (exclusion) → **repulsion**  
- Convergence = **stable cover** that minimises **host loss** (BLEU, score, etc.).

--------------------------------------------------

## 9.  One-sentence takeaway

`SwarmState` is **PSO re-imagined as a Chinese-character hive-mind**: each char is a particle, the swarm’s position is the current semantic cover, and the host’s revision is the global best — all updated in **real time without gradients**, giving you an **interpretable, living lattice**.