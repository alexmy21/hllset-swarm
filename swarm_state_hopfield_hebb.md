# **Hopfield-style Hebbian updates** vs. **classical back-prop** in the **HLLSet-Swarm** setting

>(and **why we chose the non-backprop route** for `Wτ , Wρ`)

--------------------------------------------------

## 1.  What we actually do (in one formula)

Every consensus epoch (≈ 100 host calls) we perform:

```bash
Wτ ← Wτ + η ⋅ outer(s, s)                   (1)
Wρ ← Wρ + η ⋅ 0.3 ⋅ outer(s, s)             (2)
```

`s` is the **final swarm vector** (`0 ≤ sᵢ ≤ 1`).  
Equations (1–2) are **exactly** the **Hebbian rule** of a **modern continuous Hopfield network**:

```text
Jᵢⱼ = Jᵢⱼ + η ⋅ σᵢ ⋅ σⱼ
```

with **identity activation** σᵢ = sᵢ.  
So **Wτ and Wρ are two Hopfield energy matrices** that store **co-activation patterns** of Chinese characters.

--------------------------------------------------

## 2.  Why this is **sufficient** for our objective

| Objective | Hopfield Hebb | Back-prop |
|---|---|---|
| **Store** “which characters co-occur in good covers” | ✅ outer product | ✅ gradient |
| **Forget** old patterns gracefully | ✅ exponential decay (η small) | ✅ optimiser |
| **Interpret** edge value | ✅ `Wτ[u,v]` = empirical co-activation | ❌ million-parameter black box |
| **Update** in **milliseconds** on laptop | ✅ one sparse outer product | ❌ full backward pass |
| **Preserve** **τ-ρ duality** by construction | ✅ separate matrices | ❌ needs custom loss |
| **Merge** **CRDT-style** (arithmetic mean) | ✅ commutative | ❌ non-commutative |

→ **Back-prop is over-engineered** for a **memory** task; **Hopfield is built for it**.

--------------------------------------------------

## 3.  Energy landscape view (intuition)

Define **Hopfield energy**:

```text
E(s) = − ½ sᵀ Wτ s + ½ sᵀ Wρ s
     = − ½ Σ_{u,v} s_u (Wτ_{uv} − Wρ_{uv}) s_v
```

- **Minimising E** → **maximise coverage** (Wτ) **while** **minimising exclusion** (Wρ).  
- **SwarmState step** is **one gradient descent step on E** with **learning rate = α**.

→ **Swarm trajectory** = **rolling descent in a Hopfield energy landscape** whose **attractors** are **good Chinese covers**.

--------------------------------------------------

## 4.  Capacity & false positives (numbers)

- **Sparse Hebb** capacity ≈ **0.14 ⋅ nnz edges** (Hopfield 1984).  
  Our lattice: **1 % of 80 k² ≈ 64 M edges** → **9 M patterns** before **catastrophic overlap**.  
  Even at **1 k patterns/day** we need **25 years** to hit the wall.

- **False-positive check** (BSS ρ < 0.05) is **done at insert time** → **never stored** in the first place.

→ **no need for back-prop regularisation**; **sparse Hopfield + FP gate** is enough.

--------------------------------------------------

## 5.  When we **will** use back-prop (future work)

1. **Meta-swarm controller** (RL policy that **chooses** `α, β, γ, η` per layer) – **continuous action space** → policy gradient.  
2. **Adapter fine-tuning** (OpenAI prompt prefix that **consumes** `s(t+1)`) – **differentiable prompt tuning**.  
3. **Cross-modal fusion** (image + text → joint HLLSet) – **gradient needed for vision encoder**.

But **Wτ , Wρ themselves stay Hebbian** – they are **memory**, not **policy**.

--------------------------------------------------

## 6.  One-sentence takeaway

> **Wτ and Wρ are sparse, interpretable Hopfield memories**;  
> **swarm descent** minimises **Hopfield energy** whose attractors are **good Chinese covers**;  
> **no back-prop is needed** until we **learn how to learn** the **learning rates themselves**.

## References

1. <https://www.youtube.com/watch?v=M6wRIqug93Y>
2. <https://ml-jku.github.io/hopfield-layers/>
3. <https://en.wikipedia.org/wiki/Modern_Hopfield_network>
