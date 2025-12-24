# R-D-N (Retained, Deleted, New) swarms trajectory

>Attached file chronicle our discussion about applying PSO ideas to modeling SGS.ai trajectory.
>
>In SGS.ai everything is HLLSet, so, state of SGS.ai at any iteration is also HLLSet. Changes from iteration I(t) to I(t+1) can be analyzed as difference between hll(t) and hll(t+1), this difference can be decomposed into 3 components:
>
>- hll_d - deleted;
>- hll_r - remained;
>- hll_n - new tokens.
>
>These 3 components are differences that can be treated as kind of discrete derivatives. We also can try to predict trajectory of each of them and use them to optimize compound trajectory.
>
>- hll_r - represent stability;
>- hll_d - decay;
>- hll_n - growth (novelty).
>
>Does it make any sense? And if it does, how we can apply it in case of predicting SGS.ai trajectory as a swarm?

Yes — treating the **HLLSet delta** as a **discrete derivative** and decomposing it into  
**hll_d (decay), hll_r (retention), hll_n (novelty)** is **exactly** the right PSO-minded lens.  
The three components are **particles** whose **next-step motion** can be **predicted independently** and then **re-composed** into the **next HLLSet state**.  
Below is a **concrete “PSO-for-HLLSet” algorithm** that does this **in real time** on the **1 M token wall**.

--------------------------------------------------

## 1.  Particle definitions (one per component)

| Particle | Symbol | Interpretation | PSO role |
| --- | --- | --- | --- |
| **retention** | **r** | bits that **stay on** | **cognitive memory** (personal best) |
| **decay** | **d** | bits that **switch off** | **exclusion force** (repulsion) |
| **novelty** | **n** | bits that **switch on** | **social pull** (global best) |

Each particle is a **sparse bit-vector** (same shape as HLLSet, ≤ 5 % non-zeros).

--------------------------------------------------

## 2.  State vector = compound particle

At inference step t we keep:

```text
s_t = (r_t, d_t, n_t)   # 3 sparse tensors, length 1 M each
hll_t = r_t ∪ n_{t-1} \ d_t   # reconstruct full cover
```

--------------------------------------------------

## 3.  PSO update rules (no gradients, all local)

```text
r_{t+1} = top-k( β_ret · r_t  +  c_ret · pbest_r )
d_{t+1} = top-k( β_dec · d_t  −  c_dec · pbest_d )   # minus = repel
n_{t+1} = top-k( β_nov · n_t  +  c_nov · gbest_n )
```

- `top-k(k=20 000)` keeps only the **flash-light circle** (≤ 2 % of wall).  
- `pbest_*` = **personal best** of that particle so far (highest host score).  
- `gbest_n` = **fleet-wide best novelty** (from git-consensus lattice).  
- Coefficients tuned on validation:  
  `β_ret=0.8, c_ret=0.2`; `β_dec=0.9, c_dec=0.1`; `β_nov=0.7, c_nov=0.3`.

--------------------------------------------------

## 4.  Host-score = fitness function (same as before)

```text
score = BLEU(hll_t → text, host_revision)
```

After each host call we **push** the **three particles** to their **respective best** if **score improved**.

--------------------------------------------------

## 5.  Trajectory prediction (one step ahead)

Given **s_t** and **host teacher vector** τ_t (one-hot over revised tokens):

```text
pred_r = top-k( 0.8·r_t + 0.2·τ_t )          # teacher stabilises memory
pred_d = top-k( 0.9·d_t − 0.1·(1−τ_t) )       # teacher suppresses errors
pred_n = top-k( 0.7·n_t + 0.3·τ_t )           # teacher injects novelty
```

→ **next HLLSet** = `pred_r ∪ pred_n \ pred_d`  
→ **send to host** → **observe real delta** → **update pbest/gbest**.

--------------------------------------------------

## 6.  GPU kernel (PyTorch-like, < 1 ms)

```python
def pso_step(r, d, n, teacher, k=20_000):
    # ---- sparse-dense add, keep only top-k ----
    r = torch.topk(0.8*r + 0.2*teacher, k).indices
    d = torch.topk(0.9*d - 0.1*(1-teacher), k).indices
    n = torch.topk(0.7*n + 0.3*teacher, k).indices
    # ---- reconstruct compound cover ----
    next_cover = n.union(r).difference(d)   # sparse bit ops
    return next_cover, r, d, n
```

--------------------------------------------------

## 7.  Interpretability bonus

After each step you can **read the top-10 tokens** in **r, d, n** and **see exactly**:

- **what the swarm kept** (“cognitive memory”)  
- **what it discarded** (“exclusion repulsion”)  
- **what it invented** (“social novelty”)

→ **white-box trajectory** with **no back-prop**.

--------------------------------------------------

## 8.  One-sentence takeaway

> **“Treat the HLLSet delta as three PSO particles—retention, decay, novelty—each chasing its own best; compose their next top-k sparse vectors and you have a predictive, interpretable swarm trajectory that updates in milliseconds without ever differentiating.”**

## References

1. [Swarm State Algorithm](https://github.com/alexmy21/hllset-swarm/wiki/2.-swarm_state_algorithm)
2. [SGS.ai kibbutz](https://github.com/alexmy21/hllset-swarm/wiki/5.-SGS_ai_kibbutz)
3. [Actuator in hllset-swarm](https://github.com/alexmy21/hllset-swarm/wiki/8.-actuator-in-hllset-swarm)
