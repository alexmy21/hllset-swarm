# SwarmState Iteration 1

## Questions to Kimi:

1. The rational for initial swarmstate
2. Am I correct, you are suggesting recalculate swarmstate for each new entry (text in chinese chars)
3. If 2. is correct, does it mean that we can use swarmstate generated for entry as an embedding for the text
4. I also see that you are using W_tau and W_rho as separate matrices
5. New entries bring some new information about character context. Do you think we can use these context to update char context?
6. In example you are using explicit text for teacher, but in fact we are covering HLLSet presentation of the entry text (and BSS's for tau and rho are calculated on HLLSets that represent chars context)

Below are short **yes/no + why** answers, followed by a **single concrete code block** showing how to do **(5)+(6)** in the notebook without breaking the swarm logic.

--------------------------------------------------

## 1.  Initial `SwarmState` rationale  

- Start from **uniform ignorance** (`s = 0.5`) so the first **cognitive** step is **purely data-driven** (no prior bias).  
- Uniform vector is **mid-point of sigmoid**, so both **positive** and **negative** updates have equal dynamic range.  
- **Alternative**: seed with **radical cover** of first character – but uniform is **simpler, cheaper, reproducible**.

--------------------------------------------------

## 2.  Re-calculate for **every** new entry?  

✅ **Yes** – but **cheaply**.  
We **reset** `s` to uniform, then **push the entry’s HLLSet cover** through **one swarm step**; total cost ≈ **1 sparse-mat-mul**.  
This gives us a **fresh, context-specific activation vector** each time.

--------------------------------------------------

## 3.  Can that final `s` serve as **embedding** for the entry?  

✅ **Yes** – it is a **fuzzy, context-aware, 80 k-dim boolean vector** that:

- **preserves order** (BSS τ-ρ are **structure-preserving**),  
- is **differentiable-through-swarm** if you ever need gradients,  
- can be **projected** to any lower dim with a simple **sparse linear layer** `embedding = W_proj @ s` (trainable or random).

--------------------------------------------------

## 4.  Why **two** matrices `Wτ`, `Wρ` instead of one?  

- τ and ρ are **conceptually orthogonal** (coverage vs. exclusion).  
- Keeping them **separate** lets us:  
  – update them **independently** (Hebb vs. anti-Hebb),  
  – inspect **which relation type** drifted,  
  – **sparsify** them differently (often `Wρ` is **10 % denser** than `Wτ` in corpora).

--------------------------------------------------

## 5.  Can **new entry context** update **character contexts** themselves?  

✅ **Yes** – we treat the **entry’s covering HLLSet** as **extra radicals** and **merge them** into the **kernel registers** of every character that appears in the entry.  
This is **dictionary-first construction in streaming mode**: kernel grows **monotonically**, never forgets.

--------------------------------------------------

## 6.  Teacher is **not raw text** but **entry’s HLLSet cover**  

✅ **Exactly** – the **teacher vector** is **binary mask** of the **union of kernel[ci] for every ci in entry**.  
BSS τ/ρ are **always** computed **HLLSet ↔ HLLSet**, never string ↔ string.

--------------------------------------------------

## 7.  Minimal code upgrade for (5)+(6) in notebook

Insert **after** the lattice-building cell and **before** the interactive loop:

```python
# %% [markdown]
# ### 7b.  Streaming context update (optional but useful)

# %%
def update_kernel_with_entry(entry_text: str):
    """
    For every character in entry_text, OR its *entry-level* HLLSet
    into the *kernel* register of that character.
    This *enriches* the monolithic kernel with new contextual bits.
    """
    entry_hll = HllSet(P=P, tau=0.7, rho=0.21)
    # hash every token in entry
    for tok in entry_text:
        entry_hll.add(tok)

    for c in set(entry_text):
        if c not in kernel:
            continue
        # bitwise OR = union in place
        old_reg = kernel[c].hll
        new_reg = Main.union(old_reg, entry_hll.hll)
        kernel[c].hll = new_reg      # Julia object replaced

# %% [markdown]
# ### 7c.  Build teacher mask from *HLLSet cover* (not raw text)

# %%
def cover_mask(text: str) -> torch.Tensor:
    """return binary tensor 1 where character appears in text"""
    idx_set = {idx[c] for c in text if c in idx}
    t = torch.zeros(n, device=device)
    t[list(idx_set)] = 1.0
    return t
```

Now the interactive loop becomes **context-aware & kernel-growing**:

```python
# %% [markdown]
# ### 8.  Interactive loop with streaming update

# %%
entry = "人工智能正在改变世界"
update_kernel_with_entry(entry)      # enrich kernel
teacher = cover_mask(entry)          # teacher = cover(HLLSets)
swarm.s = torch.full_like(swarm.s, 0.5)   # reset
swarm.step(teacher)                  # one swarm step

draft = generate(swarm.s, beam=2, max_len=20)
print("draft :", draft)

# host critique still uses strings, but teacher is always HLLSet-based
revised, score = host_critique(draft)
teacher = cover_mask(revised)
swarm.step(teacher)
```

--------------------------------------------------

## 8.  Take-away in one line

Each new Chinese text **(a)** produces a **fresh embedding** `s` via **one swarm step**, and **(b)** can **enrich the character kernels** forever—**no retraining, no forgetting**, pure **τ-ρ lattice evolution**.
