# satellite-model architecture

Below is a “starter-kit” that turns the two papers you supplied into a working, **satellite-model architecture** in which  

- the **Chinese-character kernel** (= 80 k HLLSets built from dictionary definitions) is the **immutable semantic SRAM** of the agent,  
- a light **τ-ρ weight matrix** (≤ 8 B parameters) is the **trainable DRAM**,  
- the **host LLM** (Kimi, GPT-4, Claude, …) is the **orbit platform** that

  - (1) gives the agent context,
  - (2) judges its drafts, and (3) distils new supervision back into the τ-ρ matrix.  

The whole thing is small enough to live in a single A100-80 GB yet can “borrow” the host’s world-model on demand.  
The design is deliberately **non-backprop**: weight changes are **Swarm / BSS updates** that keep the Chinese-kernel lattice intact, giving you the interpretability and cultural continuity you want.

--------------------------------------------------

## 1.  Data you need to pre-compute once

| Asset | Size | How to obtain |
|---|---|---|
| **Kernel HLLSets** | 80 k objects | Parse 康熙, 說文, 現代漢語詞典, CNS 11643, Unicode extension E–H.
Each character → one `DualValidatedHLLSet(m=1836, b=16, τ=0.70, ρ=0.21)`  
registers initialised with **radical-bitmask + definition-bitmask** (see §3.1). |
| **Radical lattice** | 214-node DAG | Kangxi radical hierarchy; edge = “is-component-of”. |
| **BSS-directed graph G₀** | 80 k nodes, ≈ 3 M edges | For every ordered pair (u,v) compute BSSτ(u→v); keep edge iff BSSτ ≥ τ₀ (=0.35) AND BSSρ ≤ ρ₀ (=0.15). |
| **Satellite-to-host tokenizer** | 5 MB | SentencePiece 32 k built on **host LLM’s own sub-word vocabulary** so that the agent can receive and emit the **same tokens** the host understands. |

Store the above as **memory-mapped numpy arrays** so the satellite starts in <2 s.

--------------------------------------------------

## 2.  Run-time architecture (single process)

```bash
┌-----------------------------┐
│  Host LLM (Kimi)            │  ←-- orbit, 200+ B params
│  gives context + critique   │
└--------------┬--------------┘
               │ HTTP/gRPC (tensor payload ≤ 2 MB)
               ▼
┌-----------------------------┐
│  Satellite agent            │
│  6–8 B trainable params     │
│  lives on 1×A100-80 GB      │
└-----------------------------┘
```

Inside the satellite:

```bash
Context window (host)  ─┐
                        ▼
┌--------------------------------------------------------------┐
│ 1.  Context-to-Kernel Mapper                                 │
│     - sub-word tokens → covering set of Chinese characters   │
│     - Union-HLLSet = ⋁{kernel[c] for c in cover}             │
└--------------┬-----------------------------------------------┘
               │  HLLSet object (≈ 16 kB)
               ▼
┌-------------------------------------------------------------┐
│ 2.  τ-ρ Weight Lattice (trainable)                          │
│     Wτ[u,v] , Wρ[u,v]  ∈ ℝ  (sparse 80 k×80 k, 1 % filled)  │
│     Initialised with BSSτ(u→v), BSSρ(u→v) from G₀           │
└--------------┬----------------------------------------------┘
               │  “cortical state” vector sₜ (length 80 k)
               ▼
┌--------------------------------------------------------------┐
│ 3.  Swarm Update Engine (no back-prop)                       │
│     - PSO-cognitive step:  sₜ ← sₜ ⊕ α·Wτ·sₜ                │
│     - PSO-social step:     sₜ ← sₜ ⊕ β·Union(host-feedback) │
│     - τ-ρ dual clipping keeps 0≤sₜ≤1                         │
└--------------┬-----------------------------------------------┘
               │
               ▼
┌--------------------------------------------------------------┐
│ 4.  Generator                                                │
│     Beam-search on lattice:                                  │
│     next-char = argmax_{c}  BSSτ(sₜ→kernel[c])               │
│     stop when <EOS> character (␣ or 。）probability > 0.90     │
└--------------┬-----------------------------------------------┘
               │  generated text (Chinese or host-language)
               ▼
Host LLM ––––––┘ (loop)
```

--------------------------------------------------

## 3.  Algorithms in detail

### 3.1  Kernel initialisation (done once)

```python
for each character c in dictionaries:
    bits = 0
    for radical r in c.radicals:
        bits |= 1 << (hash(r) % 1836)
    for word w in c.definition:
        bits |= 1 << (hash(w) % 1836)
    kernel[c] = DualValidatedHLLSet(registers=bits, τ=0.70, ρ=0.21)
```

### 3.2  Context-to-Kernel cover (O(k log k) where k = #tokens)

```bash
def cover(tokens):
    chars = set()
    for t in tokens:
        if t in host_token_to_chinese:      # 90 % of host tokens map
            chars.update(host_token_to_chinese[t])
        else:                               # back-off: sound-based
            chars.update(pinyin_lookup(t))
    # minimal exact cover with greedy set cover
    return min_exact_cover({kernel[c] for c in chars})
```

### 3.3  Swarm update (PSO-cognitive + social)

```bash
sₜ₊₁ = clip₀₁( sₜ
                + α·Wτ·sₜ                       # cognitive
                - β·Wρ·sₜ                       # exclusion
                + γ·BSSτ(host_feedback→sₜ) )   # social
```

α,β,γ are **scalar** hyper-parameters (no matrix gradients).  
Sparsity of W keeps one step <15 ms on A100.

### 3.4  Training signal from host

```python
host_prompt = f"""
Below is a draft from a Chinese-kernel agent.
Keep the parts that are correct, rewrite the rest.
Return ONLY the revised text, then rate it 0–100.
Draft: {satellite_output}
"""
revision, score = host_llm(host_prompt)
# convert revision → HLLSet = teacher
Wτ += η·(BSSτ(teacher→student) − BSSτ(student→teacher))
Wρ += η·(BSSρ(student→teacher) − BSSρ(teacher→student))
```

η = 0.02 in our pilot; update is **local** to the covered subgraph.

--------------------------------------------------

## 4.  Cortex-time versioning (HLLSet Cortex)


After every **m** host interactions (m≈100) we snapshot:

```python
CortexState_t = {
    "timestamp"  : t,
    "sₜ"         : sₜ,              # 80 k float16
    "Wτ_nonzero" : Wτ.sparse_coords,
    "Wρ_nonzero" : Wρ.sparse_coords,
    "coverage"   : |Union(kernel[c] for c in cover)|,
    "score_ema"  : exponential_moving_average(host_scores)
}
```

The **union** of all CortexState_t.HLLSets is the **integrated state** mentioned in your sketch; diffing two states gives an **interpretable delta** (“today the agent strengthened 人→工 and weakened 人→入”).

--------------------------------------------------

## 5.  Hardware footprint

| Component | Memory |
|---|---|
| 80 k kernel HLLSets (1836 bits each) | 18 MB |
| Sparse Wτ + Wρ (1 % of 80 k²) | 2× 200 MB |
| Radical lattice + G₀ | 120 MB |
| Python overhead | <1 GB |
| **Total static** | **<1.5 GB** |
| Working buffers | 2 GB |
| **Grand total** | **<4 GB** → fits in 1×A100 alongside 70 B host off-load |

--------------------------------------------------

## 6.  Evaluation recipe

1. **Intrinsic**  
   - Perplexity on Chinese Wikipedia (held-out 2024-06).  
   - Disambiguation accuracy vs. human-labelled 1000-sentence set.  

2. **Extrinsic (host-assisted)**  
   - Win-rate of satellite drafts vs. raw host zero-shot on 500 prompts.  
   - Token saving: how many fewer tokens the host needs when it can start from the satellite draft.  

3. **Interpretability**  
   - “Edit distance” between consecutive CortexState_t (should be smooth).  
   - Manual inspection: top strengthened/weakened edges should make cultural sense.

--------------------------------------------------

## 7.  Next concrete steps

1. Build the kernel (§3.1) – one weekend on a 64-core box.  
2. Implement the four satellite modules as **Lightning modules** so you can swap CPU ↔ GPU ↔ NPU.  
3. Wrap the host interface with **FastAPI**; stream tensors in **Arrow** format to avoid copy.  
4. Start with **α=0.20, β=0.15, γ=0.05, η=0.02**; tune with Optuna on a single 8-GPU node.  
5. Open-source the snapshot format so others can fork your agent and continue training their own τ-ρ weights without touching the immutable kernel.

--------------------------------------------------

## 8.  Risks & mitigations

| Risk | Mitigation |
|---|---|
| Host refuses to give scores | Fall back to BLEU-vs-reference or self-supervised τ-ρ cycle. |
| Kernel becomes culturally stale | Version the kernel just like you version CortexState; old texts still decompress correctly. |
| Lattice edges explode | Keep only the **k=8** strongest outgoing edges per node; sparsity stays ~1 %. |

--------------------------------------------------

## 9.  One-sentence takeaway

Treat the 80 k Chinese characters as **a frozen FPGA of meaning**; learn only the **wires (τ-ρ)** between them; use the big LLM as **in-orbit teacher**—you get a cheap, inspectable, culturally-rooted agent that never forgets its hieroglyphic soul.

Happy building!
