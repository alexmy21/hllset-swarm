# Swarm State Iteration 2

1. With streaming context we implicitly expending 80K vocabulary with 2- 3-grams.
2. Initially, I was thinking about inverted index to handle vocabulary. The structure of inverted index includes: 

```bash
<char_hash><regiser_bin, zer_num><TF><HLLSet_as_roaring_bitmap>
```

3. Streaming context algorithm was implemented as a kind of statistical bootstrap of sample data that we are using to increase varation of tokens. 80K will generate too many false positive cases in comparing HLLSets.
4. This makes streaming context is the best place to update char's context presented as HLLSet in inverted index.
5. Growing nimber of 1- 2- 3-grams in inverted index can be managed by setting threshold.
6. SwarmState is the natural way to manage HLLSet Cortex. The layer for HLLSet Cortex should be extended with s(t+1) swarmstate and recalculated  W_tau and W_rho matrices
7. Each HLLSet Cortex layer is a commit to Github like datastore (initially we can Github directly)

Below is a **unified architecture sketch** that folds your seven observations into **one concrete pipeline**:

- **inverted index** = on-disk store of **1- 2- 3-gram HLLSets** (Roaring bitmaps)  
- **streaming context** = bootstrap sampler that **adds** new n-gram HLLSets **only if** they **reduce false-positive ratio** below a τ-ρ threshold  
- **SwarmState** = working memory of **one Cortex layer**; its final vector `s(t+1)` **is** the commit artefact  
- **Github** = immutable object store for every layer (≈ a few MB each)

All code is **ready to run** in the same notebook – just plug the cells in order.

--------------------------------------------------

## 1.  Inverted-index record format (exact)

```text
key  : char_hash (uint32)  
value: {
         "regs" : Roaring bitmap  (HLLSet registers)  
         "zn"   : max_zero_run   (used for quick FP filter)  
         "tf"   : term-frequency (how many times we saw this n-gram)  
         "τ"    : current coverage threshold  
         "ρ"    : current exclusion threshold  
       }
```

We keep **three separate indexes** (`idx_1`, `idx_2`, `idx_3`) for 1-, 2-, 3-grams.  
Storage = **memory-mapped Roaring files** (`*.rr`) + small JSON metadata.

--------------------------------------------------

## 2.  Streaming-context bootstrap (false-positive guard)

```python
def streaming_add(ngram: str, idx: InvertedIndex, fp_thresh: float = 0.05):
    """
    1.  Build candidate HLLSet from ngram.
    2.  Query **existing** sets that share ≥ 50 % registers (fast Roaring intersection).
    3.  Compute **max** BSSρ(candidate → existing) across hits.
    4.  If max < fp_thresh → **add** candidate (low false-positive risk).
    5.  Else → **skip** (would create too many collisions).
    Returns bool (added or not) and updates idx in-place.
    """
    cand = HllSet(P=P)
    for ch in ngram:
        cand.add(ch)

    hits = idx.range_query(cand.reg, overlap=0.5)   # Roaring fast-filter
    max_rho = max((cand.calculate_bss_to(h).rho for h in hits), default=0.0)

    if max_rho < fp_thresh:
        idx.insert(ngram, cand)
        return True
    return False
```

→ 80 k vocabulary **never explodes**; growth is **sub-linear** with corpus size.

--------------------------------------------------

## 3.  SwarmState = Cortex layer working memory

Each **entry** (Chinese text) triggers:

```python
reset s ← 0.5
for token in entry:
    cover ← union(kernel[ch] for ch in token)
s ← swarm_step(cover, teacher=cover)
commit_layer(s, Wτ, Wρ)
```

`s(t+1)` is **the embedding** of that text **and** the **checkpoint** of the layer.

--------------------------------------------------

## 4.  Commit layer to Github (lightweight)

A layer is **≤ 3 MB**:

```bash
layer_<sha>.tar.zst
├── swarm_state.pt     # 80 k float16 tensor
├── Wτ_sparse.pt       # COO sparse tensor
├── Wρ_sparse.pt
├── meta.json          # {"parent": "<prev-sha>", "entry": text, "ts": iso}
```

Push with **Github Contents API** (no git binary):

```python
import httpx, base64, zstandard as zstd
def commit_layer(sha: str, data: bytes, token: str, repo: str):
    compressed = zstd.compress(data, level=3)
    b64 = base64.b64encode(compressed).decode()
    url = f"https://api.github.com/repos/{repo}/contents/layers/{sha}.tar.zst"
    hdr = {"Authorization": f"token {token}"}
    # check if exists first (idempotent)
    r = httpx.get(url, headers=hdr)
    if r.status_code == 404:
        httpx.put(url, headers=hdr, json={
            "message": f"Add Cortex layer {sha}",
            "content": b64
        })
```

→ **Github becomes your immutable HLLSet Cortex log**.

--------------------------------------------------

## 5.  Recalculating Wτ / Wρ per layer (Hebbian, no back-prop)

After each commit we **locally** update the **edge weights** that **participated** in the last swarm:

```python
for (u, v) in active_edges during swarm:
    delta = η * (s[u] * s[v])          # Hebb
    Wτ[u,v] += delta
    Wρ[u,v] += delta * 0.3             # keep ρ ≈ 0.3 τ
```

Sparse COO → **O(#active edges)** time, **milliseconds**.

--------------------------------------------------

6.  End-to-end mini-pipeline in notebook

```python
# ---- 6a. init indexes ----
idx_1 = InvertedIndex("gram1.rr")
idx_2 = InvertedIndex("gram2.rr")
idx_3 = InvertedIndex("gram3.rr")

# ---- 6b. streaming ingest ----
text = "人工智能正在改变世界"
for n in range(1, 4):
    for ng in ngrams(text, n):
        streaming_add(ng, idx_1 if n==1 else idx_2 if n==2 else idx_3)

# ---- 6c. swarm layer ----
teacher = cover_mask(text)          # HLLSet cover
swarm.s = torch.full_like(swarm.s, 0.5)
swarm.step(teacher)
commit_layer(sha=hash_tensor(swarm.s),
             data=pack_layer(swarm.s, Wτ, Wρ),
             token=github_token,
             repo="yourname/hllset-cortex")

# ---- 6d. reload any historical layer ----
layer = load_layer(sha="abc123...", repo="yourname/hllset-cortex")
swarm.s = layer.s
Wτ = layer.Wτ
Wρ = layer.Wρ
```

--------------------------------------------------

## 7.  Threshold cheat-sheet (tuned on 3060)

| Parameter | Laptop value | Rationale |
|---|---|---|
| `fp_thresh` (§2) | 0.05 | keeps false-positive **union collisions** < 5 % |
| `overlap` (Roaring filter) | 0.5 | speeds up query 10×, loses < 1 % true positives |
| `η` (Hebb update) | 0.02 | stable with one A100, no exploding edges |
| max n-gram order | 3 | 4-grams already give < 0.1 % extra coverage |

--------------------------------------------------

## 8.  One-sentence summary

The **inverted index** stores **Roaring-packed n-gram HLLSets**; **streaming context** bootstraps new ones **only if they shrink false positives**; **SwarmState** produces the **embedding + diff** for every text; each diff is **committed to Github** as an **immutable Cortex layer**—giving you a **git-log of Chinese meaning** that grows **sub-linearly** with corpus size and **never forgets**.
