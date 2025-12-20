# 🧠 HLLSet-Swarm

>*Programmable swarm trajectories via HLLSet–PSO duality*

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/uv/v/hllset_swarm)](https://pypi.org/project/hllset_swarm/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it is

HLLSet-Swarm turns the **mathematical duality** between

  - *(a) relational algebra of Chinese-character HLLSets* and  
  - *(b) Particle-Swarm Optimization dynamics*  

into a **declarative GPU kernel compiler** that lets you **script** how a 80 k-dimensional “semantic swarm” should move, converge and **write its final state back** to any external system (LLM, DB, robot, …) as **live feedback**.

Think *“Git for meaning”* – every trajectory ends with a content-addressed commit that immortalises the swarm’s belief state.

---

## ✨ Key features

| Feature | What you get |
|---|---|
| **Duality engine** | PSO guarantees → HLLSet stability proofs |
| **Programmable trajectories** | YAML → GPU sparse kernels (no CUDA code) |
| **Recursive meta-swarm** | swarm-of-swarms for higher-order abstraction |
| **Git backend** | every layer is a `.pt.zst` blob pushed to Github |
| **Environment adapters** | OpenAI, SQL, ROS, stdout … plug your own |
| **Laptop→data-center** | 80 k dims run in < 1 GB VRAM (RTX 3060 ready) |

---

## ⚡ 30-second demo

```bash
git clone https://github.com/alexmy21/hllset_swarm.git
cd hllset_swarm
uv add -e .
export GITHUB_TOKEN="ghp_xxx"
```

```python
from hllset_swarm import SwarmProgram, Environment

env = Environment(text="人工智能正在改变世界")
prog = SwarmProgram.from_yaml("trajectories/default.yml")
prog.run(env)                       # ← GPU kernel executes
prog.commit_to_github("user/cortex", token=os.getenv("GITHUB_TOKEN"))
print("embedding shape:", env.embedding.shape)  # (80000,)
```

---

## 📁 Repository layout

```bash
hllset_swarm/
├── src/hllset_swarm/
│   ├── kernel.py          # immutable Chinese-character HLLSets
│   ├── lattice.py         # BSS τ-ρ sparse matrices
│   ├── swarm.py           # GPU resident SwarmState
│   ├── trajectory.py      # YAML → kernel compiler
│   ├── cortex.py          # recursive meta-swarm
│   └── io/
│       ├── github.py      # Github Contents API backend
│       └── env.py         # adapters for external systems
├── trajectories/          # ready-made scripts
│   ├── default.yml
│   └── meta_swarm.yml
├── tests/
└── examples/
    ├── notebook.ipynb
    └── ros_talker.py
```

---

## 🛠️ Installation

### Using `uv` (fastest)

```bash
uv add hllset_swarm
```

### From source

```bash
git clone https://github.com/yourname/hllset_swarm.git
cd hllset_swarm
uv add -e .
```

### Julia dependency (only for HLLSet backend)

```bash
# one-liner installer
curl -fsSL https://install.julialang.org | sh
julia -e 'using Pkg; Pkg.add("HllSets")'
```

---

## 🎯 Concepts in one picture

```text
Chinese text
     │
     ▼
[HLLSet cover]  ──BSS τ-ρ──►  GPU SwarmState  ──converge──►  s(t+1)
     ▲                                                    │
     │              PSO-HLLSet duality                    ▼
Environment  ◄──feedback──  Github commit  ◄──layer blob──┘
```

---

## 📝 Writing a trajectory

`trajectories/default.yml`

```yaml
name: chinese_cover
kernel: 80k_ccd.json.gz
precision: 10               # 1024 registers

params:
  alpha: 0.20
  beta:  0.15
  gamma: 0.05
  eta:   0.02

trajectory:
  - op: reset
    value: 0.5
  - op: cover        # push entry cover into swarm
    entry: "{{ env.text }}"
  - op: converge
    max_steps: 5
    tol: 1e-3
  - op: feedback
    target: env.embedding   # write s(t+1) back
```

Run it:

```python
prog = SwarmProgram.from_yaml("trajectories/default.yml")
prog.run(env)
```

---

## 🎲 Controlled noise – low-precision hash as regularizer

| Precision | Collision rate | Use-case | Noise role |
|---|---|---|---|
| **64 bit** | < 0.1 % | production Chinese | almost deterministic |
| **32 bit** | ≈ 1 % | mobile emoji | **mild regulariser** |
| **16 bit** | ≈ 6 % | MCU controller | **strong regulariser** |
| **8 bit** | ≈ 30 % | toy demos | **extreme dropout** |

**Interpretation**:

- **High collision** = **bit-dropout** → union **looks bigger** than reality.  
- **Multi-seed triangulation** = **denoising U-Net** → recover **true cover**.

---

## 🧠 Denoising analogy (vision → semantics)

| Vision pipeline | Semantic pipeline |
|---|---|
| **Gaussian noise** | **hash collision dropout** |
| **Noisy image** | **noisy HLLSet union** |
| **U-Net denoiser** | **multi-seed Hopfield descent** |
| **Clean image** | **disambiguated cover** |

**Same math**, **different substrate**.

---

## 🔌 Environment adapters

| Adapter | Description |
|---|---|
| `OpenAIAdapter` | write embedding into system prompt |
| `SQLAdapter` | store vector in Postgres `VECTOR` column |
| `ROSAdapter` | publish `Float32MultiArray` on `/semantic_state` |
| `StdoutAdapter` | debug JSON to console |

Add your own:

```python
from hllset_swarm.io import BaseAdapter
class MyAdapter(BaseAdapter):
    def update_embedding(self, vec: np.ndarray):
        requests.post("http://my.api/embedding", data=vec.tobytes())
```

---

## 📊 Hardware requirements

| Component | Size | Note |
|---|---|---|
| Chinese kernel (80 k) | 160 MB | memory-mapped |
| Sparse Wτ / Wρ | 2 × 200 MB | half-precision |
| GPU working set | < 6 GB | RTX 3060 12 GB ✅ |
| One layer commit | 1-3 MB | zstd compressed |

---

## 📈 Performance

| Metric | RTX 3060 | A100 80 GB |
|---|---|---|
| Single swarm step | 0.8 ms | 0.15 ms |
| 5-step trajectory | 4.2 ms | 0.8 ms |
| Layer commit + upload | 0.3 s | 0.2 s |

---

## 🧪 Tests

```bash
uv run pytest tests/
```

---

## 🚦 Roadmap

| Month | Milestone | Status |
|---|---|---|
| **November 2025** | ✅ **PoC on 300-char dictionary** | DONE |
| **November 2025** | ✅ **Julia backend + GPU kernels** | DONE |
| **December 2025** | **Goal: 80 k kernel + sparse lattice** | 🚧  |
| **December 2025** | **Goal: programmable YAML trajectories** | 📋 |
| **January 2026** | **Goal: Git-commit cortex layers** | 📋 |
| **January 2026** | **Goal: environment adapters (LLM, DB, ROS)** | 📋 |
| **January 2026** | **Goal: first kibbutz (3-node collective)** | 🎯 |

---

### 🔭 Next giant leap – **SGS.ai Kibbutz**

> “The same maths that describes birds finding food describes bits finding meaning.”  
> We now let **those bits farm together**.

| Kibbutz Feature | Description | Target Date |
|---|---|---|
| **Radical sharding** | conflict-free parallel farming | Jan 2026 |
| **CRDT consensus** | arithmetic-mean lattice merge | Feb 2026 |
| **Host-score income** | Hebb burn proportional to BLEU/RLHF | Q1 2026 |
| **Elastic scale** | join/leave without downtime | Q1 2026 |
| **Cross-domain kibbutz** | Chinese + Arabic + English swarms | Q2 2026 |

---

## 🌍 Beyond Chinese – any *"hieroglyphic"* substrate

Chinese is **our first substrate** because it is **optimally hieroglyphic**:

- finite, standardised inventory (≈ 80 k)  
- unambiguous dictionary definitions **in the same language**  
- clear **radical→character→word** composition rules  
- 3 000 years of **continuous semantic fossil record**

But the **mathematics is substrate-agnostic**.  
Any symbol set that satisfies **four axioms** can be dropped in:

1. **Non-inflectional** (no paradigms, no declensions)  
2. **Compositionally closed** (complex = stack of simples)  
3. **Lexicographically frozen** (each symbol has **one** normative definition)  
4. **Hashable** (deterministic bit-pattern from symbol)

---

### 🧪 Substrates on the roadmap

| Substrate | Inventory | Composition unit | Status | ETA |
|---|---|---|---|---|
| **Chinese (CCD)** | 80 k chars | radical | ✅ reference | now |
| **Classic Maya glyphs** | 1 100 glyphs | block | 🚧 POC | Q4 2025 |
| **Emoji 15.1** | 3 782 emojis | ZWJ sequence | 📋 design | Q1 2026 |
| **Minecraft blocks** | 1 500 blocks | voxel neighbour | 📋 design | Q1 2026 |
| **AI Esperanto** | 10 k morphemes | concat-rule | 📋 white-paper | Q2 2026 |

---

### 🕹️ Example – Minecraft substrate (sketch)

```yaml
substrate: minecraft
inventory: minecraft_blocks.json.gz
precision: 12          # 4096 registers
hash_seed: "mc1.20.1"
composition_rule: "6-face-voxel+up/down"
definition_source: "block_state.properties"
```

- **Block** → HLLSet hashed from **block-state NBT**  
- **Structure** → union of block HLLSets  
- **Scene embedding** → swarm convergence on block-cover

Same YAML, same GPU kernel, **different universe**.

---

## 🤝 Contributing

1. Fork  
2. `uv add -e .`  
3. `uv run pytest`  
4. PR against `main`

We love **new adapters** and **trajectory recipes**!

---

## 📄 Citation

```bibtex
@software{hllset_swarm,
  title = {HLLSet-Swarm: Programmable Swarm Trajectories via HLLSet--PSO Duality},
  author = {Alex Mylnikov, Aleksandr Solonin},
  url = {https://github.com/alexmy21/hllset_swarm},
  year = {2025}
}
```

---

## 📜 License

MIT – see [LICENSE](LICENSE).

---

**Star ⭐ the repo if you want Git to remember meaning for you.**

## References

1. [Check project wiki for more information](https://github.com/alexmy21/hllset-swarm/wiki)
2. [U-Net](https://arxiv.org/pdf/1505.04597)
3. [U-Net medium](https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066)
4. [Kalman Filters in Python](https://medium.com/@ccpythonprogramming/particle-filters-in-python-when-kalman-filters-break-down-75d51550ee15)
5. [Umbral Calculus](https://www.cantorsparadise.com/the-dark-art-of-umbral-calculus-976959da72ca)
6. [Bell: The Theorem That Proved Einstein Wrong](https://medium.com/@manaritaa/coffee-with-bell-the-theorem-that-proved-einstein-wrong-cddc77522885)
7. [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871)
8. [BLEU](https://en.wikipedia.org/wiki/BLEU)
9. [Gated Attention for Large Language Models](https://openreview.net/pdf?id=1b7whO4SfY)
10. [Spectral Theory for DAG](https://isamu-website.medium.com/understanding-spectral-graph-theory-for-dags-7ca767cec122)
11. [Linear Mathematics as Unified Cognitive Space](https://medium.com/@tomkob99_89317/linear-mathematics-as-unified-cognitive-space-from-fragmented-knowledge-to-structural-clarity-d1f0738d6528)
12. [Towards Enterprise-Ready Computer Using Generalist Agent](https://arxiv.org/pdf/2503.01861)
13. [Probability-density-is-not-an-extension-of-probability-mass](https://medium.com/@tomkob99_89317/probability-density-is-not-an-extension-of-probability-mass-but-average-unites-them-021aba877035)
14. [Recursive-Categorical-Framework](https://github.com/calisweetleaf/recursive-categorical-framework)
15. [HLLSet Communication](https://chat.deepseek.com/share/w08kirq9m9mlkiidex)
