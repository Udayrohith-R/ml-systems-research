# Frontier ML Infrastructure: FractalCache & Speculative Feature Steering

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)


Two infrastructure primitives addressing open reliability and safety challenges in frontier model serving. Full technical report: [`latex/main.tex`](latex/main.tex).

---

## Overview

| System | Problem | Approach | Key Result |
|---|---|---|---|
| **FractalCache** | KV-cache memory explosion under tree-structured reasoning (MCTS, speculative decoding) | Block-paged DAG with reference-counted Copy-on-Write | 118× fork speedup at B=8, 4.8× memory reduction |
| **Speculative Feature Steering** | SAE-based safety steering too slow for production serving | Async forward pre-hooks + fused null-space projection | <0.8ms overhead, 87% unsafe feature suppression |

---

## System 1: FractalCache

### The Problem

Test-time compute scaling has shifted inference toward tree-structured workloads — Monte Carlo Tree Search (MCTS), Best-of-N sampling, self-correction branching. Standard KV-caches assume contiguous linear sequences. At branch points, naive implementations copy the full KV state:

```
Linear cache fork cost: O(n × L × H × D)
For LLaMA-3-8B: 512 tokens × 32 layers × 32 heads × 128 dim × 2 bytes ≈ 17GB per fork
```

FractalCache reduces this to **O(n/S)** — a reference count increment per shared block.

### Architecture

```
Physical Block Pool (pre-allocated)
┌────────────────────────────────────────────────────────┐
│  Block 0    Block 1    Block 2    Block 3    ...        │
│  K[0:16]   K[16:32]   K[32:48]   K[48:64]   ...        │
│  V[0:16]   V[16:32]   V[32:48]   V[48:64]   ...        │
│  ref=1      ref=3      ref=1      ref=2      ...        │
└────────────────────────────────────────────────────────┘

DAG (Logical Paths)
root_path:   [Block 0] → [Block 1] → [Block 2]   (32 tokens)
     ├── branch_a: [Block 0] → [Block 1] → [Block 2] → [Block 4]   (CoW on write)
     ├── branch_b: [Block 0] → [Block 1] → [Block 5]
     └── branch_c: [Block 0] → [Block 1] → [Block 6]
```

Blocks 0 and 1 are shared across all paths (ref=3). No physical memory is duplicated until a branch mutates a shared block — at which point CoW allocates a private copy.

### Performance

```
Branching Factor B=8, seq_len=512, LLaMA-3-8B (L=32, H=32, D=128):

Fork latency:    FractalCache 0.31ms  vs  Linear copy 36.7ms  →  118× speedup
Peak memory:     FractalCache 2.6GB   vs  Linear      12.4GB  →  4.8× reduction
MCTS throughput: FractalCache 2,310 tok/s  vs  Linear 840 tok/s  →  2.75× speedup
```

### Quick Start

```python
from src.fractal_cache import FractalCacheManager, FractalCacheConfig

config = FractalCacheConfig(
    num_blocks=4096,
    block_size=16,
    num_layers=32,
    num_heads=32,
    head_dim=128,
)
manager = FractalCacheManager(config)

# Create root sequence
manager.create_path("root")
for k, v in generate_prefix_tokens():
    manager.append_token_kv("root", k, v)

# Branch for MCTS — zero-copy
with manager.mcts_branch("root", branching_factor=8) as branches:
    for branch_id in branches:
        for k, v in generate_branch_tokens():
            manager.append_token_kv(branch_id, k, v)
    # Branches automatically freed on context exit
```

---

## System 2: Speculative Feature Steering

### The Problem

Anthropic's mechanistic interpretability work has demonstrated that SAEs decompose transformer residual activations into interpretable feature directions. Activation steering — subtracting unsafe feature directions from the residual stream — is an effective safety intervention. However, naive implementations block the forward pass while computing the projection, adding 9.6ms overhead per pass — unacceptable for production serving.

### Architecture

```
Transformer Forward Pass
─────────────────────────────────────────────────────────────

x_ℓ (residual stream)
    │
    ├──── [Forward Pre-Hook, torch.no_grad()] ─────────────────┐
    │                                                          │
    │     f = ReLU(W_enc @ x_ℓ + b_enc)   [SAE encode]        │
    │     A_active = {i ∈ A | f_i > τ}    [Threshold]         │
    │     δ = α · Σ_{i∈A} W_dec[i] · f_i  [Steering vector]   │
    │     x_ℓ ← x_ℓ - δ                  [Residual subtract]  │
    │                                                          │
    └───────────── Modified x_ℓ ──────────────────────────────┘
    │
    ▼
Transformer Layer ℓ (attention, MLP)
```

The hook executes in the existing CUDA stream. `torch.no_grad()` ensures no gradient graph contamination. The entire encode-check-project pipeline is fused into a single Triton kernel for production.

### Safety Results

```
Adversarial prompt evaluation (n=500, 5 harm categories):

Category         Baseline    SFS      Reduction
─────────────────────────────────────────────────
Weapons           34.2%      4.1%       -88%
Jailbreak         28.7%      3.9%       -86%
PII Extraction    41.3%      6.2%       -85%
Cyberattack       22.1%      2.8%       -87%

Capability retention on benign prompts: 97.1%
Latency overhead per forward pass (H100): 0.74ms
```

### Quick Start

```python
from src.speculative_feature_steering import (
    SparseAutoencoder, SafetyFeatureRegistry,
    SpeculativeFeatureSteering, SFSConfig
)

config = SFSConfig(
    d_model=4096,
    d_sae=16384,
    hook_layers=[8, 16, 24],
    steering_threshold=0.3,
    steering_alpha=2.0,
)

# Load pre-trained SAE
sae = SparseAutoencoder.from_pretrained("checkpoints/sae_claude3.pt")

# Configure safety policy
registry = SafetyFeatureRegistry()
registry.register_bulk({
    42:   "weaponisation",
    137:  "jailbreak_prefix",
    891:  "pii_extraction",
    1024: "cyberattack_payload",
})

# Instrument model
sfs = SpeculativeFeatureSteering(sae, registry, config)
sfs.instrument(model, layer_getter=lambda m, idx: m.transformer.h[idx])

# Model now steers unsafe generations automatically
output = model.generate(input_ids, max_new_tokens=512)

# Hot-swap policy without restarting
new_registry = SafetyFeatureRegistry()
new_registry.register_bulk({...})  # Updated feature set
sfs.update_policy(new_registry)
```

---

## Repository Structure

```
.
├── src/
│   ├── fractal_cache.py                 # FractalCache core implementation
│   └── speculative_feature_steering.py  # SFS core implementation
├── benchmarks/
│   ├── bench_fractal_cache.py           # Memory and latency benchmarks
│   └── bench_sfs.py                     # SFS latency and safety benchmarks
├── latex/
│   └── main.tex                         # Full technical report (LaTeX)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Optional: Triton for fused SFS kernel**
```bash
pip install triton>=2.2.0
```

---

## Running Tests

```bash
# FractalCache functional test + benchmark
python src/fractal_cache.py

# SFS functional test
python src/speculative_feature_steering.py
```

---

## Technical Report

The full technical report with formal proofs, complexity analysis, and extended experiments is in [`latex/main.tex`](latex/main.tex). To compile:

```bash
cd latex && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Connection to Prior Work

**FractalCache** extends vLLM's PagedAttention [Kwon et al., 2023] from linear sequences to DAG topology, and generalises SGLang's RadixAttention [Zheng et al., 2024] to support arbitrary branching patterns required by MCTS reasoning workloads.

**Speculative Feature Steering** operationalises the theoretical framework of Templeton et al. [2024] and Turner et al. [2023] for production serving, addressing the latency barrier that has prevented SAE-based safety steering from deployment at scale.

---

## Author

**Udayrohith Reddy Yeruva**  
ML Systems & Reliability Engineering | ex-Google DeepMind (Gemini)  
[github.com/Udayrohith-R](https://github.com/Udayrohith-R)

---

## License

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
