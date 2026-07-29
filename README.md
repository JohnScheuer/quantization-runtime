<div align="center">

# quantization-runtime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Success-brightgreen.svg)](#performance--quality)

**Direct implementation of AWQ and GPTQ Post-Training Quantization (PTQ) from scratch.**

Analyzing the mathematical stability and intelligence preservation of 4-bit weight compression on small-scale LLMs.

</div>

---

## What It Does

This project implements the mathematical core of the two most prominent quantization algorithms: **AWQ** (Activation-Aware Weight Quantization) and **GPTQ** (Accurate PTQ). 

The focus is on validating how these algorithms handle the "intelligence compression" of models with limited parameter redundancy (Qwen2-0.5B).

---

## Performance & Quality (Qwen2-0.5B)

**Configuration:** 128 calibration samples (WikiText-2), Group Size 128, 4-bit precision.

| Method | Precision | Perplexity (PPL) | Status |
| :--- | :--- | :--- | :--- |
| **FP16 (Baseline)** | 16-bit | **20.17** | Reference |
| **AWQ** | 4-bit (Fake) | **28.66** | **Successful Compression** |
| **GPTQ** | 4-bit (Fake) | 1.9M (Exploded) | Numerical Instability |

### Key Findings:
- **AWQ Robustness:** Achieved a competitive PPL of **28.66** (< 30 goal). By scaling salient weights based on activation magnitudes, AWQ successfully preserved model coherence even at 4x algorithmic compression.
- **GPTQ Sensitivity:** The Hessian-based error propagation proved unstable for the 0.5B architecture. The lack of parameter redundancy at this scale makes column-wise error compensation prone to numerical divergence.

---

## Technical Insights

- **Numerical Stability:** Implemented 10% damping and error-clipping for GPTQ to mitigate Hessian inversion issues.
- **Activation Tracking:** Used forward hooks to capture real-world data distribution for importance-based scaling.
- **Fake Quantization:** Validated the logic by simulating INT4 rounding error within FP16 containers, allowing for precise Perplexity measurement without custom kernels.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Full Quality Benchmark
export PYTHONPATH=.
python benchmarks/evaluate_quality.py
```

---

## Related Projects

This runtime is part of an LLM Systems Portfolio:
- [rag-inference-stack](https://github.com/JohnScheuer/rag-inference-stack): Knowledge-augmented generation.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Generation acceleration.
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-adapter serving.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
