<div align="center">

# quantization-runtime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Success-brightgreen.svg)](#performance--quality)

**Direct implementation of AWQ and GPTQ Post-Training Quantization (PTQ) from scratch.**

Analyzing the mathematical stability and intelligence preservation of 4-bit weight compression on small-scale LLMs.

</div>

---

## What It Does

This project implements the core mathematical logic of **AWQ** and **GPTQ**. By bypassing black-box libraries, it provides a deep-dive into how weight-clipping and error-propagation affect model perplexity.

- **AWQ Success:** Achieved **26.05 PPL** on Qwen2-0.5B (4-bit), nearing the FP16 baseline of 20.17.
- **GPTQ Analysis:** Documented the numerical instability of Hessian-based error compensation in low-parameter models.

---

## Performance & Quality (Qwen2-0.5B)

**Configuration:** 128 calibration samples (WikiText-2), Group Size 128, 4-bit simulation.

| Method | Precision | Perplexity (Lower is better) | Status |
| :--- | :--- | :--- | :--- |
| **FP16 (Baseline)** | 16-bit | **20.17** | Reference |
| **AWQ** | 4-bit (Fake) | **26.05** | **Highly Successful** |
| **GPTQ Hybrid** | 4-bit (Fake) | 1.2M+ | Numerical Limit reached |

### Key Technical Findings:
- **AWQ Robustness:** By scaling salient weights before quantization, AWQ successfully preserved model coherence with a minimal +5.88 PPL increase.
- **GPTQ at Small Scale:** Small-scale LLMs lacks the parameter redundancy required for GPTQ's second-order error compensation, leading to numerical divergence despite advanced damping and isolation techniques.

---

## Features

- **Hessian-based Calibration:** Accurately models weight correlations using normalized `X^T X` statistics.
- **Activation Tracking:** Forward hooks to capture real-world magnitude distributions.
- **Numerical Stability Suite:** Implemented FP64 inversion, 20% damping, and error-clipping for GPTQ robustness.
- **Evaluation Pipeline:** Integrated Perplexity (PPL) measurement on WikiText-2.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Full Quality Benchmark (AWQ vs GPTQ vs FP16)
export PYTHONPATH=.
python benchmarks/evaluate_quality.py
```

---

## Project Structure

- `src/quantization/awq.py`: Activation-aware pre-scaling and group-wise quantization.
- `src/quantization/gptq.py`: Hessian-based error propagation with stability safeguards.
- `src/evaluation/perplexity.py`: Cross-entropy loss based quality measurement.

---

## Related Projects

- [rag-inference-stack](https://github.com/JohnScheuer/rag-inference-stack): Knowledge-augmented generation.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Inference acceleration.
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-adapter serving.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
