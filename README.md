<div align="center">

# quantization-runtime

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Latest-FFD21E.svg)](https://huggingface.co/docs/transformers/)

**GPTQ and AWQ Post-Training Quantization (PTQ) implemented from scratch.**

4-bit Weight Compression | Hessian-based Optimization | Activation-Aware Scaling

</div>

---

## What It Does

This project implements the mathematical core of two industry-leading quantization algorithms. It transforms 16-bit floating-point weights (FP16) into 4-bit integers (INT4) while minimizing the loss of model "intelligence."

- **AWQ:** Protects critical weights connection to high-magnitude activations.
- **GPTQ:** Uses second-order information (Hessian matrix) to compensate for quantization errors column-by-column.

---

## Features

- **Hessian-based Calibration:** Accurately models weight importance using `X^T X` statistics.
- **Activation Tracking:** Forward hooks to capture real-world data distribution.
- **Group-wise Quantization:** High-precision scaling (Group Size: 128).
- **Correctness Verification:** Generates coherent text after 4-bit compression.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run AWQ Sanity Check
export PYTHONPATH=.
python examples/sanity_check_awq.py

# 3. Run GPTQ Comparison
python examples/compare_quantization.py
```

---

## Performance Insights (Qwen2-0.5B)

| Method | Target Precision | Optimization | Quality (Greedy) |
| :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 16-bit | None | Reference |
| **AWQ** | 4-bit | Activation-Aware | Coherent / Fast |
| **GPTQ** | 4-bit | Hessian-Inversion | High Accuracy |

---

## Project Structure

- `src/quantization/awq.py`: Activation-aware scaling logic.
- `src/quantization/gptq.py`: Column-by-column Hessian error propagation.
- `src/calibration/collector.py`: Hooks for data-driven stats collection.

---

## Limitations

- **Fake Quantization:** Currently simulates INT4 precision within FP16 containers to validate algorithmic logic. Real VRAM reduction requires custom INT4 matmul kernels.

---

## Related Projects

- [rag-inference-stack](https://github.com/JohnScheuer/rag-inference-stack): High-performance RAG serving.
- [speculative-decoding-runtime](https://github.com/JohnScheuer/speculative-decoding-runtime): Acceleration via draft models.
- [lora-inference-runtime](https://github.com/JohnScheuer/lora-inference-runtime): Multi-adapter LoRA serving.

---

## License

[MIT](LICENSE) - Joao Felipe De Souza, 2026
