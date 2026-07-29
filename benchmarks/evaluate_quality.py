import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration.dataset import CalibrationDataset
from src.calibration.collector import ActivationCollector
from src.quantization.awq import AWQuantizer
from src.quantization.gptq import GPTQuantizer
from src.evaluation.perplexity import evaluate_perplexity

def get_model_size_mb(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2

def main():
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Aumentamos para 128 amostras para estabilizar a Hessiana
    dataset = CalibrationDataset(model_id, n_samples=128) 

    print("\n--- FP16 Baseline ---")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    ppl_fp16 = evaluate_perplexity(model, tokenizer, dataset)

    # 1. AWQ
    print("\n--- AWQ 4-bit (Fake) ---")
    collector = ActivationCollector(device=device)
    stats, _ = collector.collect(model, dataset)
    awq = AWQuantizer(bits=4)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in stats:
            scales = awq.get_weight_scale(module.weight.data, stats[name])
            module.weight.data.copy_(awq.quantize_layer(module.weight.data, scales))
    ppl_awq = evaluate_perplexity(model, tokenizer, dataset)

    # 2. GPTQ (Recarregado)
    print("\n--- GPTQ 4-bit (Fake) ---")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    _, hessians = collector.collect(model, dataset)
    gptq = GPTQuantizer(bits=4, damp_percent=0.1) # Damping mais alto
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in hessians:
            module.weight.data.copy_(gptq.quantize_layer(module.weight.data, hessians[name]))
    ppl_gptq = evaluate_perplexity(model, tokenizer, dataset)

    print("\n" + "="*50)
    print(f"IMPROVED RESULTS FOR {model_id}")
    print("="*50)
    print(f"FP16: PPL={ppl_fp16:.2f}")
    print(f"AWQ:  PPL={ppl_awq:.2f} (Goal: < 30)")
    print(f"GPTQ: PPL={ppl_gptq:.2f} (Goal: < 30)")

if __name__ == "__main__":
    main()
