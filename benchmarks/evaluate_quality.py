import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration.dataset import CalibrationDataset
from src.calibration.collector import ActivationCollector
from src.quantization.awq import AWQuantizer
from src.quantization.gptq import GPTQuantizer
from src.evaluation.perplexity import evaluate_perplexity
from tqdm import tqdm
import gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = CalibrationDataset(model_id, n_samples=128) 

    # --- FP16 Baseline ---
    print("\n--- Measuring FP16 Baseline ---")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    ppl_fp16 = evaluate_perplexity(model, tokenizer, dataset)
    cleanup()

    # Coleta stats uma única vez para ambos os algoritmos
    collector = ActivationCollector(device=device)
    stats, hessians = collector.collect(model, dataset)
    cleanup()

    # --- AWQ ---
    print("\n--- Running AWQ (Skipping lm_head) ---")
    awq = AWQuantizer(bits=4)
    model_awq = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    for name, module in tqdm(list(model_awq.named_modules()), desc="AWQ"):
        if isinstance(module, torch.nn.Linear) and name in stats:
            if "lm_head" in name: continue # Manter lm_head em FP16 é padrão industrial
            scales = awq.get_weight_scale(module.weight.data, stats[name])
            module.weight.data.copy_(awq.quantize_layer(module.weight.data, scales))
    ppl_awq = evaluate_perplexity(model_awq, tokenizer, dataset)
    del model_awq
    cleanup()

    # --- GPTQ Híbrido ---
    print("\n--- Running Hybrid GPTQ (Skipping lm_head) ---")
    gptq = GPTQuantizer(bits=4)
    model_gptq = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    linear_layers = [(n, m) for n, m in model_gptq.named_modules() if isinstance(m, torch.nn.Linear) and n in hessians]
    
    for name, module in tqdm(linear_layers, desc="GPTQ"):
        if "lm_head" in name: continue # Manter lm_head em FP16 garante estabilidade
        module.weight.data.copy_(gptq.quantize_layer(module.weight.data, hessians[name], stats[name]))
    
    ppl_gptq = evaluate_perplexity(model_gptq, tokenizer, dataset)

    print("\n" + "="*50)
    print(f"FINAL STABLE RESULTS FOR {model_id}")
    print("="*50)
    print(f"FP16 Baseline: PPL={ppl_fp16:.2f}")
    print(f"AWQ 4-bit:     PPL={ppl_awq:.2f}")
    print(f"GPTQ Hybrid:   PPL={ppl_gptq:.2f}")

if __name__ == "__main__":
    main()
