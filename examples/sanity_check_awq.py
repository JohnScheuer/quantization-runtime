import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration.dataset import CalibrationDataset
from src.calibration.collector import ActivationCollector
from src.quantization.awq import AWQuantizer
import os

def main():
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model & Tokenizer
    print(f"Loading {model_id} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Collect Activations (Calibração)
    dataset = CalibrationDataset(model_id, n_samples=32, seq_length=512)
    collector = ActivationCollector(device=device)
    print("Starting calibration...")
    activation_stats = collector.collect(model, dataset)

    # 3. Quantize with AWQ
    quantizer = AWQuantizer(bits=4, group_size=128)
    print("\nApplying AWQ quantization layer by layer...")
    
    # Quantizamos apenas as camadas de atenção e MLP (nn.Linear)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in activation_stats:
                # Calcula escala AWQ baseada nas ativações coletadas
                scales = quantizer.get_weight_scale(module.weight.data, activation_stats[name])
                
                # Aplica a quantização (Fake Quant)
                new_weight = quantizer.quantize_layer(module.weight.data, scales)
                module.weight.data.copy_(new_weight)

    # 4. Test Generation
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: {prompt}")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Quantized Output: {response}")

if __name__ == "__main__":
    main()
