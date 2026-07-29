import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.calibration.dataset import CalibrationDataset
from src.calibration.collector import ActivationCollector
from src.quantization.awq import AWQuantizer
from src.quantization.gptq import GPTQuantizer

def main():
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    device = "cuda"
    
    print("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = CalibrationDataset(model_id, n_samples=32)
    collector = ActivationCollector(device=device)
    stats, hessians = collector.collect(model, dataset)

    # Vamos testar GPTQ primeiro pois ele altera os pesos in-place
    gptq = GPTQuantizer(bits=4, group_size=128)
    print("\nApplying GPTQ...")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in hessians:
            new_weight = gptq.quantize_layer(module.weight.data, hessians[name])
            module.weight.data.copy_(new_weight)

    prompt = "The key to learning machine learning is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    print(f"\nGPTQ Result: {tokenizer.decode(out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
