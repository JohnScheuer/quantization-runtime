import torch
import torch.nn as nn

def apply_fake_quant(model, quantizer, activation_stats):
    """Substitui os pesos do modelo por suas versões quantizadas."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in activation_stats:
            print(f"Quantizing layer: {name}")
            scales = activation_stats[name]
            new_weight = quantizer.quantize_layer(module.weight.data, scales)
            module.weight.data.copy_(new_weight)
    return model
