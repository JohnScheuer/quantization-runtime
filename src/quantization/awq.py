import torch
import torch.nn as nn

class AWQuantizer:
    def __init__(self, bits=4, group_size=128):
        self.bits = bits
        self.group_size = group_size
        self.max_q = 2**(bits-1) - 1
        self.min_q = -(2**(bits-1))

    @torch.no_grad()
    def get_weight_scale(self, weight, activation_abs_mean):
        """Calcula o fator de escala AWQ para proteger pesos salientes."""
        # Se as ativações são grandes, o peso é importante.
        # scale = (act_mean / max_act_mean) ^ alpha
        # Para simplificar o MVP, usamos alpha=0.5 (padrão AWQ)
        scales = activation_abs_mean.pow(0.5).clamp(min=1e-4)
        scales = scales / (scales.max() + 1e-4)
        return scales

    def quantize_layer(self, weight, scales):
        """Quantização por grupo (Group-wise)."""
        # Aplica a proteção AWQ (pesos importantes ficam maiores antes de arredondar)
        w_protected = weight * scales.view(1, -1)
        
        # Quantização simétrica básica por grupo
        orig_shape = w_protected.shape
        w_reshaped = w_protected.view(-1, self.group_size)
        
        # Encontra a escala por grupo
        max_val = w_reshaped.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
        scale = max_val / self.max_q
        
        # Quantiza e Dequantiza (Simula o erro de INT4 em FP16)
        q_w = (w_reshaped / scale).round().clamp(self.min_q, self.max_q)
        deq_w = q_w * scale
        
        # Reverte a proteção AWQ para voltar à escala original do modelo
        final_w = deq_w.view(orig_shape) / scales.view(1, -1)
        return final_w
