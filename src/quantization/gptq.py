import torch
import math

class GPTQuantizer:
    def __init__(self, bits=4, group_size=128, damp_percent=0.01):
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.max_q = 2**(bits-1) - 1
        self.min_q = -(2**(bits-1))

    @torch.no_grad()
    def quantize_layer(self, weight, hessian):
        """
        Algoritmo GPTQ simplificado:
        W_quant = quantize(W + Error * Hessian_inv)
        """
        W = weight.clone().float()
        H = hessian.clone().float()
        num_features = W.shape[1]
        
        # 1. Estabilização da Hessiana (Damping)
        damp = self.damp_percent * torch.mean(torch.diag(H))
        diag = torch.arange(num_features, device=W.device)
        H[diag, diag] += damp
        
        # 2. Inversão da Hessiana via Cholesky (mais estável)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H # Matriz triangular superior para propagação de erro

        # 3. Quantização Bloco a Bloco (Group-wise)
        Q = torch.zeros_like(W)
        
        for i1 in range(0, num_features, self.group_size):
            i2 = min(i1 + self.group_size, num_features)
            count = i2 - i1
            
            W_block = W[:, i1:i2]
            Hinv_block = Hinv[i1:i2, i1:i2]
            
            # Encontra escala para o bloco
            scale = (W_block.abs().max() / self.max_q).clamp(min=1e-5)
            
            # Quantiza bloco e calcula erro
            q_block = (W_block / scale).round().clamp(self.min_q, self.max_q)
            deq_block = q_block * scale
            Q[:, i1:i2] = deq_block
            
            # Propaga o erro para o resto da matriz usando a Hessiana inversa
            error = (W_block - deq_block)
            if i2 < num_features:
                # W_rest = W_rest + error * Hinv_correlation
                W[:, i2:] += error @ (Hinv_block.t() @ Hinv[i1:i2, i2:])

        return Q.to(weight.dtype)
