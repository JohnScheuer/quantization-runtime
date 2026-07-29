import torch

class GPTQuantizer:
    def __init__(self, bits=4, group_size=128, damp_percent=0.1): # Aumentado damping para 10%
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.max_q = 2**(bits-1) - 1
        self.min_q = -(2**(bits-1))

    @torch.no_grad()
    def quantize_layer(self, weight, hessian):
        W = weight.clone().float()
        H = hessian.clone().float()
        
        # 1. Damping agressivo para estabilidade em modelos pequenos
        damp = self.damp_percent * torch.diag(H).mean()
        diag = torch.arange(W.shape[1], device=W.device)
        H[diag, diag] += damp
        
        # 2. Inversão com tratamento de erro
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except RuntimeError:
            # Se a Hessiana ainda for instável, retorna o peso original
            return weight

        Q = torch.zeros_like(W)
        
        # 3. Quantização com proteção de overflow
        for i1 in range(0, W.shape[1], self.group_size):
            i2 = min(i1 + self.group_size, W.shape[1])
            W_block = W[:, i1:i2]
            Hinv_block = Hinv[i1:i2, i1:i2]
            
            scale = (W_block.abs().max() / self.max_q).clamp(min=1e-5)
            q_block = (W_block / scale).round().clamp(self.min_q, self.max_q)
            deq_block = q_block * scale
            Q[:, i1:i2] = deq_block
            
            error = (W_block - deq_block)
            if i2 < W.shape[1]:
                # Otimização de propagação: Error @ Hinv_correlation
                update = error @ (Hinv_block.t() @ Hinv[i1:i2, i2:])
                # Clipping de segurança para evitar explosão
                W[:, i2:] += update.clamp(min=-1.0, max=1.0) 

        return Q.to(weight.dtype)
