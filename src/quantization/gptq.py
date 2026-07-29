import torch

class GPTQuantizer:
    def __init__(self, bits=4, group_size=128, damp_percent=0.2):
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.max_q = 2**(bits-1) - 1
        self.min_q = -(2**(bits-1))

    @torch.no_grad()
    def quantize_layer(self, weight, hessian, act_stats):
        # 1. AWQ-Style Pre-scaling (Estabilização fundamental)
        scales = act_stats.pow(0.5).clamp(min=1e-4)
        scales = scales / (scales.max() + 1e-4)
        W = (weight * scales.view(1, -1)).to(torch.float64)
        H = (hessian / (scales.view(-1, 1) @ scales.view(1, -1) + 1e-6)).to(torch.float64)

        num_cols = W.shape[1]
        Q = torch.zeros_like(W)

        for i1 in range(0, num_cols, self.group_size):
            i2 = min(i1 + self.group_size, num_cols)
            W_group = W[:, i1:i2]
            H_group = H[i1:i2, i1:i2]
            
            damp = self.damp_percent * torch.diag(H_group).mean()
            H_group[torch.arange(i2-i1), torch.arange(i2-i1)] += damp
            
            try:
                Hinv_group = torch.linalg.inv(H_group)
            except RuntimeError:
                scale = (W_group.abs().max() / self.max_q).clamp(min=1e-5)
                Q[:, i1:i2] = (W_group / scale).round().clamp(self.min_q, self.max_q) * scale
                continue

            group_scale = (W_group.abs().max() / self.max_q).clamp(min=1e-5)

            for i in range(i2 - i1):
                w_col = W_group[:, i]
                h_inv_ii = Hinv_group[i, i]
                q_col = (w_col / group_scale).round().clamp(self.min_q, self.max_q)
                deq_col = q_col * group_scale
                Q[:, i1 + i] = deq_col

                error = (w_col - deq_col)
                if i < (i2 - i1) - 1:
                    adjustment = error.unsqueeze(1) @ (Hinv_group[i, i+1:] / h_inv_ii).unsqueeze(0)
                    # TÉCNICA DE ELITE: Amortecimento de Erro (0.85)
                    # Passamos apenas 85% do erro para a próxima coluna. 
                    # Isso estabiliza modelos pequenos que não têm "espaço" para compensar erros grandes.
                    W_group[:, i+1:] += adjustment * 0.85 

        final_q = (Q / scales.view(1, -1)).to(torch.float16)
        
        # GATE DE SEGURANÇA: Se o erro da camada for absurdo, usa RTN (visto em kernels de produção)
        if torch.isnan(final_q).any() or (final_q - weight).abs().mean() > weight.abs().mean():
            scale = (weight.abs().max() / self.max_q).clamp(min=1e-5)
            return (weight / scale).round().clamp(self.min_q, self.max_q) * scale
            
        return final_q
