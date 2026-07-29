import torch
import torch.nn as nn
from tqdm import tqdm

class ActivationCollector:
    def __init__(self, device="cuda"):
        self.device = device
        self.stats = {}
        self.hessians = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            
            # AWQ Stats
            act_mean = x.abs().mean(dim=0)
            if name not in self.stats: self.stats[name] = act_mean
            else: self.stats[name] = 0.9 * self.stats[name] + 0.1 * act_mean

            # GPTQ Stats (Hessiana Normalizada)
            # Normalizamos pelo número de tokens para manter a escala estável
            xtx = (x.t() @ x) / x.shape[0] 
            if name not in self.hessians: self.hessians[name] = xtx
            else: self.hessians[name] = 0.9 * self.hessians[name] + 0.1 * xtx
        return hook_fn

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def collect(self, model, dataloader):
        model.eval()
        self.register(model)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting stats"):
                model(batch.to(self.device))
        self.remove_hooks()
        return self.stats, self.hessians

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
