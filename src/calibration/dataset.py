from datasets import load_dataset
from transformers import AutoTokenizer
import torch

class CalibrationDataset:
    def __init__(self, model_name, n_samples=32, seq_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_length = seq_length
        print(f"Loading calibration data (Salesforce/wikitext)...")
        
        # Usando o nome completo para evitar erros de URI
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        
        text = "\n\n".join(dataset["text"])
        self.tokens = self.tokenizer(text, return_tensors="pt").input_ids[0]
        self.n_samples = n_samples

    def __iter__(self):
        # Garante que não vamos exceder o tamanho do dataset
        max_samples = len(self.tokens) // self.seq_length
        actual_samples = min(self.n_samples, max_samples)
        
        for i in range(actual_samples):
            start = i * self.seq_length
            end = start + self.seq_length
            yield self.tokens[start:end].unsqueeze(0)
