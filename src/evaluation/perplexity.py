import torch
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer, dataset, device="cuda"):
    """
    Calcula a Perplexity: exp(média da CrossEntropyLoss).
    Uma perplexity baixa indica que o modelo 'entende' bem o texto.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    print("\nEvaluating Perplexity...")
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Eval"):
            input_ids = batch.to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift para alinhar labels e logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()
