import torch
import torch.nn as nn
from model.transformer import Transformer
from scripts.load_tokenizer import load_tokenizer

def main():
    # Load trained tokenizer
    tokenizer = load_tokenizer("tokenizer.json")
    
    # Parameters
    vocab_size = len(tokenizer.vocab)  # correct vocab size from tokenizer
    seq_len = 20
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # Read corpus and encode
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    all_ids = tokenizer.encode(text)

    # Make sure we have enough tokens
    needed_tokens = batch_size * seq_len
    if len(all_ids) < needed_tokens:
        # pad with <pad> token if not enough
        pad_id = tokenizer.special_tokens["<pad>"]
        all_ids += [pad_id] * (needed_tokens - len(all_ids))

    # Create input tensor
    x = torch.tensor(all_ids[:needed_tokens], dtype=torch.long).view(batch_size, seq_len).to(device)

    # Targets (toy example: next-token prediction)
    y = x.clone()

    # Create model
    model = Transformer(vocab_size=vocab_size).to(device)

    # Forward pass
    logits = model(x)  # [batch, seq_len, vocab_size]

    # Compute loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    print("Forward done. Loss:", loss.item())

if __name__ == "__main__":
    main()
