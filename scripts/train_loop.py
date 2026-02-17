import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from scripts.load_tokenizer import load_tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters
    seq_len = 50
    stride = 1        # window shift
    batch_size = 4
    n_epochs = 10    # number of epochs for testing

    # Load tokenizer
    tokenizer = load_tokenizer("tokenizer.json")
    vocab_size = len(tokenizer.vocab)

    # Read corpus and encode
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
    all_ids = tokenizer.encode(text)

    # Pad corpus if needed
    if len(all_ids) < seq_len:
        all_ids += [tokenizer.special_tokens["<pad>"]] * (seq_len - len(all_ids))

    # Convert to tensor
    all_ids = torch.tensor(all_ids, dtype=torch.long)

    # Create model and optimizer
    model = Transformer(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # --- Mini-step 1: create sliding-window batches ---
    x_batches = []
    y_batches = []
    for i in range(0, len(all_ids) - seq_len, stride):
        x_batches.append(all_ids[i:i+seq_len])
        y_batches.append(all_ids[i+1:i+seq_len+1])

    # Convert to tensors
    x_batches = torch.stack(x_batches).to(device)  # [num_batches, seq_len]
    y_batches = torch.stack(y_batches).to(device)  # [num_batches, seq_len]

    # Training loop
    for epoch in range(n_epochs):
        total_loss = 0
        for i in range(0, len(x_batches), batch_size):
            x = x_batches[i:i+batch_size]
            y = y_batches[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(x_batches)
        print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.4f}")

        # Save model each epoch
        torch.save(model.state_dict(), f"transformer_epoch{epoch+1}.pth")
        print(f"Model saved: transformer_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
