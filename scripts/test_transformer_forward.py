import torch
from model.transformer import Transformer

def main():
    vocab_size = 300 
    seq_len = 20
    batch_size = 4

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = Transformer(vocab_size=vocab_size)

    # forward pass
    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)  #[batch, seq_len, vocab_size]

if __name__ == "__main__":
    main()
