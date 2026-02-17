import torch
from model.transformer import Transformer

def main():
    vocab_size = 300  # должен совпадать с твоим токенизатором
    seq_len = 20
    batch_size = 4

    # 1️⃣ случайный батч токенов
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 2️⃣ создаём модель
    model = Transformer(vocab_size=vocab_size)

    # 3️⃣ forward pass
    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)  # должно быть [batch, seq_len, vocab_size]

if __name__ == "__main__":
    main()
