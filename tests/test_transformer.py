import torch
from model.transformer import Transformer

def test_forward_pass():
    vocab_size = 100
    seq_len = 10
    batch_size = 2
    model = Transformer(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size), "Forward pass output shape incorrect"
