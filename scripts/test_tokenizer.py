# scripts/test_tokenizer.py
from tokenizer.bpe_trainer import BPETrainer

text = "hello world"
trainer = BPETrainer()
tokenizer = trainer.train(text, vocab_size=3000)

ids = tokenizer.encode(text)
print("Encoded ids:", ids)

decoded = tokenizer.decode(ids)
print("Decoded text:", decoded)
