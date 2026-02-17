from tokenizer.bpe_trainer import BPETrainer

def main():
    text = "hello world"
    trainer = BPETrainer()
    tokenizer = trainer.train(text, vocab_size=300)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Original text:", text)
    print("Encoded ids:", ids)
    print("Decoded text:", decoded)
    print("Merges:", tokenizer.merges)

    # Проверка
    assert decoded == text, "Decode != original text"
    print("Test passed!")

if __name__ == "__main__":
    main()
