from tokenizer.bpe_trainer import BPETrainer
from scripts.save_tokenizer import save_tokenizer
from scripts.load_tokenizer import load_tokenizer

def main():
    text = "hello world"

    # 1️⃣ Train tokenizer
    trainer = BPETrainer()
    tokenizer = trainer.train(text, vocab_size=300)
    print("Tokenizer trained. Merges:", tokenizer.merges)

    # 2️⃣ Save tokenizer
    save_tokenizer(tokenizer, path="tokenizer.json")
    print("Tokenizer saved to tokenizer.json")

    # 3️⃣ Load tokenizer
    loaded_tokenizer = load_tokenizer(path="tokenizer.json")
    print("Tokenizer loaded from tokenizer.json")

    # 4️⃣ Test encode/decode
    ids = loaded_tokenizer.encode(text)
    decoded = loaded_tokenizer.decode(ids)

    print("Original text:", text)
    print("Encoded ids:", ids)
    print("Decoded text:", decoded)

    assert decoded == text, "Decode != original text"
    print("Save/Load test passed!")

if __name__ == "__main__":
    main()
