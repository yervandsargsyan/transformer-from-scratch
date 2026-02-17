from tokenizer.bpe_trainer import BPETrainer
from scripts.save_tokenizer import save_tokenizer

def main():
    # 1️⃣ Read corpus
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Corpus length:", len(text))

    # 2️⃣ Train BPE tokenizer
    trainer = BPETrainer()
    tokenizer = trainer.train(text, vocab_size=1000)  # можно увеличить vocab_size

    print("Tokenizer trained. Number of merges:", len(tokenizer.merges))

    # 3️⃣ Save tokenizer
    save_tokenizer(tokenizer, path="tokenizer.json")
    print("Tokenizer saved to tokenizer.json")

if __name__ == "__main__":
    main()
