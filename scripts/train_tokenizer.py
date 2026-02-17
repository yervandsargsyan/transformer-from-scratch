from tokenizer.bpe_trainer import BPETrainer
from scripts.save_tokenizer import save_tokenizer

def main():
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    trainer = BPETrainer()
    tokenizer = trainer.train(text, vocab_size=300)

    save_tokenizer(tokenizer, "tokenizer.json")
    print("Tokenizer trained and saved to tokenizer.json")

if __name__ == "__main__":
    main()
