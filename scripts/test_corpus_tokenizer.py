from scripts.load_tokenizer import load_tokenizer

def main():
    text = "He unaffected sympathize discovered at no am conviction principles."

    # 1️⃣ Load trained tokenizer
    tokenizer = load_tokenizer("tokenizer.json")

    # 2️⃣ Encode / Decode
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Original text:", text)
    print("Encoded ids:", ids)
    print("Decoded text:", decoded)

    assert decoded == text, "Decode != original text"
    print("Corpus tokenizer test passed!")

if __name__ == "__main__":
    main()
