import json

def save_tokenizer(tokenizer, path="tokenizer.json"):
    data = {
        "vocab": {k.decode("latin1"): v for k, v in tokenizer.vocab.items()},
        "merges": [[a.decode("latin1"), b.decode("latin1")] for a, b in tokenizer.merges],
        "special_tokens": tokenizer.special_tokens
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
