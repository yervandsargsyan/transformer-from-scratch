import json
import base64

def save_tokenizer(tokenizer, path="tokenizer.json"):
    data = {
        "vocab": {
            base64.b64encode(k).decode("ascii") if isinstance(k, bytes) else k: v
            for k, v in tokenizer.vocab.items()
        },
        "merges": [
            [
                base64.b64encode(a).decode("ascii") if isinstance(a, bytes) else a,
                base64.b64encode(b).decode("ascii") if isinstance(b, bytes) else b
            ]
            for a, b in tokenizer.merges
        ],
        "special_tokens": tokenizer.special_tokens
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
