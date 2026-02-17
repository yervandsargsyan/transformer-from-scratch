import json
import base64
from tokenizer.bpe_tokenizer import BPETokenizer

def load_tokenizer(path="tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = {}
    for k, v in data["vocab"].items():
        try:
            decoded_key = base64.b64decode(k.encode("ascii"))
        except Exception:
            decoded_key = k.encode("utf-8") 
        vocab[decoded_key] = v

    merges = []
    for a, b in data["merges"]:
        try:
            a_bytes = base64.b64decode(a.encode("ascii"))
        except Exception:
            a_bytes = a.encode("utf-8")
        try:
            b_bytes = base64.b64decode(b.encode("ascii"))
        except Exception:
            b_bytes = b.encode("utf-8")
        merges.append((a_bytes, b_bytes))

    special_tokens = data["special_tokens"]

    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    if tokenizer is None or not hasattr(tokenizer, "vocab"):
        raise ValueError("Error: The tokenizer was not initialized correctly!")

    return tokenizer
