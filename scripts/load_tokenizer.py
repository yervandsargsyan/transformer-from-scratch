import json

def load_tokenizer(path="tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = {k.encode("latin1"): v for k, v in data["vocab"].items()}
    merges = [(a.encode("latin1"), b.encode("latin1")) for a, b in data["merges"]]
    special_tokens = data["special_tokens"]

    from tokenizer.bpe_tokenizer import BPETokenizer
    return BPETokenizer(vocab, merges, special_tokens)

