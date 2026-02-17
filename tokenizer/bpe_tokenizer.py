from typing import List, Dict, Tuple
from .base import Tokenizer

class BPETokenizer(Tokenizer):
    def __init__(self, vocab: Dict[bytes, int], merges: List[Tuple[bytes, bytes]], special_tokens: Dict[str, int]):
        """
        vocab: mapping token (bytes) -> id
        merges: list of tuples representing merges
        special_tokens: dict of special token names to ids (like <pad>, <bos>, etc.)
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # reverse vocab for decoding
        self.id_to_token = {id_: token for token, id_ in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode string into list of token ids applying BPE merges and special tokens
        """
        # Step 1: start with bytes
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        # Step 2: apply merges sequentially
        for a, b in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == (a, b):
                    new_tokens.append(a + b)  # merge
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Step 3: convert tokens to ids (use <unk> for unknown tokens)
        ids = [self.vocab.get(t, self.special_tokens["<unk>"]) for t in tokens]

        # Step 4: add special tokens
        ids = [self.special_tokens["<bos>"]] + ids + [self.special_tokens["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode list of token ids back into string, ignoring special tokens
        """
        bytes_list = [self.id_to_token.get(i, b"?") for i in ids if i not in self.special_tokens.values()]
        flat_bytes = b"".join(bytes_list)
        return flat_bytes.decode("utf-8", errors="replace")
