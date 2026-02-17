from collections import Counter
from typing import List, Tuple
from .bpe_tokenizer import BPETokenizer

class BPETrainer:
    def __init__(self):
        self.vocab = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.special_tokens = {"<pad>": 256, "<bos>": 257, "<eos>": 258, "<unk>": 259}

    def _get_pair_frequencies(self, tokens_list: List[List[bytes]]) -> Counter:
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_pair(self, tokens_list: List[List[bytes]], pair_to_merge: Tuple[bytes, bytes]) -> Tuple[List[List[bytes]], bytes]:
        a, b = pair_to_merge
        new_token = a + b
        new_tokens_list = []

        for tokens in tokens_list:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens_list.append(new_tokens)

        return new_tokens_list, new_token

    def train(self, text: str, vocab_size: int) -> BPETokenizer:
        # Step 1: convert text to bytes
        tokens_list = [[bytes([b]) for b in text.encode("utf-8")]]

        # Step 2: init vocab
        self.vocab = {bytes([i]): i for i in range(256)}
        self.vocab.update({bytes(t, "utf-8"): id_ for t, id_ in self.special_tokens.items()})

        # Step 3: BPE loop
        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_frequencies(tokens_list)
            if not pair_freqs:
                break
            most_common_pair = pair_freqs.most_common(1)[0][0]
            tokens_list, new_token = self._merge_pair(tokens_list, most_common_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(most_common_pair)

        # Step 4: return tokenizer
        return BPETokenizer(self.vocab, self.merges, self.special_tokens)
