# building numerical representations of vocabulary for training

def read_file(path):
    # Read one pre-tokenized file into a list of token-lists (one per line)
    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            lines.append(line.split())
    return lines


class Vocab:
    def __init__(self, token_lists, min_freq=2,
                 specials=("<pad>", "<unk>", "<sos>", "<eos>")):
        # Four special tokens: Padding, Unknown, Start of Sequence, End of Sequence

        # count every token across all input sentences
        counts = {}
        for line in token_lists:
            for token in line:
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1

        # build itos: specials first (locks indices 0-3), then tokens >= min_freq
        self.itos = list(specials)
        for token, freq in counts.items():
            if freq >= min_freq and token not in specials:
                self.itos.append(token)

        # 3. derive stoi from the finished itos so they can never drift out of sync
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        # 4. store special indices by name so nothing downstream hardcodes numbers
        self.pad_idx = self.stoi["<pad>"]
        self.unk_idx = self.stoi["<unk>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        # tokens (list of strings) -> list of ints, unknown tokens -> unk_idx."""
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices):
        # indices (list of ints) -> list of strings."""
        return [self.itos[i] for i in indices]


if __name__ == "__main__":
    # test
    v = Vocab([["the", "cat", "sat"], ["the", "dog", "ran"]], min_freq=1)
    assert v.pad_idx == 0 and v.unk_idx == 1
    assert v.decode(v.encode(["the", "cat"])) == ["the", "cat"]  # round-trips
    assert v.encode(["zebra"]) == [v.unk_idx]                    # unk fallback
    print("vocab size:", len(v))
    print("first 6:", v.itos[:6])
    print("smoke test passed")