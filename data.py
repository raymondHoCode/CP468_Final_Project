# data layer - line-aligned src/dst files -> padded tensor batches

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocab import Vocab, read_file


class SimplificationDataset(Dataset):
    # Pairs line N of src_path with line N of dst_path as encoded tensors
    def __init__(self, src_path, dst_path, vocab):
        self.vocab = vocab
        self.sources = read_file(src_path)
        self.targets = read_file(dst_path)
        assert len(self.sources) == len(self.targets), \
            "src and dst files must have the same number of lines"

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        # source: plain encode, no wrapping
        src_tokens = self.sources[idx]
        src_tensor = torch.tensor(self.vocab.encode(src_tokens), dtype=torch.long)

        # target: wrapped with <sos> ... <eos> so the decoder learns start/stop
        dst_tokens = [self.vocab.sos_idx] + self.vocab.encode(self.targets[idx]) \
                     + [self.vocab.eos_idx]
        dst_tensor = torch.tensor(dst_tokens, dtype=torch.long)

        return src_tensor, dst_tensor


def make_collate_fn(pad_idx):
    # Build a collate_fn that pads a batch to its own max length (batch_first)
    def collate_fn(batch):
        sources, targets = zip(*batch)

        # unpadded lengths, needed for pack_padded_sequence in the encoder
        src_lengths = torch.tensor([len(s) for s in sources], dtype=torch.long)

        src_padded = pad_sequence(sources, batch_first=True, padding_value=pad_idx)
        dst_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)

        return src_padded, src_lengths, dst_padded

    return collate_fn


if __name__ == "__main__":
    # test - tiny synthetic pair on disk -> vocab -> dataset -> one batch
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.txt")
        dst_path = os.path.join(tmp, "dst.txt")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("the cat sat on the mat\n")
            f.write("a dog ran fast\n")
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write("the cat sat\n")
            f.write("a dog ran quickly\n")

        lines = read_file(src_path) + read_file(dst_path)
        vocab = Vocab(lines, min_freq=1)
        dataset = SimplificationDataset(src_path, dst_path, vocab)

        collate_fn = make_collate_fn(vocab.pad_idx)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        src_padded, src_lengths, dst_padded = next(iter(loader))

        # target rows start with <sos> and end with <eos> (before padding)
        assert dst_padded[:, 0].tolist() == [vocab.sos_idx] * 2
        for row in dst_padded:
            last_non_pad = row[row != vocab.pad_idx][-1].item()
            assert last_non_pad == vocab.eos_idx

        # src_padded is [batch, max_len]
        assert src_padded.shape[0] == 2
        assert src_padded.shape[1] == max(len(s) for s in dataset.sources)

        # src_lengths matches the true unpadded lengths
        true_lengths = [len(s) for s in dataset.sources]
        assert src_lengths.tolist() == true_lengths

        print("src_padded:", src_padded.tolist())
        print("src_lengths:", src_lengths.tolist())
        print("dst_padded:", dst_padded.tolist())
        print("smoke test passed")
