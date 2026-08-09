# LSTM seq2seq with Bahdanau attention for text simplification

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    # Bidirectional LSTM over the source; packs padding so the recurrence
    # only processes real tokens also bridges the bidirectional final
    # states down to a unidirectional decoder initial state.
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden=None,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        if dec_hidden is None:
            dec_hidden = enc_hidden
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, enc_hidden, batch_first=True,
                            bidirectional=True)
        # bridge: 2*enc_hidden -> dec_hidden, one Linear per of h and c
        self.h_bridge = nn.Linear(2 * enc_hidden, dec_hidden)
        self.c_bridge = nn.Linear(2 * enc_hidden, dec_hidden)

    def forward(self, src, src_lengths):
        # Reads input, takes the source token tensor, embeds it, packs it to skip padding and runs bidirection LSTM
        embedded = self.dropout(self.embedding(src))   # [batch, src_len, emb_dim]

        # enforce_sorted=False because src_lengths is not sorted
        packed = pack_padded_sequence(embedded, src_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        # pad back to [batch, src_len, 2*enc_hidden] (both directions per step)
        encoder_outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        # h/c: [2*num_layers, batch, enc_hidden]; top layer = last two entries
        # (with 1 layer: index 0 = forward, index 1 = backward)
        h_fwd, h_bwd = h[-2], h[-1]            # each [batch, enc_hidden]
        c_fwd, c_bwd = c[-2], c[-1]            # each [batch, enc_hidden]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # [batch, 2*enc_hidden]
        c_cat = torch.cat([c_fwd, c_bwd], dim=-1)   # [batch, 2*enc_hidden]
        # bridge + tanh, then unsqueeze to the LSTM's (num_layers, batch, dim)
        dec_h0 = torch.tanh(self.h_bridge(h_cat)).unsqueeze(0)  # [1, batch, dec_hidden]
        dec_c0 = torch.tanh(self.c_bridge(c_cat)).unsqueeze(0)  # [1, batch, dec_hidden]

        return encoder_outputs, (dec_h0, dec_c0)


class Attention(nn.Module):
    # Bahdanau (additive) attention over encoder outputs.
    def __init__(self, enc_hidden, dec_hidden, attn_dim=256):
        super().__init__()
        self.W_dec = nn.Linear(dec_hidden, attn_dim)
        self.W_enc = nn.Linear(2 * enc_hidden, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, dec_hidden, encoder_outputs, mask):
        # Figuring out which input words matter
        dec_proj = self.W_dec(dec_hidden).unsqueeze(1)   # [batch, 1, attn_dim]
        enc_proj = self.W_enc(encoder_outputs)           # [batch, src_len, attn_dim]
        energy = torch.tanh(dec_proj + enc_proj)         # [batch, src_len, attn_dim]
        scores = self.v(energy).squeeze(-1)              # [batch, src_len]

        # pad positions get -inf -> exactly zero weight after softmax
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)          # [batch, src_len]
        return weights


class Decoder(nn.Module):
    # One decoding timestep: embed token, attend, feed [embed; context] to LSTM.
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(enc_hidden, dec_hidden)
        self.lstm = nn.LSTM(emb_dim + 2 * enc_hidden, dec_hidden, batch_first=True)
        self.fc = nn.Linear(dec_hidden, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # Takes the current input token, embeds it, calls attention to build a context vector and feeds it through the decoder
        embedded = self.dropout(self.embedding(input))   # [batch, emb_dim]

        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)
        # context = weighted sum of encoder outputs: [batch, 2*enc_hidden]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        lstm_input = torch.cat([embedded, context], dim=-1)  # [batch, emb_dim+2*enc_hidden]
        lstm_out, (hidden, cell) = self.lstm(lstm_input.unsqueeze(1), (hidden, cell))
        # lstm_out: [batch, 1, dec_hidden]; hidden/cell stay [1, batch, dec_hidden]

        logits = self.fc(lstm_out.squeeze(1))             # [batch, vocab_size]
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, enc_hidden=512, dec_hidden=512,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, enc_hidden, dec_hidden,
                               dropout, pad_idx)
        self.decoder = Decoder(vocab_size, emb_dim, enc_hidden, dec_hidden,
                               dropout, pad_idx)
        self.pad_idx = pad_idx

    def forward(self, src, src_lengths, dst, teacher_forcing_ratio=1.0):
        # Orchestrating everything for training
        batch = src.size(0)
        dst_len = dst.size(1)
        vocab_size = self.decoder.fc.out_features

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        # encoder_outputs: [batch, src_len, 2*enc_hidden]
        # hidden/cell: [1, batch, dec_hidden]
        mask = src != self.pad_idx                          # [batch, src_len]

        logits = torch.zeros(batch, dst_len - 1, vocab_size, device=src.device)
        input = dst[:, 0]                                    # <sos>
        for t in range(dst_len - 1):
            step_logits, hidden, cell = self.decoder(
                input, hidden, cell, encoder_outputs, mask)
            logits[:, t] = step_logits
            # teacher forcing: feed the true next token or our own argmax
            if torch.rand(1).item() < teacher_forcing_ratio:
                input = dst[:, t + 1]
            else:
                input = step_logits.argmax(-1)
        return logits                                        # [batch, dst_len-1, vocab_size]

    @torch.no_grad()
    def greedy_decode(self, src, src_lengths, sos_idx, eos_idx, max_len=100):
        # Feeding the model's own top prediction back each step
        self.eval()
        batch = src.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        mask = src != self.pad_idx

        input = torch.full((batch,), sos_idx, dtype=torch.long, device=src.device)
        steps = []
        for _ in range(max_len):
            logits, hidden, cell = self.decoder(
                input, hidden, cell, encoder_outputs, mask)
            input = logits.argmax(-1)
            steps.append(input)                              # [batch] per step
        preds = torch.stack(steps, dim=1)                    # [batch, max_len]

        out = []
        for row in preds:
            seq = row.tolist()
            if eos_idx in seq:
                seq = seq[: seq.index(eos_idx) + 1]
            out.append(seq)
        return out


def count_parameters(model):
    # trainable parameter count, for the report
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from vocab import Vocab

    # tiny vocab
    lines = [["the", "cat", "sat", "on", "mat"],
             ["a", "dog", "ran", "fast"],
             ["the", "cat", "sat"],
             ["a", "dog", "ran", "quickly"]]
    vocab = Vocab(lines, min_freq=1)

    torch.manual_seed(0)
    batch, src_len, dst_len = 3, 6, 6
    src = torch.randint(4, len(vocab), (batch, src_len))
    src[:, 0] = vocab.stoi["the"]                            # keep first token real
    src_lengths = torch.tensor([6, 4, 3])
    for i, length in enumerate(src_lengths):                # pad past true length
        src[i, length:] = vocab.pad_idx
    dst = torch.randint(4, len(vocab), (batch, dst_len))
    dst[:, 0] = vocab.sos_idx
    dst[:, -1] = vocab.eos_idx

    model = Seq2Seq(len(vocab), emb_dim=16, enc_hidden=24, dec_hidden=24,
                    dropout=0.1, pad_idx=vocab.pad_idx)

    logits = model(src, src_lengths, dst, teacher_forcing_ratio=1.0)
    assert logits.shape == (batch, dst_len - 1, len(vocab)), logits.shape

    # attention masking: weights over pad positions must be ~0
    dec_h = torch.randn(batch, 24)
    enc_out = torch.randn(batch, src_len, 48)
    mask = src != vocab.pad_idx
    weights = model.decoder.attention(dec_h, enc_out, mask)
    assert torch.allclose(weights[~mask], torch.zeros_like(weights[~mask]),
                          atol=1e-6), "pad positions got attention"
    assert torch.allclose(weights.sum(-1), torch.ones(batch), atol=1e-5)

    # greedy decode runs without error
    preds = model.greedy_decode(src, src_lengths, vocab.sos_idx, vocab.eos_idx,
                                max_len=10)
    assert len(preds) == batch
    assert all(isinstance(p, list) for p in preds)

    print("logits shape:", tuple(logits.shape))
    print("attention weights row 0:", weights[0].tolist())
    print("greedy preds:", preds)
    print("trainable parameters:", count_parameters(model))
    print("test passed")
