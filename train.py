# training loop for the LSTM seq2seq simplifier: ties together vocab, data, model

import argparse
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from vocab import Vocab, read_file
from data import SimplificationDataset, make_collate_fn
from model import Seq2Seq, count_parameters


def set_seed(seed):
    # reproducible runs: seed random, numpy, torch (and cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_loader, criterion, optimizer, device,
                teacher_forcing_ratio):
    model.train()
    total_loss, n_batches = 0.0, 0
    for src, src_lengths, dst in train_loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        dst = dst.to(device)

        optimizer.zero_grad()
        logits = model(src, src_lengths, dst,
                       teacher_forcing_ratio=teacher_forcing_ratio)
        # logits [B, T-1, V] aligned with dst[:, 1:] (see model.py convention)
        target = dst[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, valid_loader, criterion, device):
    # average validation loss
    # is computed on teacher-forced logits either way (we only need a
    # comparable number, not free-running output), and no_grad saves memory.
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for src, src_lengths, dst in valid_loader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            dst = dst.to(device)
            logits = model(src, src_lengths, dst, teacher_forcing_ratio=1.0)
            target = dst[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


def monitor_copy_trap(model, dataset, vocab, device, n=5, max_len=100):
    # greedy-decode n fixed validation examples and compare to the source,
    # to check the model isn't just echoing its input.
    # NOTE: greedy_decode sets model.eval() internally; the caller must call
    # model.train() afterwards before resuming training.
    n = min(n, len(dataset))
    srcs, lengths, refs = [], [], []
    for i in range(n):
        src_t, _ = dataset[i]
        srcs.append(src_t)
        lengths.append(len(src_t))
        refs.append(dataset.sources[i])
    src_padded = pad_sequence(srcs, batch_first=True,
                              padding_value=vocab.pad_idx).to(device)
    # keep src_lengths on the same device as src_padded, matching the
    # training loop; the encoder's src_lengths.cpu() handles pack_padded_sequence
    src_lengths = torch.tensor(lengths, dtype=torch.long).to(device)

    preds = model.greedy_decode(src_padded, src_lengths, vocab.sos_idx,
                                vocab.eos_idx, max_len=max_len)

    identical = 0
    for src_tokens, pred in zip(refs, preds):
        # strip the trailing <eos> (if any) before comparing to the source
        pred_trimmed = pred[:-1] if pred and pred[-1] == vocab.eos_idx else pred
        if pred_trimmed == vocab.encode(src_tokens):
            identical += 1
        print(f"    src: {' '.join(src_tokens)}")
        print(f"    out: {' '.join(vocab.decode(pred))}")
    ratio = identical / n
    print(f"    identical_ratio on {n} samples: {ratio:.2f}")
    return ratio


def main():
    parser = argparse.ArgumentParser(
        description="Train the LSTM seq2seq text simplifier.")
    parser.add_argument("--data-dir", default="data",
                        help="dir with train.src/train.dst/valid.src/valid.ref.0")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--enc-hidden", type=int, default=512)
    parser.add_argument("--dec-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3,
                        help="early stop after this many epochs without val improvement")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    # shared vocab from training data only
    train_src = os.path.join(args.data_dir, "train.src")
    train_dst = os.path.join(args.data_dir, "train.dst")
    valid_src = os.path.join(args.data_dir, "valid.src")
    valid_ref = os.path.join(args.data_dir, "valid.ref.0")
    lines = read_file(train_src) + read_file(train_dst)
    vocab = Vocab(lines, min_freq=args.min_freq)
    print(f"vocab size: {len(vocab)}")

    train_ds = SimplificationDataset(train_src, train_dst, vocab)
    valid_ds = SimplificationDataset(valid_src, valid_ref, vocab)

    collate = make_collate_fn(vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    model = Seq2Seq(len(vocab), emb_dim=args.emb_dim, enc_hidden=args.enc_hidden,
                    dec_hidden=args.dec_hidden, dropout=args.dropout,
                    pad_idx=vocab.pad_idx).to(device)
    print(f"trainable parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    start = time.time()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 device, args.teacher_forcing_ratio)
        val_loss = evaluate(model, valid_loader, criterion, device)
        epochs_run = epoch
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,  # pickled so the eval script can rebuild exactly
                "hyperparams": {
                    "emb_dim": args.emb_dim,
                    "enc_hidden": args.enc_hidden,
                    "dec_hidden": args.dec_hidden,
                    "dropout": args.dropout,
                    "pad_idx": vocab.pad_idx,
                    "vocab_size": len(vocab),
                },
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"  saved best model (val loss {val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # copy-trap qualitative check on fixed validation examples
        print("  sample decodes:")
        monitor_copy_trap(model, valid_ds, vocab, device, n=5)
        model.train()  # undo the eval() that greedy_decode set

        if epochs_no_improve >= args.patience:
            print(f"  early stopping: no val improvement for "
                  f"{args.patience} epochs")
            break

    total_time = time.time() - start
    hardware = (torch.cuda.get_device_name(0) if device.type == "cuda"
                else platform.processor())
    summary = [
        f"device: {device}",
        f"hardware: {hardware}",
        f"vocab size: {len(vocab)}",
        f"trainable parameters: {count_parameters(model)}",
        f"epochs run: {epochs_run}",
        f"best val loss: {best_val_loss:.4f}",
        f"total wall-clock time: {total_time:.1f}s",
        f"hyperparams: emb_dim={args.emb_dim}, enc_hidden={args.enc_hidden}, "
        f"dec_hidden={args.dec_hidden}, dropout={args.dropout}, "
        f"lr={args.lr}, batch_size={args.batch_size}, "
        f"teacher_forcing_ratio={args.teacher_forcing_ratio}",
        f"checkpoint: {checkpoint_path}",
    ]
    print("\n".join(summary))
    with open(os.path.join(args.checkpoint_dir, "training_log.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
