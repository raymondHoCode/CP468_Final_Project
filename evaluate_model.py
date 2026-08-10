# evaluate_model.py - final evaluation harness for the LSTM seq2seq simplifier.
#
# Loads a trained checkpoint, generates simplifications for the ASSET test set,
# and scores them with SARI (primary), BLEU (secondary), and identical_ratio
# (copy-trap). Also scores a copy baseline (output = input) and, optionally, an
# LLM baseline's outputs through the same metric functions, so all systems are
# judged by exactly the same pipeline.
#
# Metrics:
#   - SARI: EASSE if importable, else a vendored port of EASSE's corpus
#     SARI (lowercase + sacrebleu 13a tokenization, macro/JAVA variant, F1 for
#     deletion), verified to match EASSE's corpus_sari() exactly.
#     NOTE: EASSE is GitHub-only
#       (pip install git+https://github.com/feralvam/easse.git)
#     and pins old deps (nltk==3.6.2, spacy, ...) that can fail on newer Python,
#     so we vendor its exact algorithm as the fallback instead of depending on it.
#     sacrebleu provides BLEU but NOT SARI.

import argparse
import os
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from vocab import Vocab, read_file
from model import Seq2Seq


def load_checkpoint(checkpoint_path, device):
    # weights_only=False: the checkpoint embeds a pickled Vocab object.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]  # reload exactly as trained - do NOT rebuild from data
    hp = ckpt["hyperparams"]
    model = Seq2Seq(hp["vocab_size"], emb_dim=hp["emb_dim"],
                    enc_hidden=hp["enc_hidden"], dec_hidden=hp["dec_hidden"],
                    dropout=hp["dropout"], pad_idx=hp["pad_idx"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, ckpt


def generate_predictions(model, vocab, sources, device, batch_size=32, max_len=100):
    # sources: list of token-lists (from read_file). Returns list of strings,
    # one per source, in the same order.
    preds = []
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        tensors = [torch.tensor(vocab.encode(tokens), dtype=torch.long)
                   for tokens in batch]
        src_lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
        src_padded = pad_sequence(tensors, batch_first=True,
                                  padding_value=vocab.pad_idx).to(device)
        src_lengths = src_lengths.to(device)

        idx_seqs = model.greedy_decode(src_padded, src_lengths, vocab.sos_idx,
                                       vocab.eos_idx, max_len=max_len)
        for seq in idx_seqs:
            # strip special tokens by index (<sos>/<eos>/<pad>), then decode
            kept = [i for i in seq
                    if i not in (vocab.sos_idx, vocab.eos_idx, vocab.pad_idx)]
            preds.append(" ".join(vocab.decode(kept)))
    return preds


def normalize(text):
    # strip, lowercase, collapse whitespace - used for identical_ratio
    return " ".join(text.strip().lower().split())


def identical_ratio(sources, predictions):
    # sources: list of strings (or token-lists); predictions: list of strings
    src_strs = [" ".join(s) if isinstance(s, list) else s for s in sources]
    n = len(src_strs)
    if n == 0:
        return 0.0
    same = sum(1 for s, p in zip(src_strs, predictions)
               if normalize(s) == normalize(p))
    return same / n


# EASSE (feralvam/easse) is git-only and pulls a heavy, older dep tree, so we
# vendor its exact algorithm here: lowercase + sacrebleu 13a tokenization, the
# macro (JAVA) variant with F1 for deletion, n=4. Verified to match EASSE's
# corpus_sari() on the ASSET test set (34.96 / 20.73).


def _sari_multiply_counter(counter, value):
    return Counter({k: v * value for k, v in counter.items()})


def _sari_extract_ngrams(line, max_order=4):
    tokens = line.split()
    return [
        Counter(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        for n in range(1, max_order + 1)
    ]


def _sari_normalize(sents):
    # mirrors EASSE utils.preprocessing.normalize(lowercase=True, tokenizer="13a")
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
    tok = Tokenizer13a()
    return [tok(s.lower()) for s in sents]


def _sari_ngram_stats(orig_sents, sys_sents, refs_sents):
    # refs_sents: shape (n_references, n_samples), as EASSE expects
    n_order = 4
    (add_correct, add_total, add_ref_total,
     keep_correct, keep_total, keep_ref_total,
     del_correct, del_total, del_ref_total) = ([0] * n_order for _ in range(9))

    for orig_sent, sys_sent, *ref_sents in zip(orig_sents, sys_sents, *refs_sents):
        orig_ngrams = _sari_extract_ngrams(orig_sent)
        sys_ngrams = _sari_extract_ngrams(sys_sent)

        refs_ngrams = [Counter() for _ in range(n_order)]
        for ref_sent in ref_sents:
            for n, counter in enumerate(_sari_extract_ngrams(ref_sent)):
                refs_ngrams[n] += counter

        num_refs = len(ref_sents)
        for n in range(n_order):
            # ADD (binary)
            sys_and_not_orig = set(sys_ngrams[n]) - set(orig_ngrams[n])
            ref_and_not_orig = set(refs_ngrams[n]) - set(orig_ngrams[n])
            add_total[n] += len(sys_and_not_orig)
            add_ref_total[n] += len(ref_and_not_orig)
            add_correct[n] += len(sys_and_not_orig & set(refs_ngrams[n]))

            # KEEP (weighted)
            orig_and_sys = _sari_multiply_counter(orig_ngrams[n], num_refs) \
                & _sari_multiply_counter(sys_ngrams[n], num_refs)
            orig_and_ref = _sari_multiply_counter(orig_ngrams[n], num_refs) & refs_ngrams[n]
            keep_total[n] += sum(orig_and_sys.values())
            keep_ref_total[n] += sum(orig_and_ref.values())
            keep_correct[n] += sum((orig_and_sys & orig_and_ref).values())

            # DELETE (weighted)
            orig_and_not_sys = _sari_multiply_counter(orig_ngrams[n], num_refs) \
                - _sari_multiply_counter(sys_ngrams[n], num_refs)
            orig_and_not_ref = _sari_multiply_counter(orig_ngrams[n], num_refs) - refs_ngrams[n]
            del_total[n] += sum(orig_and_not_sys.values())
            del_ref_total[n] += sum(orig_and_not_ref.values())
            del_correct[n] += sum((orig_and_not_sys & orig_and_not_ref).values())

    return (add_correct, add_total, add_ref_total,
            keep_correct, keep_total, keep_ref_total,
            del_correct, del_total, del_ref_total)


def _sari_precision_recall_f1(sys_correct, sys_total, ref_total):
    precision = sys_correct / sys_total if sys_total > 0 else 0.0
    recall = sys_correct / ref_total if ref_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision > 0 and recall > 0 else 0.0
    return precision, recall, f1


def _sari_macro(stats):
    # EASSE compute_macro_sari(use_f1_for_deletion=True)
    (add_correct, add_total, add_ref_total,
     keep_correct, keep_total, keep_ref_total,
     del_correct, del_total, del_ref_total) = stats
    n_order = 4
    add = keep = dele = 0.0
    for n in range(n_order):
        _, _, add_f1 = _sari_precision_recall_f1(add_correct[n], add_total[n], add_ref_total[n])
        _, _, keep_f1 = _sari_precision_recall_f1(keep_correct[n], keep_total[n], keep_ref_total[n])
        _, _, del_f1 = _sari_precision_recall_f1(del_correct[n], del_total[n], del_ref_total[n])
        add += add_f1 / n_order
        keep += keep_f1 / n_order
        dele += del_f1 / n_order
    return add, keep, dele


def _sari_fallback(orig_sents, sys_sents, refs_sents):
    # refs_sents: shape (n_references, n_samples) - same convention as EASSE's
    # corpus_sari(), so the port is a drop-in replacement for it.
    orig_sents = _sari_normalize(orig_sents)
    sys_sents = _sari_normalize(sys_sents)
    refs_sents = [_sari_normalize(r) for r in refs_sents]
    stats = _sari_ngram_stats(orig_sents, sys_sents, refs_sents)
    add, keep, dele = _sari_macro(stats)
    return 100.0 * (add + keep + dele) / 3


def compute_sari(orig_sents, sys_sents, refs_sents):
    # refs_sents: list of lists (one inner list of 10 refs per source).
    # Both EASSE and the vendored port take refs transposed to
    # (n_references, n_samples), so transpose once here.
    refs_t = [list(r) for r in zip(*refs_sents)]
    try:
        from easse.sari import corpus_sari
        return corpus_sari(orig_sents, sys_sents, refs_t)
    except Exception as e:
        print(f"  easse SARI failed ({e}); falling back to vendored EASSE port")
        return _sari_fallback(orig_sents, sys_sents, refs_t)


def compute_bleu(sys_sents, refs_sents):
    from sacrebleu.metrics import BLEU
    return BLEU().corpus_score(sys_sents, refs_sents).score


def score_system(sources, predictions, references):
    # sources: list of strings; predictions: list of strings;
    # references: list of lists of strings (10 per source).
    return {
        "sari": compute_sari(sources, predictions, references),
        "bleu": compute_bleu(predictions, references),
        "identical_ratio": identical_ratio(sources, predictions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the LSTM seq2seq simplifier on ASSET test.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--llm-output", default=None,
                        help="path to LLM simplifications, one per line, "
                             "aligned with test_asset.src")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N test sentences "
                             "(sanity checks)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # sources + 10 references, aligned by line
    sources = read_file(os.path.join(args.data_dir, "test_asset.src"))
    refs = [read_file(os.path.join(args.data_dir, f"test_asset.ref.{i}"))
            for i in range(10)]
    references = [list(r) for r in zip(*refs)]  # per-sentence list of 10 refs
    if args.limit is not None:
        sources = sources[:args.limit]
        references = references[:args.limit]
    src_strs = [" ".join(s) for s in sources]
    ref_strs = [[" ".join(r) for r in ref_list] for ref_list in references]
    print(f"test sentences: {len(src_strs)}")

    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"checkpoint: epoch {ckpt['epoch']}, "
          f"best_val_loss {ckpt['best_val_loss']:.4f}, vocab {len(vocab)}")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "lstm_test_asset.txt")
    predictions = generate_predictions(model, vocab, sources, device,
                                       args.batch_size, args.max_len)
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions) + "\n")
    print(f"wrote {len(predictions)} predictions to {pred_path}")

    # score every system through the same metric functions
    systems = [("LSTM", score_system(src_strs, predictions, ref_strs))]

    # copy baseline: prediction = source verbatim
    systems.append(("copy baseline", score_system(src_strs, src_strs, ref_strs)))

    if args.llm_output:
        with open(args.llm_output, encoding="utf-8") as f:
            llm_preds = [line.rstrip("\n") for line in f]
        if len(llm_preds) != len(src_strs):
            raise SystemExit(
                f"llm output has {len(llm_preds)} lines, "
                f"expected {len(src_strs)} (aligned with test_asset.src)")
        systems.append(("LLM", score_system(src_strs, llm_preds, ref_strs)))

    # results table
    header = (f"checkpoint: {args.checkpoint}\n"
              f"epoch: {ckpt['epoch']}   "
              f"best_val_loss: {ckpt['best_val_loss']:.4f}\n"
              f"{'system':<14} {'SARI':>8} {'BLEU':>8} {'identical_ratio':>16}")
    rows = [f"{name:<14} {s['sari']:>8.2f} {s['bleu']:>8.2f} "
            f"{s['identical_ratio']:>16.4f}"
            for name, s in systems]
    table = "\n".join([header] + rows)
    print("\n" + table)

    with open(os.path.join(args.output_dir, "results.txt"),
              "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nresults written to {os.path.join(args.output_dir, 'results.txt')}")


if __name__ == "__main__":
    main()
