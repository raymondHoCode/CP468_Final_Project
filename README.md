# CP468 Final Project — LSTM Sequence-to-Sequence Text Simplification

An LSTM sequence-to-sequence model with Bahdanau (additive) attention that simplifies
English sentences, trained and evaluated on standard text-simplification corpora.

## What this project is

- **Task:** text simplification — rewrite a complex sentence into a simpler one that
  preserves meaning (e.g. *"The committee, chaired by Dr. Smith, will convene tomorrow"*
  → *"The committee will meet tomorrow. Dr. Smith leads it."*).
- **Model:** a bidirectional-LSTM encoder + unidirectional-LSTM decoder with Bahdanau
  attention (`model.py`). The encoder packs padded sequences so the recurrence only
  processes real tokens, bridges its bidirectional final states down to the decoder's
  initial state, and attention is masked so padding never receives weight. Training uses
  teacher forcing with a configurable ratio; decoding at evaluation is greedy.
- **Data:** WikiAuto (ACL 2020) for training/validation and ASSET for testing, pulled and
  cleaned by `data_script.py` (unescapes PTB-style tokens, drops identical / too-short /
  too-long / target-longer pairs). The ASSET test set has 10 reference simplifications per
  sentence.
- **Evaluation:** `evaluate_model.py` scores the trained model on the ASSET test set with
  **SARI** (primary), **BLEU** (secondary), and **identical_ratio** (a copy-trap: the
  fraction of outputs that are verbatim copies of the input). It also scores a **copy
  baseline** (output = input) through the same pipeline so the LSTM is judged against an
  upper bound on BLEU / and reference point on SARI.

## Setup

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **macOS / Linux:** `source venv/bin/activate`

Then install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.13.0`, `numpy==2.5.2`, and `sacrebleu==2.6.0`.
`evaluate_model.py` prefers **EASSE** for SARI and falls back to a vendored
port of EASSE's corpus SARI if EASSE isn't importable; `sacrebleu`
provides BLEU (sacrebleu does NOT implement SARI).

EASSE is the SARI implementation that produced the recorded numbers below, but it is
**git-only** (not on PyPI) and pulls in a heavy, older dependency tree, so it is
intentionally kept out of `requirements.txt`. To use it (optional):

```bash
pip install git+https://github.com/feralvam/easse.git
```

If EASSE isn't installed, evaluation still runs and reports the same SARI via the
vendored port in `evaluate_model.py` (see the note there).

## Reproducing the results

The full reproduce path is four steps, in order. Steps 1–2 prepare the data, step 3 trains,
step 4 evaluates.

### Step 1 — pull the data

Downloads WikiAuto (train) and ASSET (valid/test) into `data/`, cleaning and filtering them:

```bash
python data_script.py
```

This writes `data/train.src`, `data/train.dst` (~469k pairs), `data/valid.src` +
`data/valid.ref.{0..9}`, and `data/test_asset.src` + `data/test_asset.ref.{0..9}`.

### Step 2 — make the 150k training subset

The golden training command below uses `train.small.src` / `train.small.dst`, which are the
first 150,000 pairs of the full training data (a cap to keep training time reasonable on a
single GPU). Create them with the first 150k lines of the full files:

**Windows (PowerShell):**

```powershell
Get-Content data\train.src -TotalCount 150000 | Set-Content data\train.small.src
Get-Content data\train.dst -TotalCount 150000 | Set-Content data\train.small.dst
```

**macOS / Linux equivalent:**

```bash
head -n 150000 data/train.src > data/train.small.src
head -n 150000 data/train.dst > data/train.small.dst
```

> Skipping this step will make the training command below fail with a "file not found" on
> `train.small.src`.

### Step 3 — train

```bash
python train.py --epochs 10 --min-freq 4 --emb-dim 128 --enc-hidden 256 --dec-hidden 256 --batch-size 32 --teacher-forcing-ratio 0.8 --train-src train.small.src --train-dst train.small.dst
```

What the flags do and why they're set this way:

| Flag | Value | Why |
| --- | --- | --- |
| `--epochs` | `10` | Maximum training epochs. Early stopping (`--patience 3`, default) can stop sooner if validation loss stops improving. |
| `--min-freq` | `4` | Tokens appearing fewer than 4 times across the training src+dst are mapped to `<unk>`. Keeps the vocabulary at a manageable 63,819 instead of exploding on rare words. |
| `--emb-dim` | `128` | Token embedding size. Smaller than the 256 default to keep the model lean for the 150k subset. |
| `--enc-hidden` | `256` | Encoder LSTM hidden size (bidirectional, so each step emits 512). |
| `--dec-hidden` | `256` | Decoder LSTM hidden size. |
| `--batch-size` | `32` | Examples per batch; fits comfortably on an RTX 3060. |
| `--teacher-forcing-ratio` | `0.8` | 80% of decoder steps are fed the true next token, 20% feed the model's own prediction. The self-fed fraction acts as a mild regularizer / exposure-bias mitigation so the model learns to recover from its own errors. |
| `--train-src` / `--train-dst` | `train.small.src` / `train.small.dst` | Point training at the 150k subset from Step 2 instead of the default full `train.src` / `train.dst`. |

The best model (lowest validation loss) is saved to `checkpoints/best_model.pt` together
with the vocab and hyperparameters, and a run summary is written to
`checkpoints/training_log.txt`.

### Step 4 — evaluate

```bash
python evaluate_model.py
```

Loads `checkpoints/best_model.pt`, generates greedy simplifications for the ASSET test set
(written to `outputs/lstm_test_asset.txt`), and scores the LSTM plus a copy baseline. The
results table is printed and saved to `outputs/results.txt`.

## Key results

Evaluated on the ASSET test set (359 sentences, 10 references each):

| System | SARI | BLEU | identical_ratio |
| --- | ---: | ---: | ---: |
| **LSTM** | **34.96** | 5.59 | 0.0000 |
| copy baseline | 20.73 | **52.33** | 1.0000 |

The LSTM more than doubles the copy baseline's SARI (34.96 vs 20.73) and never simply
echoes its input (`identical_ratio` 0.00), at the expected cost of BLEU — it actually
rewrites the sentence rather than copying it. The copy baseline's `identical_ratio` of 1.00
is exactly the copy-trap it exists to catch.

**Checkpoint details** (`checkpoints/best_model.pt`):

- Best epoch: **8** (of 10)
- Validation loss: **2.9265**
- Vocabulary size: **63,819**
- Trainable parameters: **~34.9M** (34,909,260)
- Hardware: **NVIDIA GeForce RTX 3060**

## Project layout

```
data_script.py       # download + clean WikiAuto/ASSET -> data/
data.py              # line-aligned src/dst dataset -> padded tensor batches
vocab.py             # Vocab: <pad>/<unk>/<sos>/<eos> + min-freq filtering
model.py             # Encoder, Bahdanau Attention, Decoder, Seq2Seq
train.py             # training loop, early stopping, checkpointing
evaluate_model.py    # ASSET eval: SARI / BLEU / identical_ratio + copy baseline
```
