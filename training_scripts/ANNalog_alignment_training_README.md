# ANNalog Alignment and Scratch Training Scripts

This folder contains two scripts:

```text
alignment.py   # builds aligned and augmented ANNalog src/trg datasets
training.py    # trains an ANNalog seq2seq model from scratch from src/trg files
```

The usual workflow is:

```text
original train/val/test .src/.trg files
        |
        v
alignment.py
        |
        v
aligned + augmented train/val/test .src/.trg files
        |
        v
training.py
        |
        v
trained ANNalog model weights
```

---

## Expected ANNalog repository structure

The scripts expect an ANNalog Python package like this:

```text
ANNalog/
  annalog/
    __init__.py
    model_files/
      __init__.py
      vocabulary.py
      seq2seq_attention.py
    ckpt_and_vocab/
      stereo_experiment_vocab_ttf.pkl

  alignment.py
  training.py
```

The important files are:

```text
annalog/model_files/vocabulary.py
annalog/model_files/seq2seq_attention.py
```

`alignment.py` uses:

```python
from annalog.model_files.vocabulary import SMILESTokenizer
```

`training.py` uses:

```python
from annalog.model_files import vocabulary
from annalog.model_files import seq2seq_attention
```

When using `--repo-root`, pass the folder that contains the `annalog/` package.

Correct:

```bash
--repo-root /path/to/ANNalog
```

Incorrect:

```bash
--repo-root /path/to/ANNalog/annalog
```

If you run the scripts from the repository root, this is usually enough:

```bash
--repo-root .
```

---

# 1. `alignment.py`

## Purpose

`alignment.py` prepares ANNalog training data by generating randomized SMILES variants and aligning source/target pairs.

For each input pair:

```text
A, B
```

it creates both directions:

```text
A -> B
B -> A
```

With the default setting:

```bash
--aug-per-direction 4
```

each original pair nominally becomes 10 training rows:

```text
Forward direction:
  A  -> B
  A1 -> best aligned B1
  A2 -> best aligned B2
  A3 -> best aligned B3
  A4 -> best aligned B4

Reverse direction:
  B  -> A
  B1 -> best aligned A1
  B2 -> best aligned A2
  B3 -> best aligned A3
  B4 -> best aligned A4
```

So the nominal expansion is:

```text
2 * (1 + 4) = 10 rows per input pair
```

## How randomized variants are selected

For a molecule `A`, the script generates many randomized SMILES strings.

It then measures token-level Levenshtein distance between each randomized variant and the original SMILES `A`.

For:

```bash
--aug-per-direction 4
```

the selected query variants come from these positions in the internal distance distribution:

```text
0%, 25%, 50%, 75%
```

This gives a spread from similar to more dissimilar randomized SMILES.

For each selected `A'`, the script searches the randomized SMILES pool of the partner molecule `B` and picks the `B'` with the smallest token-level Levenshtein distance to `A'`.

The same process is repeated in the reverse direction.

## Input format

The input dataset folder should contain:

```text
input_dataset/
  train.src
  train.trg
  val.src
  val.trg
```

Optional test files are also supported:

```text
input_dataset/
  test.src
  test.trg
```

Each line in `.src` must correspond to the same line number in `.trg`.

Example:

```text
train.src line 1  <->  train.trg line 1
train.src line 2  <->  train.trg line 2
```

## Output format

The output dataset folder will contain:

```text
aligned_augmented_dataset/
  train.src
  train.trg
  val.src
  val.trg
  train.alignment.tsv
  val.alignment.tsv
  alignment_summary.json
```

If test files are present, it will also contain:

```text
aligned_augmented_dataset/
  test.src
  test.trg
  test.alignment.tsv
```

The `.src` and `.trg` files are the files used by `training.py`.

The `.alignment.tsv` files are reports showing how each output row was generated.

## Example usage

From the ANNalog repo root:

```bash
python alignment.py \
  --repo-root . \
  --input-dataset /path/to/original_dataset \
  --output-dataset /path/to/aligned_augmented_dataset \
  --randomized-per-molecule 1000 \
  --aug-per-direction 4 \
  --max-raw-tokens 100 \
  --num-workers 8
```

## Important arguments

### `--input-dataset`

Folder containing the original `.src` and `.trg` files.

### `--output-dataset`

Folder where aligned and augmented files will be written.

### `--repo-root`

Path to the folder containing the `annalog/` package.

### `--randomized-per-molecule`

Number of randomized SMILES strings to attempt per molecule.

Default:

```text
1000
```

Higher values may improve alignment diversity but increase runtime.

### `--aug-per-direction`

Number of randomized query variants to select per direction.

Default:

```text
4
```

With the default value, each input pair nominally expands to 10 rows:

```text
2 * (1 + 4) = 10
```

### `--max-raw-tokens`

Maximum tokenized SMILES length before adding `<sos>` and `<eos>`.

Default:

```text
100
```

This matches the original ANNalog preprocessing/alignment logic.

### `--num-workers`

Number of parallel worker processes.

Example:

```bash
--num-workers 8
```

Use `1` for debugging.

---

# 2. `training.py`

## Purpose

`training.py` trains an ANNalog sequence-to-sequence model from scratch using existing `.src` and `.trg` files.

This script does **not** fine-tune from a pretrained checkpoint.

It does **not** load:

```text
Lev_extended.pt
```

or any other pretrained model weights.

It initializes the model from the ANNalog model class definitions and trains from scratch.

The script still uses an existing vocabulary pickle, for example:

```text
annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl
```

So this is:

```text
model training from scratch
```

not:

```text
vocabulary training from scratch
```

## Input format

The dataset folder should contain:

```text
dataset/
  train.src
  train.trg
  val.src
  val.trg
```

Optional test files are supported:

```text
dataset/
  test.src
  test.trg
```

Usually this dataset is the output from `alignment.py`.

## Split usage

The splits are used as follows:

```text
train -> used for gradient updates
val   -> used to select the best checkpoint
test  -> used only once at the end for final reporting
```

Validation loss is **not** used for backpropagation.

Test loss is **not** used for training or model selection.

## Output format

The script writes results to:

```text
output_dir/
  training/
    best_model.pt
    final_model.pt
    best_model_training.ckpt
    final_model_training.ckpt
    dataset_stats.json
    run_config.json
    training_log.tsv
    summary.json
    summary.txt
  pipeline_summary.json
```

Main files:

```text
best_model.pt
```

Weights from the epoch with the best validation loss.

```text
final_model.pt
```

Weights from the final epoch.

```text
training_log.tsv
```

Per-epoch loss log.

```text
summary.txt
summary.json
```

Final training summary.

## Example usage

Train from the aligned dataset:

```bash
python training.py \
  --repo-root . \
  --dataset-dir /path/to/aligned_augmented_dataset \
  --vocab-path annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl \
  --output-dir /path/to/scratch_training_run \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-4 \
  --device cuda
```

For CPU:

```bash
python training.py \
  --repo-root . \
  --dataset-dir /path/to/aligned_augmented_dataset \
  --vocab-path annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl \
  --output-dir /path/to/scratch_training_run \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-4 \
  --device cpu
```

## Important arguments

### `--dataset-dir`

Folder containing:

```text
train.src
train.trg
val.src
val.trg
```

and optionally:

```text
test.src
test.trg
```

### `--vocab-path`

Path to the ANNalog vocabulary pickle.

Example:

```bash
--vocab-path annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl
```

### `--output-dir`

Folder where training outputs will be saved.

### `--repo-root`

Path to the folder containing the `annalog/` package.

If running from the repository root:

```bash
--repo-root .
```

### `--epochs`

Number of training epochs.

Example:

```bash
--epochs 200
```

### `--batch-size`

Training batch size.

Default:

```text
64
```

### `--lr`

Learning rate.

Default:

```text
1e-4
```

### `--device`

One of:

```text
auto
cpu
cuda
```

Example:

```bash
--device cuda
```

### `--max-seq-len`

Maximum sequence length after adding `<sos>` and `<eos>`.

Default:

```text
102
```

This corresponds to:

```text
100 raw SMILES tokens + <sos> + <eos> = 102
```

---

# Full workflow example

## Step 1: Prepare aligned and augmented dataset

```bash
python alignment.py \
  --repo-root . \
  --input-dataset /path/to/original_dataset \
  --output-dataset /path/to/aligned_augmented_dataset \
  --randomized-per-molecule 1000 \
  --aug-per-direction 4 \
  --max-raw-tokens 100 \
  --num-workers 8
```

## Step 2: Train from scratch

```bash
python training.py \
  --repo-root . \
  --dataset-dir /path/to/aligned_augmented_dataset \
  --vocab-path annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl \
  --output-dir /path/to/scratch_training_run \
  --epochs 200 \
  --batch-size 64 \
  --lr 1e-4 \
  --device cuda
```

## Step 3: Check outputs

```bash
cat /path/to/scratch_training_run/training/summary.txt
```

Check the loss log:

```bash
head /path/to/scratch_training_run/training/training_log.tsv
```

Check alignment summary:

```bash
cat /path/to/aligned_augmented_dataset/alignment_summary.json
```

---

# Dataset file checks

Before training, confirm that each split has matching source and target line counts.

```bash
wc -l /path/to/aligned_augmented_dataset/train.src
wc -l /path/to/aligned_augmented_dataset/train.trg

wc -l /path/to/aligned_augmented_dataset/val.src
wc -l /path/to/aligned_augmented_dataset/val.trg

wc -l /path/to/aligned_augmented_dataset/test.src
wc -l /path/to/aligned_augmented_dataset/test.trg
```

For each split, `.src` and `.trg` should have the same number of lines.

---

# Requirements

Core Python packages:

```text
torch
rdkit
```

Optional but recommended for faster edit distance:

```text
python-Levenshtein
```

or:

```text
rapidfuzz
```

If neither is installed, `alignment.py` falls back to a pure Python Levenshtein implementation, but it may be slower.

---

# GitHub usage notes

## Put the scripts and README into a folder on GitHub

A clean layout is:

```text
ANNalog/
  scripts/
    alignment.py
    training.py
    README.md
```

### Option A: Use the GitHub website

1. Open your repository on GitHub.
2. Click **Add file**.
3. Click **Upload files**.
4. Drag in `alignment.py`, `training.py`, and this `README.md`.
5. To place them in a folder, upload a folder from your computer, or create a file path like:

```text
scripts/README.md
```

GitHub treats the `/` as a folder separator when creating files.

### Option B: Use the command line

From your local repo:

```bash
mkdir -p scripts
cp /path/to/alignment.py scripts/alignment.py
cp /path/to/training.py scripts/training.py
cp /path/to/README.md scripts/README.md

git add scripts/alignment.py scripts/training.py scripts/README.md
git commit -m "Add ANNalog alignment and scratch training scripts"
git push origin main
```

Use your actual branch name instead of `main` if your repository uses another branch.

## Download only one folder from a GitHub repository

GitHub's web interface normally downloads the whole repository as a ZIP. To get only one folder, use Git sparse checkout.

Example: download only the `scripts/` folder from a repository.

```bash
mkdir annalog-scripts-only
cd annalog-scripts-only

git init
git remote add origin https://github.com/OWNER/REPO.git
git sparse-checkout init --cone
git sparse-checkout set scripts
git pull origin main
```

Replace:

```text
OWNER/REPO
```

with the GitHub repository owner and repository name.

Replace:

```text
scripts
```

with the folder path you want.

Replace:

```text
main
```

with the correct branch if needed.

---

# Notes

## This is not fine-tuning

`training.py` does not load pretrained ANNalog model weights.

There is no argument like:

```text
--checkpoint-path
```

and there is no model weight loading before training.

## The vocabulary is still required

The model is trained from scratch, but the script still needs a vocabulary pickle so that SMILES tokens can be converted into token IDs.

## Validation and test are different

Validation loss is used to choose the best checkpoint.

Test loss is only reported after training is complete and should not be used to tune hyperparameters.

## Alignment expansion may be less than nominal

With:

```bash
--aug-per-direction 4
```

the nominal expansion is 10 rows per input pair.

However, if RDKit cannot generate enough valid randomized SMILES variants for a molecule, some input rows may produce fewer than 10 output rows. The alignment summary reports full-count and partial-count rows.
