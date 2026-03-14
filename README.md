# ANNalog

ANNalog is a **SMILES-to-SMILES** generative model for medicinal chemistry analogue design.

- **Paper (ChemRxiv):** https://chemrxiv.org/doi/10.26434/chemrxiv-2025-9c1v6

ANNalog is a transformer-based sequence-to-sequence (Seq2Seq) model designed to generate medicinal-chemistry-relevant analogues of an input molecule. It supports:
- **local chemical-space exploration** (small, SAR-like modifications), and
- **scaffold hopping** (changing the core scaffold while remaining chemically relevant).

---

## Dependencies / environment (recommended)

A tested dependency set is provided in **`seq2seq_environment.yml`** in this repo (recommended for reproducibility).

Notes:
- The PyPI package **does not pin or install PyTorch** for you. Please install a PyTorch build that matches your system (CPU/CUDA).
- The provided conda YAML is the recommended environment for ANNalog generation and includes the required `chembl-gen-check` dependency.
- `chembl-gen-check` requires Python `>=3.10`; the provided environment targets Python 3.12.

---

## Google Colab page

A place where you could try to generate some molecules online:
https://colab.research.google.com/drive/1aJhaBOG7xuYFwMGzfUmbMsLe8T462Ptc#scrollTo=Ss1QOzXjzKSP

---

## Installation

### Option A — Install from PyPI (recommended for “just use it”)

```bash
pip install annalog
```

After installation, you can use the installed CLI:

```bash
annalog-generate -h
```

### Option B — Install from GitHub (recommended for development / editing code)

```bash
git clone https://github.com/DVNecromancer/ANNalog.git
cd ANNalog
```

**Conda (recommended):**
```bash
conda env create -f seq2seq_environment.yml
conda activate <env_name_from_yml>
pip install -e .
```

---

## Generating molecules

You have **two ways** to generate:

1) **Installed CLI** (works after `pip install annalog`): `annalog-generate ...`  
2) **Repo script** `generation.py` (works from a cloned repo; easy to modify)

Both share the same core options:
- decoding methods: `beam`, `BF-beam`, `sampling`
- exploration modes: `normal`, `variants`, `recursive`
- optional post-generation annotation with `--cgc`
- TSV/CSV output

---

### Decoding methods (what they mean)

#### `-m beam`
Classical beam search. Keeps the top-k partial sequences at each decoding step.

#### `-m BF-beam`
Best-first beam search. Expands the current best partial sequence while keeping unexplored partial sequences in memory. This is usually slower and more memory-hungry than classical beam search.

#### `-m sampling`
Samples each next token from the model probability distribution.

---

### Exploration methods (what they mean)

#### `-e normal` (default)
Generate directly from the input SMILES.

#### `-e variants`
1) Create `--variant-number` *SMILES variants of the same molecule* by **randomizing atom order** and writing **non-canonical SMILES** (i.e., different syntactic representations of the same structure).
2) Run generation from **each** variant and pool all results.

#### `-e recursive`
Run generation in multiple rounds. In round 1 you generate from the input SMILES.  
In round 2, you generate again using **the round-1 outputs as new inputs**, and so on for `--loops` rounds.

---

### Main generation options

- `--temperature`: sampling temperature. Higher values increase diversity; lower values make sampling more conservative. Used only with `-m sampling`.
- `--seed`: random seed for reproducible sampling. Used only with `-m sampling`.
- `--prefix`: fixed starting prefix for generation. This can be either:
  - an integer number of starting characters taken from the beginning of the input SMILES, or
  - a literal starting string that must match the beginning of the input SMILES.
- `--keep-invalid`: keep invalid generated SMILES instead of filtering them out.
- `--max-length`: maximum generated sequence length.
- `--variant-number`: number of variants to create in `-e variants` mode.
- `--loops`: number of recursive rounds in `-e recursive` mode.
- `--cgc`: run `chembl-gen-check` on each generated SMILES **after generation** and append five extra output columns:
  - `cgc_scaffold`
  - `cgc_skeleton`
  - `cgc_ring_systems`
  - `cgc_structural_alerts`
  - `cgc_lacan`

`chembl-gen-check` is a lightweight structural sanity-check package for rapid verification of scaffold, generic scaffold (here reported as `skeleton`), and ring-system precedent in ChEMBL, and it can also report structural alerts and LACAN-related uncommon-bond information.

**Reference:** [chembl-gen-check (PyPI)](https://pypi.org/project/chembl-gen-check/)

---

## A) Using the installed CLI (PyPI / installed package)

Help:
```bash
annalog-generate -h
```

**Quick start (single SMILES, beam, 50 outputs):**
```bash
annalog-generate -i "CCO" -n 50 -m beam -o gen.tsv
```

**Sampling (10 outputs):**
```bash
annalog-generate -i "CC(Cl)Br" -n 10 -m sampling --temperature 1.2 --seed 42 -o gen.tsv
```

**Variants exploration:**
```bash
annalog-generate -i "CCO" -n 20 -e variants --variant-number 10 -o gen_variants.tsv
```

**Recursive exploration (2 loops):**
```bash
annalog-generate -i "CCO" -n 10 -e recursive --loops 2 -o gen_recursive.tsv
```

**Generation with chembl-gen-check annotation:**
```bash
annalog-generate -i "CCO" -n 50 -m beam --cgc -o gen_with_cgc.tsv
```

You can also invoke the same CLI via Python module form:
```bash
python -m annalog.cli -i "CCO" -n 50 -o gen.tsv
```

**Resources (ckpt + vocab):**
- For the installed CLI, the checkpoint + vocab are shipped **inside the package** and used by default.
- You can still override them if needed using `--resources-dir` or `--checkpoint/--vocab`.

---

## B) Using the repo script (generation.py)

From the repo root (after `pip install -e .`), you can run:

```bash
python generation.py -h
```

**Note about resources in the repo:**  
In this repository the checkpoint/vocab live under:

`annalog/ckpt_and_vocab/`

So when running `generation.py`, point it explicitly:

```bash
python generation.py \
  -i "CCO" \
  -n 50 \
  -m beam \
  --resources-dir annalog/ckpt_and_vocab \
  -o gen.tsv
```

---

## Output format

The output file includes a header row with columns:

- `input_smiles`
- `rank` (1-based)
- `generated_smiles`
- `score`

When `--cgc` is enabled, five additional columns are appended:

- `cgc_scaffold`
- `cgc_skeleton`
- `cgc_ring_systems`
- `cgc_structural_alerts`
- `cgc_lacan`
