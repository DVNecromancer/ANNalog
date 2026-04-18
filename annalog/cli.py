#!/usr/bin/env python3
"""annalog/cli.py - ANNalog command-line SMILES generation (inference)

This is the package-installed CLI version of generation.py.

Defaults (no args needed for ckpt/vocab):
  Uses packaged resources:
    annalog/ckpt_and_vocab/Lev_extended.pt
    annalog/ckpt_and_vocab/stereo_experiment_vocab_ttf.pkl

Input (choose exactly one):
  -i/--input "CC(Cl)Br"        (single SMILES)  OR
  -i/--input inputs.smi        (file; one SMILES per line)

Input handling:
  User-provided SMILES are first checked against the ANNalog input vocabulary.
  If the original input is already recognized by ANNalog, it is kept unchanged.
  If it is not recognized, RDKit processing is attempted and the processed
  version is used only if ANNalog recognizes that processed form.

Output:
  Default TSV to stdout
  Optional CSV via -f/--format csv
  chembl-gen-check columns included by default (disable with --no-check)

Generation methods:
  beam:
    Classical beam search. Keeps the top-k partial sequences at each decoding step.

  BF-beam:
    Best-first beam search. Expands the current best partial sequence while keeping
    unexplored partial sequences in memory; usually slower and more memory-hungry.

  sampling:
    Samples each next token from the model probability distribution.

Method options:
  --temperature:
    Sampling temperature. Higher values increase diversity; lower values make
    sampling more conservative. Used only with sampling.

  --seed:
    Random seed for reproducible sampling. Used only with sampling.

  --prefix:
    Fixed starting prefix for generation. Can be either:
      - an integer number of starting characters taken from the current input SMILES, or
      - a literal starting string that must match the beginning of the current input SMILES.

  --keep-invalid:
    Keep invalid generated SMILES instead of filtering them out.

  --max-length:
    Maximum generated sequence length.

  --check / --no-check:
    Enable or disable chembl-gen-check output columns. Checking is on by default.

Exploration methods:
  normal:
    Generate directly from the input SMILES.

  variants:
    First create SMILES variants of the input, then generate from each variant.

  recursive:
    Repeatedly generate from the previous round's outputs.

Exploration options:
  --variant-number:
    Number of variants to create in variants mode.

  --loops:
    Number of recursive rounds in recursive mode.
"""

import argparse
import csv
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional, Union

import torch
from rdkit import Chem

from .model_handler import SMILESModelHandler
from .SMILES_generator import SMILESGenerator


CHECK_HEADER = [
    "check_scaffold",
    "check_skeleton",
    "check_ring_systems",
    "check_structural_alerts",
    "check_lacan",
]


def _normalize_method(user_method: str) -> str:
    """Internal decoder supports: 'beam', 'BF-beam', 'sampling'."""
    m = (user_method or "").strip()
    ml = m.lower()

    aliases = {
        "beam": "beam",
        "c-beam": "beam",
        "bf-beam": "BF-beam",
        "best-first": "BF-beam",
        "sampling": "sampling",
        "sample": "sampling",
    }
    if m == "BF-beam":
        return "BF-beam"
    if ml in aliases:
        return aliases[ml]
    raise ValueError(
        "Invalid --method. Use: beam, BF-beam, sampling (aliases: C-beam, sample)."
    )


def _parse_prefix(prefix: str) -> Union[int, str]:
    """Prefix is optional.

    - digits -> int number of starting characters
    - otherwise -> literal starting string (must match current input SMILES start)
    """
    if prefix is None:
        return 0
    p = str(prefix).strip()
    if p == "" or p == "0":
        return 0
    if p.isdigit():
        return int(p)
    return p


def _read_inputs_cli(
    input_any: Optional[str],
    input_smiles: Optional[str],
    input_file: Optional[str],
) -> List[str]:
    """Resolve CLI input.

    Priority:
      1) -i/--input: if it points to an existing file -> read SMILES lines.
         otherwise -> treat as a single SMILES string.
      2) legacy flags: --input-smiles / --input-file.

    Notes:
      - Blank lines in files are ignored.
      - If -i ends with a typical SMILES-file extension (.smi/.smiles/.txt)
        but the file does NOT exist, we raise FileNotFoundError.
        (We do NOT use '/' or '\\' heuristics because SMILES can contain them.)
    """
    if input_any is not None:
        s = input_any.strip()
        if not s:
            raise ValueError("-i/--input is empty")

        p = Path(s)
        if p.exists() and p.is_file():
            inputs: List[str] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    smi = line.strip()
                    if smi:
                        inputs.append(smi)
            if not inputs:
                raise ValueError(f"No SMILES found in file: {p}")
            return inputs

        if p.suffix.lower() in {".smi", ".smiles", ".txt"}:
            raise FileNotFoundError(f"Input file not found: {s}")

        return [s]

    if input_smiles:
        s = input_smiles.strip()
        if not s:
            raise ValueError("--input-smiles is empty")
        return [s]

    if not input_file:
        raise ValueError("Provide either -i/--input, --input-smiles, or --input-file")

    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    inputs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                inputs.append(s)

    if not inputs:
        raise ValueError(f"No SMILES found in file: {input_file}")
    return inputs


def _smiles_is_annalog_ready(smiles: str, src_field) -> bool:
    """Return True if every token in the input SMILES is present in ANNalog source vocab."""
    if src_field is None or src_field.vocab is None:
        raise RuntimeError("ANNalog source field is not initialized.")

    try:
        tokens = src_field.preprocess(smiles)
    except Exception:
        return False

    return all(tok in src_field.vocab.stoi for tok in tokens)


def _looks_kekulized_input(smiles: str, mol: Chem.Mol) -> bool:
    """Heuristic: aromatic molecule written without aromatic lowercase atom tokens."""
    has_aromatic_atoms = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    has_lowercase_aromatic_tokens = any(ch in {"b", "c", "n", "o", "p", "s"} for ch in smiles)
    return has_aromatic_atoms and not has_lowercase_aromatic_tokens


def _prepare_input_smiles(smiles: str, src_field) -> str:
    """Validate input, preserve ANNalog-ready strings, otherwise fall back to RDKit-processed SMILES."""
    s = (smiles or "").strip()
    if not s:
        raise ValueError("Input SMILES is empty")

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"Invalid input SMILES: {s!r}")

    if _smiles_is_annalog_ready(s, src_field):
        return s

    processed = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        kekuleSmiles=False,
    )

    if not _smiles_is_annalog_ready(processed, src_field):
        raise ValueError(
            f"Input SMILES {s!r} is valid, but neither the original input nor the RDKit processed version "
            f"{processed!r} is recognized by ANNalog."
        )

    if _looks_kekulized_input(s, mol):
        print(
            f"[INFO] kekulized input SMILES detected, using RDKit processed version : {processed} as input",
            file=sys.stderr,
        )
    else:
        print(
            f"[INFO] input SMILES not recognized by ANNalog, using RDKit processed version : {processed} as input",
            file=sys.stderr,
        )

    return processed


def _validate_prefix_after_processing(
    prefix: Union[int, str],
    raw_input: str,
    processed_input: str,
) -> None:
    """Guard against literal prefixes becoming invalid after RDKit processing."""
    if raw_input == processed_input:
        return

    if isinstance(prefix, str):
        if not processed_input.startswith(prefix):
            raise ValueError(
                f"Input SMILES {raw_input!r} was converted to {processed_input!r} before generation, "
                f"so the literal prefix {prefix!r} no longer matches the processed input. "
                "Please update the prefix or provide an ANNalog-ready input SMILES."
            )


def _run_check(checker, smiles: str) -> List[object]:
    """Run chembl-gen-check on one SMILES and return output-column values."""
    try:
        checker.load_smiles(smiles)
        return [
            checker.check_scaffold(),
            checker.check_skeleton(),
            checker.check_ring_systems(),
            checker.check_structural_alerts(),
            checker.check_lacan(),
        ]
    except Exception as e:
        print(
            f"[WARN] chembl-gen-check failed for generated SMILES {smiles!r}: {e}",
            file=sys.stderr,
        )
        return ["", "", "", "", ""]


def _write_generated_rows(
    writer: csv.writer,
    root_input: str,
    start_rank: int,
    generated,
    check_runner=None,
) -> int:
    """Write generated rows and return the next rank."""
    out_rank = start_rank
    for gen_smi, score in generated:
        row = [root_input, out_rank, gen_smi, score]
        if check_runner is not None:
            row.extend(_run_check(check_runner, gen_smi))
        writer.writerow(row)
        out_rank += 1
    return out_rank


def main(argv: Optional[List[str]] = None) -> int:
    # Packaged defaults (inside installed annalog)
    default_resources_dir = files("annalog") / "ckpt_and_vocab"
    default_checkpoint_res = default_resources_dir / "Lev_extended.pt"
    default_vocab_res = default_resources_dir / "stereo_experiment_vocab_ttf.pkl"

    parser = argparse.ArgumentParser(
        description="Generate SMILES with ANNalog (CLI). Outputs TSV by default.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.set_defaults(check=True)

    # Inputs (choose one)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "-i",
        "--input",
        dest="input_any",
        type=str,
        help="Input SMILES OR path to a .smi file (one SMILES per line).",
    )
    inp.add_argument(
        "--input-smiles",
        dest="input_smiles",
        type=str,
        help="(Legacy) Single SMILES.",
    )
    inp.add_argument(
        "--input-file",
        dest="input_file",
        type=str,
        help="(Legacy) .smi file path.",
    )

    # Resources (allow override)
    parser.add_argument(
        "--resources-dir",
        type=str,
        default=None,
        help="Folder containing checkpoint + vocab. If omitted, uses packaged ckpt_and_vocab.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help="Model checkpoint (.pt). If omitted, uses packaged Lev_extended.pt.",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Vocab pickle. If omitted, uses packaged stereo_experiment_vocab_ttf.pkl.",
    )

    # Generation options
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="beam",
        help="Decoding method: beam, BF-beam, or sampling. Default: beam",
    )
    parser.add_argument(
        "-n",
        "--generation-number",
        "--n",
        dest="generation_number",
        type=int,
        required=True,
        help="Number to generate (beam width or sample count). REQUIRED.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature. Higher = more diverse, lower = more conservative. Used only with --method sampling. Default: 1.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Used only with --method sampling. Default: 42",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="0",
        help="Fixed starting prefix: integer number of starting characters from the current input SMILES, or a literal starting string that matches the current input SMILES start. Default: 0",
    )
    parser.add_argument(
        "-k",
        "--keep-invalid",
        action="store_true",
        help="Keep invalid generated SMILES instead of filtering them out. Default: filtering is ON.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=102,
        help="Maximum generated sequence length. Default: 102",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Run chembl-gen-check on each generated SMILES and append scaffold, skeleton, ring-system, structural-alert, and LACAN outputs. Default: on",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Disable chembl-gen-check output columns.",
    )

    # Exploration options
    parser.add_argument(
        "-e",
        "--exploration-method",
        dest="exploration_method",
        choices=["normal", "variants", "recursive"],
        default="normal",
        help="Exploration method: normal, variants, recursive. Default: normal",
    )
    parser.add_argument(
        "--variant-number",
        type=int,
        default=10,
        help="Number of variants to create in variants mode. Default: 10",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of recursive rounds in recursive mode. Default: 1",
    )

    # Output
    parser.add_argument(
        "-f",
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output format. Default: tsv",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="-",
        help="Output path. Default: '-' (stdout)",
    )

    # Device
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device. Default: auto",
    )

    args = parser.parse_args(argv)

    if args.variant_number < 1:
        raise ValueError("--variant-number must be >= 1")
    if args.loops < 1:
        raise ValueError("--loops must be >= 1")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    method = _normalize_method(args.method)
    prefix = _parse_prefix(args.prefix)
    inputs = _read_inputs_cli(args.input_any, args.input_smiles, args.input_file)

    with ExitStack() as stack:
        if args.resources_dir is not None:
            res_dir = Path(args.resources_dir).expanduser().resolve()
            default_ckpt_path = res_dir / "Lev_extended.pt"
            default_vocab_path = res_dir / "stereo_experiment_vocab_ttf.pkl"
            ckpt_path = Path(args.checkpoint).expanduser() if args.checkpoint else default_ckpt_path
            vocab_path = Path(args.vocab).expanduser() if args.vocab else default_vocab_path
        else:
            ckpt_path = stack.enter_context(as_file(default_checkpoint_res))
            vocab_path = stack.enter_context(as_file(default_vocab_res))

            if args.checkpoint:
                ckpt_path = Path(args.checkpoint).expanduser()
            if args.vocab:
                vocab_path = Path(args.vocab).expanduser()

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocab not found: {vocab_path}")

        handler = SMILESModelHandler(
            src_vocab_path=str(vocab_path),
            trg_vocab_path=str(vocab_path),
            model_path=str(ckpt_path),
            device=device,
            max_length=args.max_length,
        )
        generator = SMILESGenerator(handler)

        check_runner = None
        if args.check:
            try:
                from chembl_gen_check import Checker
            except ImportError as e:
                raise ImportError(
                    "`--check` requires the installed package `chembl-gen-check`."
                ) from e
            check_runner = Checker("chembl")

        out_fh = sys.stdout if args.out == "-" else open(
            args.out, "w", encoding="utf-8", newline=""
        )
        try:
            delimiter = "\t" if args.format == "tsv" else ","
            writer = csv.writer(out_fh, delimiter=delimiter)

            header = ["input_smiles", "rank", "generated_smiles", "score"]
            if args.check:
                header.extend(CHECK_HEADER)
            writer.writerow(header)

            for raw_in_smi in inputs:
                in_smi = _prepare_input_smiles(raw_in_smi, handler.SRC)
                _validate_prefix_after_processing(prefix, raw_in_smi, in_smi)

                exploration = args.exploration_method
                out_rank = 1

                if exploration == "normal":
                    generated = generator.generate_smiles(
                        input_smiles=in_smi,
                        generation_number=args.generation_number,
                        generation_method=method,
                        temperature=args.temperature,
                        prefix=prefix,
                        filter_invalid=(not args.keep_invalid),
                        seed=args.seed,
                    )
                    out_rank = _write_generated_rows(
                        writer=writer,
                        root_input=raw_in_smi,
                        start_rank=out_rank,
                        generated=generated,
                        check_runner=check_runner,
                    )

                elif exploration == "variants":
                    try:
                        variants = generator.generate_variants(in_smi, args.variant_number)
                    except Exception as e:
                        print(
                            f"[WARN] generate_variants failed for input {raw_in_smi!r}: {e}",
                            file=sys.stderr,
                        )
                        variants = []

                    for variant in variants:
                        try:
                            generated = generator.generate_smiles(
                                input_smiles=variant,
                                generation_number=args.generation_number,
                                generation_method=method,
                                temperature=args.temperature,
                                prefix=prefix,
                                filter_invalid=(not args.keep_invalid),
                                seed=args.seed,
                            )
                        except Exception as e:
                            print(
                                f"[WARN] generate_smiles failed for variant {variant!r} (root {raw_in_smi!r}): {e}",
                                file=sys.stderr,
                            )
                            continue

                        out_rank = _write_generated_rows(
                            writer=writer,
                            root_input=raw_in_smi,
                            start_rank=out_rank,
                            generated=generated,
                            check_runner=check_runner,
                        )

                elif exploration == "recursive":
                    current_smiles: List[str] = [in_smi]

                    for loop_idx in range(args.loops):
                        next_smiles: List[str] = []

                        for parent in current_smiles:
                            try:
                                generated = list(
                                    generator.generate_smiles(
                                        input_smiles=parent,
                                        generation_number=args.generation_number,
                                        generation_method=method,
                                        temperature=args.temperature,
                                        prefix=prefix,
                                        filter_invalid=(not args.keep_invalid),
                                        seed=args.seed,
                                    )
                                )
                            except Exception as e:
                                print(
                                    f"[WARN] generate_smiles failed in loop {loop_idx + 1} for parent {parent!r} (root {raw_in_smi!r}): {e}",
                                    file=sys.stderr,
                                )
                                continue

                            out_rank = _write_generated_rows(
                                writer=writer,
                                root_input=raw_in_smi,
                                start_rank=out_rank,
                                generated=generated,
                                check_runner=check_runner,
                            )
                            next_smiles.extend(gen_smi for gen_smi, _score in generated)

                        current_smiles = next_smiles

                else:
                    raise ValueError(
                        "Invalid exploration method. Choose from: normal, variants, recursive."
                    )

        finally:
            if out_fh is not sys.stdout:
                out_fh.close()

    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
