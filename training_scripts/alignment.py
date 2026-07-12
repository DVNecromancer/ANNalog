"""
Bidirectional distributed-alignment ANNalog dataset augmenter.

Input layout:

  input_dataset/
    train.src
    train.trg
    val.src
    val.trg
    test.src    optional
    test.trg    optional

Output layout:

  output_dataset/
    train.src
    train.trg
    val.src
    val.trg
    test.src                 if test exists
    test.trg                 if test exists
    train.alignment.tsv
    val.alignment.tsv
    test.alignment.tsv       if test exists
    alignment_summary.json

For each input pair A, B:

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

The A1..A4 variants are selected from A's randomized SMILES pool by
internal token-level Levenshtein distance from A.

For --aug-per-direction 4, the selected positions are:

  0.00, 0.25, 0.50, 0.75

across the sorted internal distance distribution.

The partner side is not selected by quantile. For each selected query variant,
the script searches the other molecule's randomized pool and picks the best
aligned partner by minimum token-level Levenshtein distance.

Nominal expansion with --aug-per-direction 4:

  2 * (1 + 4) = 10 rows per original pair
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


DEFAULT_MAX_RAW_TOKENS = 100
DEFAULT_RANDOMIZED_PER_MOLECULE = 1000
DEFAULT_AUG_PER_DIRECTION = 4

TOKENIZER = None
TOKEN_TO_CHAR = {}
MAX_RAW_TOKENS_GLOBAL = DEFAULT_MAX_RAW_TOKENS
N_RANDOMIZED_GLOBAL = DEFAULT_RANDOMIZED_PER_MOLECULE
AUG_PER_DIRECTION_GLOBAL = DEFAULT_AUG_PER_DIRECTION


try:
    import Levenshtein

    def edit_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)

except ImportError:
    try:
        from rapidfuzz.distance import Levenshtein as RFLevenshtein

        def edit_distance(a: str, b: str) -> int:
            return RFLevenshtein.distance(a, b)

    except ImportError:
        def edit_distance(a: str, b: str) -> int:
            if len(a) < len(b):
                a, b = b, a

            prev = list(range(len(b) + 1))

            for i, ca in enumerate(a, 1):
                curr = [i]

                for j, cb in enumerate(b, 1):
                    curr.append(
                        min(
                            prev[j] + 1,
                            curr[j - 1] + 1,
                            prev[j - 1] + (ca != cb),
                        )
                    )

                prev = curr

            return prev[-1]


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m}m {s}s"

    if m > 0:
        return f"{m}m {s}s"

    return f"{s}s"


def add_repo_root_to_syspath(repo_root: Optional[str]) -> Path:
    if repo_root is None:
        repo_root_path = Path(__file__).resolve().parent
    else:
        repo_root_path = Path(repo_root).resolve()

    if str(repo_root_path) not in sys.path:
        sys.path.insert(0, str(repo_root_path))

    return repo_root_path


def init_worker(
    repo_root: str,
    max_raw_tokens: int,
    randomized_per_molecule: int,
    aug_per_direction: int,
) -> None:
    global TOKENIZER
    global TOKEN_TO_CHAR
    global MAX_RAW_TOKENS_GLOBAL
    global N_RANDOMIZED_GLOBAL
    global AUG_PER_DIRECTION_GLOBAL

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from annalog.model_files.vocabulary import SMILESTokenizer

    TOKENIZER = SMILESTokenizer()
    TOKEN_TO_CHAR = {}
    MAX_RAW_TOKENS_GLOBAL = max_raw_tokens
    N_RANDOMIZED_GLOBAL = randomized_per_molecule
    AUG_PER_DIRECTION_GLOBAL = aug_per_direction


def token_len_ok(smi: str) -> bool:
    return len(TOKENIZER.tokenize(smi)) <= MAX_RAW_TOKENS_GLOBAL


def encode_smiles_tokenwise(smi: str) -> str:
    """
    Encode SMILES tokenizer tokens as private unicode chars.

    This makes Levenshtein distance token-level instead of raw character-level.
    For example, Cl and Br are treated as one token each.
    """
    chars = []

    for tok in TOKENIZER.tokenize(smi):
        if tok not in TOKEN_TO_CHAR:
            TOKEN_TO_CHAR[tok] = chr(0x100 + len(TOKEN_TO_CHAR))

        chars.append(TOKEN_TO_CHAR[tok])

    return "".join(chars)


def generate_randomized_smiles_pool(smi: str) -> List[str]:
    """
    Generate randomized SMILES for one molecule.

    The original SMILES is excluded from this pool.
    The original row is added separately.
    """
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return []

    pool = set()

    for _ in range(N_RANDOMIZED_GLOBAL):
        rsmi = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
            isomericSmiles=True,
        )

        if rsmi and rsmi != smi and token_len_ok(rsmi):
            pool.add(rsmi)

    return list(pool)


def select_distributed_query_variants(
    original_smi: str,
    generated_smiles: Sequence[str],
    n_select: int,
) -> List[Dict[str, object]]:
    """
    Select query-side variants across the internal distance distribution.

    The generated variants are scored by token-level Levenshtein distance
    to the original SMILES.

    Then we sort ascending by distance and choose positions:

      i / n_select for i in range(n_select)

    For n_select = 4, this gives:

      0.00, 0.25, 0.50, 0.75

    This intentionally does not use 1.00, matching the original behavior.
    """
    if n_select <= 0:
        return []

    original_enc = encode_smiles_tokenwise(original_smi)

    scored = []
    seen_smiles = set()

    for smi in generated_smiles:
        if smi == original_smi or smi in seen_smiles:
            continue

        seen_smiles.add(smi)

        dist = edit_distance(
            original_enc,
            encode_smiles_tokenwise(smi),
        )

        scored.append((dist, smi))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    n = len(scored)

    target_quantiles = [i / n_select for i in range(n_select)]

    selected = []
    used_indices = set()

    for q in target_quantiles:
        target_idx = min(n - 1, max(0, round(q * (n - 1))))

        best_idx = None
        best_delta = None

        for idx in range(n):
            if idx in used_indices:
                continue

            delta = abs(idx - target_idx)

            if best_idx is None or delta < best_delta:
                best_idx = idx
                best_delta = delta

        if best_idx is not None:
            dist, smi = scored[best_idx]

            selected.append(
                {
                    "smi": smi,
                    "internal_distance": dist,
                    "pool_rank_index": best_idx,
                    "pool_size": n,
                    "target_quantile": q,
                    "target_index": target_idx,
                }
            )

            used_indices.add(best_idx)

    return selected


def select_best_aligned_partner(
    query_smi: str,
    partner_pool: Sequence[str],
    fallback_smi: str,
) -> Tuple[str, int]:
    """
    Choose the partner SMILES with minimum token-level edit distance to query_smi.

    The original partner is used as fallback and is included as a valid candidate.
    """
    query_enc = encode_smiles_tokenwise(query_smi)

    best_smi = fallback_smi
    best_dist = edit_distance(
        query_enc,
        encode_smiles_tokenwise(fallback_smi),
    )

    seen = {fallback_smi}

    for smi in partner_pool:
        if smi in seen:
            continue

        seen.add(smi)

        dist = edit_distance(
            query_enc,
            encode_smiles_tokenwise(smi),
        )

        if dist < best_dist:
            best_dist = dist
            best_smi = smi

    return best_smi, best_dist


def make_original_row(
    line_number: int,
    direction: str,
    input_src: str,
    input_trg: str,
    query_original: str,
    partner_original: str,
) -> Dict[str, object]:
    pair_dist = edit_distance(
        encode_smiles_tokenwise(query_original),
        encode_smiles_tokenwise(partner_original),
    )

    return {
        "line": line_number,
        "direction": direction,
        "aug_index": 0,
        "target_quantile": "original",
        "query_pool_rank_index": "",
        "query_pool_size": "",
        "partner_pool_size": "",
        "query_internal_distance": 0,
        "pair_distance": pair_dist,
        "input_src": input_src,
        "input_trg": input_trg,
        "aligned_src": query_original,
        "aligned_trg": partner_original,
        "status": "original_pair",
    }


def build_direction_rows(
    line_number: int,
    direction: str,
    input_src: str,
    input_trg: str,
    query_original: str,
    partner_original: str,
    query_pool: Sequence[str],
    partner_pool: Sequence[str],
    n_select: int,
) -> List[Dict[str, object]]:
    """
    Build one direction:

      query_original -> partner_original
      query_variant_i -> best_partner_i
    """
    rows = [
        make_original_row(
            line_number=line_number,
            direction=direction,
            input_src=input_src,
            input_trg=input_trg,
            query_original=query_original,
            partner_original=partner_original,
        )
    ]

    selected_query_variants = select_distributed_query_variants(
        original_smi=query_original,
        generated_smiles=query_pool,
        n_select=n_select,
    )

    for aug_index, variant in enumerate(selected_query_variants, start=1):
        query_variant = str(variant["smi"])

        best_partner, pair_dist = select_best_aligned_partner(
            query_smi=query_variant,
            partner_pool=partner_pool,
            fallback_smi=partner_original,
        )

        rows.append(
            {
                "line": line_number,
                "direction": direction,
                "aug_index": aug_index,
                "target_quantile": variant["target_quantile"],
                "query_pool_rank_index": variant["pool_rank_index"],
                "query_pool_size": variant["pool_size"],
                "partner_pool_size": len(partner_pool),
                "query_internal_distance": variant["internal_distance"],
                "pair_distance": pair_dist,
                "input_src": input_src,
                "input_trg": input_trg,
                "aligned_src": query_variant,
                "aligned_trg": best_partner,
                "status": "augmented_aligned_pair",
            }
        )

    return rows


def make_skipped_result(
    line_number: int,
    src: str,
    trg: str,
    status: str,
) -> Dict[str, object]:
    return {
        "line": line_number,
        "input_src": src,
        "input_trg": trg,
        "status": status,
        "rows": [],
        "src_pool_size": 0,
        "trg_pool_size": 0,
        "n_output_rows": 0,
    }


def augment_one_pair_bidirectional(job) -> Dict[str, object]:
    line_number, src, trg = job

    src = src.strip()
    trg = trg.strip()

    if not src or not trg:
        return make_skipped_result(line_number, src, trg, "blank_src_or_trg")

    src_mol = Chem.MolFromSmiles(src)
    trg_mol = Chem.MolFromSmiles(trg)

    if src_mol is None:
        return make_skipped_result(line_number, src, trg, "invalid_src")

    if trg_mol is None:
        return make_skipped_result(line_number, src, trg, "invalid_trg")

    if not token_len_ok(src):
        return make_skipped_result(line_number, src, trg, "src_too_long")

    if not token_len_ok(trg):
        return make_skipped_result(line_number, src, trg, "trg_too_long")

    pool_src = generate_randomized_smiles_pool(src)
    pool_trg = generate_randomized_smiles_pool(trg)

    rows = []

    rows.extend(
        build_direction_rows(
            line_number=line_number,
            direction="forward_src_to_trg",
            input_src=src,
            input_trg=trg,
            query_original=src,
            partner_original=trg,
            query_pool=pool_src,
            partner_pool=pool_trg,
            n_select=AUG_PER_DIRECTION_GLOBAL,
        )
    )

    rows.extend(
        build_direction_rows(
            line_number=line_number,
            direction="reverse_trg_to_src",
            input_src=src,
            input_trg=trg,
            query_original=trg,
            partner_original=src,
            query_pool=pool_trg,
            partner_pool=pool_src,
            n_select=AUG_PER_DIRECTION_GLOBAL,
        )
    )

    expected_rows = 2 * (1 + AUG_PER_DIRECTION_GLOBAL)

    if len(rows) == expected_rows:
        status = "ok_full_nominal_count"
    else:
        status = "ok_partial_count"

    return {
        "line": line_number,
        "input_src": src,
        "input_trg": trg,
        "status": status,
        "rows": rows,
        "src_pool_size": len(pool_src),
        "trg_pool_size": len(pool_trg),
        "n_output_rows": len(rows),
    }


def read_parallel_jobs(
    src_path: Path,
    trg_path: Path,
    split_name: str,
) -> List[Tuple[int, str, str]]:
    if not src_path.exists():
        raise FileNotFoundError(f"Missing {split_name} source file: {src_path}")

    if not trg_path.exists():
        raise FileNotFoundError(f"Missing {split_name} target file: {trg_path}")

    with src_path.open("r", encoding="utf-8") as f_src:
        src_lines = [line.rstrip("\n\r") for line in f_src]

    with trg_path.open("r", encoding="utf-8") as f_trg:
        trg_lines = [line.rstrip("\n\r") for line in f_trg]

    if len(src_lines) != len(trg_lines):
        raise ValueError(
            f"Line-count mismatch for {split_name}: "
            f"{src_path} has {len(src_lines)} lines, "
            f"but {trg_path} has {len(trg_lines)} lines."
        )

    return [
        (line_number, src, trg)
        for line_number, (src, trg) in enumerate(zip(src_lines, trg_lines), start=1)
    ]


def tsv_value(value) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    return text


def write_aligned_split(
    results: Sequence[Dict[str, object]],
    output_dir: Path,
    split_name: str,
) -> Dict[str, object]:
    src_path = output_dir / f"{split_name}.src"
    trg_path = output_dir / f"{split_name}.trg"
    report_path = output_dir / f"{split_name}.alignment.tsv"

    output_rows = []

    for result in results:
        output_rows.extend(result["rows"])

    with src_path.open("w", encoding="utf-8") as f_src:
        with trg_path.open("w", encoding="utf-8") as f_trg:
            for row in output_rows:
                f_src.write(str(row["aligned_src"]) + "\n")
                f_trg.write(str(row["aligned_trg"]) + "\n")

    header = [
        "line",
        "direction",
        "aug_index",
        "target_quantile",
        "query_pool_rank_index",
        "query_pool_size",
        "partner_pool_size",
        "query_internal_distance",
        "pair_distance",
        "status",
        "input_src",
        "input_trg",
        "aligned_src",
        "aligned_trg",
    ]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

        for result in results:
            if result["rows"]:
                for row in result["rows"]:
                    f.write(
                        "\t".join(
                            [
                                tsv_value(row["line"]),
                                tsv_value(row["direction"]),
                                tsv_value(row["aug_index"]),
                                tsv_value(row["target_quantile"]),
                                tsv_value(row["query_pool_rank_index"]),
                                tsv_value(row["query_pool_size"]),
                                tsv_value(row["partner_pool_size"]),
                                tsv_value(row["query_internal_distance"]),
                                tsv_value(row["pair_distance"]),
                                tsv_value(row["status"]),
                                tsv_value(row["input_src"]),
                                tsv_value(row["input_trg"]),
                                tsv_value(row["aligned_src"]),
                                tsv_value(row["aligned_trg"]),
                            ]
                        )
                        + "\n"
                    )
            else:
                f.write(
                    "\t".join(
                        [
                            tsv_value(result["line"]),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            tsv_value(result["status"]),
                            tsv_value(result["input_src"]),
                            tsv_value(result["input_trg"]),
                            "",
                            "",
                        ]
                    )
                    + "\n"
                )

    status_counts = {}
    n_input_rows = len(results)
    n_output_rows = len(output_rows)
    n_skipped_input_rows = 0
    full_nominal_input_rows = 0
    partial_input_rows = 0

    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

        if not result["rows"]:
            n_skipped_input_rows += 1
        elif status == "ok_full_nominal_count":
            full_nominal_input_rows += 1
        else:
            partial_input_rows += 1

    print(f"{split_name}: input rows = {n_input_rows}")
    print(f"{split_name}: output rows = {n_output_rows}")
    print(f"{split_name}: skipped input rows = {n_skipped_input_rows}")
    print(f"{split_name}: full nominal 10x rows = {full_nominal_input_rows}")
    print(f"{split_name}: partial rows = {partial_input_rows}")
    print(f"{split_name}: src written = {src_path}")
    print(f"{split_name}: trg written = {trg_path}")
    print(f"{split_name}: report written = {report_path}")

    return {
        "split": split_name,
        "input_rows": n_input_rows,
        "output_rows": n_output_rows,
        "skipped_input_rows": n_skipped_input_rows,
        "full_nominal_input_rows": full_nominal_input_rows,
        "partial_input_rows": partial_input_rows,
        "status_counts": status_counts,
        "paths": {
            "src": str(src_path),
            "trg": str(trg_path),
            "alignment_report": str(report_path),
        },
    }


def align_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    src_filename: str,
    trg_filename: str,
    args,
) -> Dict[str, object]:
    src_path = input_dir / src_filename
    trg_path = input_dir / trg_filename

    jobs = read_parallel_jobs(
        src_path=src_path,
        trg_path=trg_path,
        split_name=split_name,
    )

    print(f"\nAligning and augmenting {split_name}")
    print(f"Source: {src_path}")
    print(f"Target: {trg_path}")
    print(f"Input rows: {len(jobs)}")

    start = time.time()
    results = []
    progress_every = max(1, args.progress_every)

    if args.num_workers == 1:
        init_worker(
            repo_root=str(args.repo_root_resolved),
            max_raw_tokens=args.max_raw_tokens,
            randomized_per_molecule=args.randomized_per_molecule,
            aug_per_direction=args.aug_per_direction,
        )

        for i, job in enumerate(jobs, start=1):
            results.append(augment_one_pair_bidirectional(job))

            if i % progress_every == 0 or i == len(jobs):
                elapsed = time.time() - start
                print(
                    f"{split_name}: processed {i}/{len(jobs)} "
                    f"in {format_seconds(elapsed)}"
                )

    else:
        chunksize = max(1, min(50, len(jobs) // max(1, args.num_workers)))

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(
                str(args.repo_root_resolved),
                args.max_raw_tokens,
                args.randomized_per_molecule,
                args.aug_per_direction,
            ),
        ) as ex:
            for i, result in enumerate(
                ex.map(augment_one_pair_bidirectional, jobs, chunksize=chunksize),
                start=1,
            ):
                results.append(result)

                if i % progress_every == 0 or i == len(jobs):
                    elapsed = time.time() - start
                    print(
                        f"{split_name}: processed {i}/{len(jobs)} "
                        f"in {format_seconds(elapsed)}"
                    )

    split_summary = write_aligned_split(
        results=results,
        output_dir=output_dir,
        split_name=split_name,
    )

    elapsed = time.time() - start
    split_summary["elapsed_sec"] = elapsed

    print(f"{split_name}: finished in {format_seconds(elapsed)}")

    return split_summary


def optional_split_exists(
    input_dir: Path,
    src_filename: str,
    trg_filename: str,
    split_name: str,
) -> bool:
    src_path = input_dir / src_filename
    trg_path = input_dir / trg_filename

    src_exists = src_path.exists()
    trg_exists = trg_path.exists()

    if src_exists and trg_exists:
        return True

    if not src_exists and not trg_exists:
        return False

    raise FileNotFoundError(
        f"Only one {split_name} file exists. Both source and target are required. "
        f"src_exists={src_exists}, trg_exists={trg_exists}, "
        f"src={src_path}, trg={trg_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bidirectional distributed-alignment augmenter for ANNalog src/trg files."
    )

    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Folder containing train.src, train.trg, val.src, val.trg, and optionally test.src/test.trg.",
    )

    parser.add_argument(
        "--output-dataset",
        required=True,
        help="Folder where augmented aligned src/trg files will be written.",
    )

    parser.add_argument(
        "--repo-root",
        required=True,
        help="Parent folder containing the annalog package.",
    )

    parser.add_argument("--train-src", default="train.src")
    parser.add_argument("--train-trg", default="train.trg")
    parser.add_argument("--val-src", default="val.src")
    parser.add_argument("--val-trg", default="val.trg")
    parser.add_argument("--test-src", default="test.src")
    parser.add_argument("--test-trg", default="test.trg")

    parser.add_argument(
        "--randomized-per-molecule",
        type=int,
        default=DEFAULT_RANDOMIZED_PER_MOLECULE,
        help="Number of randomized SMILES to generate per molecule.",
    )

    parser.add_argument(
        "--aug-per-direction",
        type=int,
        default=DEFAULT_AUG_PER_DIRECTION,
        help="Number of distributed randomized query variants per direction.",
    )

    parser.add_argument(
        "--max-raw-tokens",
        type=int,
        default=DEFAULT_MAX_RAW_TOKENS,
        help="Maximum tokenizer token count allowed for each randomized SMILES.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes for alignment.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N input rows.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")

    if args.randomized_per_molecule < 0:
        raise ValueError("--randomized-per-molecule must be non-negative.")

    if args.aug_per_direction < 0:
        raise ValueError("--aug-per-direction must be non-negative.")

    if args.max_raw_tokens < 1:
        raise ValueError("--max-raw-tokens must be at least 1.")

    args.input_dataset = Path(args.input_dataset).resolve()
    args.output_dataset = Path(args.output_dataset).resolve()
    args.output_dataset.mkdir(parents=True, exist_ok=True)

    args.repo_root_resolved = add_repo_root_to_syspath(args.repo_root)

    nominal_rows_per_pair = 2 * (1 + args.aug_per_direction)

    print("Bidirectional distributed-alignment dataset augmentation")
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output dataset: {args.output_dataset}")
    print(f"Repo root: {args.repo_root_resolved}")
    print(f"Randomized SMILES per molecule: {args.randomized_per_molecule}")
    print(f"Augmented variants per direction: {args.aug_per_direction}")
    print(f"Nominal rows per input pair: {nominal_rows_per_pair}")
    print(f"Max raw tokens: {args.max_raw_tokens}")
    print(f"Workers: {args.num_workers}")

    summaries = {}

    summaries["train"] = align_split(
        split_name="train",
        input_dir=args.input_dataset,
        output_dir=args.output_dataset,
        src_filename=args.train_src,
        trg_filename=args.train_trg,
        args=args,
    )

    summaries["val"] = align_split(
        split_name="val",
        input_dir=args.input_dataset,
        output_dir=args.output_dataset,
        src_filename=args.val_src,
        trg_filename=args.val_trg,
        args=args,
    )

    has_test = optional_split_exists(
        input_dir=args.input_dataset,
        src_filename=args.test_src,
        trg_filename=args.test_trg,
        split_name="test",
    )

    if has_test:
        summaries["test"] = align_split(
            split_name="test",
            input_dir=args.input_dataset,
            output_dir=args.output_dataset,
            src_filename=args.test_src,
            trg_filename=args.test_trg,
            args=args,
        )
    else:
        summaries["test"] = None
        print("\nTest files not found. Skipping test alignment/augmentation.")

    summary = {
        "mode": "bidirectional_distributed_alignment_augmentation",
        "input_dataset": str(args.input_dataset),
        "output_dataset": str(args.output_dataset),
        "repo_root": str(args.repo_root_resolved),
        "randomized_per_molecule": args.randomized_per_molecule,
        "aug_per_direction": args.aug_per_direction,
        "nominal_rows_per_input_pair": nominal_rows_per_pair,
        "max_raw_tokens": args.max_raw_tokens,
        "num_workers": args.num_workers,
        "splits": summaries,
    }

    summary_path = args.output_dataset / "alignment_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nAlignment and augmentation finished.")
    print(f"Augmented aligned dataset folder: {args.output_dataset}")
    print(f"Alignment summary: {summary_path}")


if __name__ == "__main__":
    main()
