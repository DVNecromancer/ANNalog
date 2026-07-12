#!/usr/bin/env python3
"""
ANNalog training from scratch using existing parallel .src/.trg files.

Expected required input layout:

  dataset_dir/
    train.src
    train.trg
    val.src
    val.trg

Optional test layout:

  dataset_dir/
    test.src
    test.trg

Usage of splits:

  train -> updates model weights
  val   -> selects best checkpoint
  test  -> final reporting only, never used for training or checkpoint selection

This script does NOT load a pretrained checkpoint.
It initializes the seq2seq model normally from the ANNalog model definitions.
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_SAVE_INTERVAL = 10
DEFAULT_CLIP = 1.0
DEFAULT_MAX_SEQ_LEN = 102


def str2bool(v):
    if isinstance(v, bool):
        return v

    v = v.lower()

    if v in {"true", "1", "yes", "y"}:
        return True

    if v in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m}m {s}s"

    if m > 0:
        return f"{m}m {s}s"

    return f"{s}s"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_repo_root_to_syspath(repo_root: Optional[str]) -> Path:
    if repo_root is None:
        repo_root_path = Path(__file__).resolve().parent
    else:
        repo_root_path = Path(repo_root).resolve()

    if str(repo_root_path) not in sys.path:
        sys.path.insert(0, str(repo_root_path))

    return repo_root_path


def load_vocab_pickle(vocab_path: Path):
    try:
        with vocab_path.open("rb") as f:
            return pickle.load(f)

    except ModuleNotFoundError as e:
        if "torchtext" in str(e):
            raise RuntimeError(
                "The vocab pickle appears to have been serialized from a torchtext class. "
                "Use the ANNalog packaged torchtext-free vocab or convert the old vocab first."
            ) from e

        raise


class VocabAdapter:
    def __init__(self, vocab_obj):
        self.vocab_obj = vocab_obj
        self.kind = None

        if isinstance(vocab_obj, dict) and "itos" in vocab_obj and "stoi" in vocab_obj:
            self.kind = "dict_itos_stoi"
            self.itos = list(vocab_obj["itos"])
            self.stoi = dict(vocab_obj["stoi"])

        elif hasattr(vocab_obj, "_tokens") and hasattr(vocab_obj, "__getitem__"):
            self.kind = "annalog_vocabulary"

        elif hasattr(vocab_obj, "stoi"):
            self.kind = "torchtext_like"

        else:
            raise TypeError(
                "Unsupported vocabulary object. Expected one of: "
                "dict with {'itos','stoi'}, ANNalog Vocabulary, "
                "or torchtext-like vocab with .stoi"
            )

    def __len__(self) -> int:
        if self.kind == "dict_itos_stoi":
            return len(self.itos)

        return len(self.vocab_obj)

    def contains(self, token: str) -> bool:
        if self.kind == "dict_itos_stoi":
            return token in self.stoi

        if self.kind == "annalog_vocabulary":
            return token in self.vocab_obj

        return token in self.vocab_obj.stoi

    def token_to_idx(self, token: str) -> int:
        if self.kind == "dict_itos_stoi":
            return int(self.stoi[token])

        if self.kind == "annalog_vocabulary":
            return int(self.vocab_obj[token])

        return int(self.vocab_obj.stoi[token])


class PairSequenceDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        split_name: str,
        tokenizer,
        vocab: VocabAdapter,
        bos_token: str,
        eos_token: str,
        max_seq_len: int,
    ):
        self.examples = []

        self.stats = {
            "split": split_name,
            "input_pairs": 0,
            "kept_pairs": 0,
            "dropped_blank_pairs": 0,
            "dropped_oov_pairs": 0,
            "dropped_too_long_pairs": 0,
        }

        for src, trg in pairs:
            self.stats["input_pairs"] += 1

            src = src.strip()
            trg = trg.strip()

            if not src or not trg:
                self.stats["dropped_blank_pairs"] += 1
                continue

            src_ids, src_issue = encode_smiles(
                src,
                tokenizer,
                vocab,
                bos_token,
                eos_token,
                max_seq_len,
            )

            trg_ids, trg_issue = encode_smiles(
                trg,
                tokenizer,
                vocab,
                bos_token,
                eos_token,
                max_seq_len,
            )

            if src_issue is not None or trg_issue is not None:
                issue = src_issue if src_issue is not None else trg_issue

                if issue["reason"] == "oov":
                    self.stats["dropped_oov_pairs"] += 1

                elif issue["reason"] == "too_long":
                    self.stats["dropped_too_long_pairs"] += 1

                continue

            self.examples.append((src_ids, trg_ids))

        self.stats["kept_pairs"] = len(self.examples)

        if len(self.examples) == 0:
            raise ValueError(f"No usable examples remain in split '{split_name}'.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DeviceDataLoader:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield SimpleNamespace(
                src=batch.src.to(self.device, non_blocking=True),
                trg=batch.trg.to(self.device, non_blocking=True),
            )


def make_collate_fn(pad_idx: int):
    def collate(batch):
        src_tensors, trg_tensors = zip(*batch)

        src = pad_sequence(
            src_tensors,
            batch_first=True,
            padding_value=pad_idx,
        )

        trg = pad_sequence(
            trg_tensors,
            batch_first=True,
            padding_value=pad_idx,
        )

        return SimpleNamespace(src=src, trg=trg)

    return collate


def encode_smiles(
    smi: str,
    tokenizer,
    vocab: VocabAdapter,
    bos_token: str,
    eos_token: str,
    max_seq_len: int,
):
    tokens = tokenizer.tokenize(smi, with_begin_and_end=False)
    full_tokens = [bos_token] + tokens + [eos_token]

    missing = sorted({tok for tok in full_tokens if not vocab.contains(tok)})

    if missing:
        return None, {
            "reason": "oov",
            "missing_tokens": missing,
            "seq_len": len(full_tokens),
        }

    if len(full_tokens) > max_seq_len:
        return None, {
            "reason": "too_long",
            "seq_len": len(full_tokens),
        }

    ids = [vocab.token_to_idx(tok) for tok in full_tokens]

    return torch.tensor(ids, dtype=torch.long), None


def save_weights_only(save_path: Path, model: torch.nn.Module) -> None:
    torch.save(model.state_dict(), save_path)


def make_jsonable_args(args) -> dict:
    output = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value

    return output


def save_training_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    test_loss: Optional[float],
    best_valid_loss: float,
    args_dict: dict,
    dataset_stats: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "test_loss": test_loss,
        "best_valid_loss": best_valid_loss,
        "args": args_dict,
        "dataset_stats": dataset_stats,
    }

    torch.save(payload, save_path)


def resolve_dataset_file(dataset_dir: Path, path_arg: str) -> Path:
    path = Path(path_arg)

    if not path.is_absolute():
        path = dataset_dir / path

    return path.resolve()


def read_parallel_pairs(
    src_path: Path,
    trg_path: Path,
    split_name: str,
) -> List[Tuple[str, str]]:
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

    return list(zip(src_lines, trg_lines))


def resolve_optional_test_paths(args, dataset_dir: Path) -> None:
    args.test_src_path = resolve_dataset_file(dataset_dir, args.test_src)
    args.test_trg_path = resolve_dataset_file(dataset_dir, args.test_trg)

    test_src_exists = args.test_src_path.exists()
    test_trg_exists = args.test_trg_path.exists()

    if test_src_exists and test_trg_exists:
        args.has_test = True

    elif not test_src_exists and not test_trg_exists:
        args.has_test = False

    else:
        raise FileNotFoundError(
            "Only one test file exists. Both test source and test target are required. "
            f"test_src_exists={test_src_exists}, test_trg_exists={test_trg_exists}, "
            f"test_src={args.test_src_path}, test_trg={args.test_trg_path}"
        )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    loader_workers: int,
    collate_fn,
    pin_memory: bool,
    device: torch.device,
) -> DeviceDataLoader:
    return DeviceDataLoader(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=loader_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        ),
        device,
    )


def run_training_from_scratch(
    args,
    tokenizer,
    vocab: VocabAdapter,
    output_dir: Path,
) -> Dict[str, object]:
    from annalog.model_files import seq2seq_attention

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining model from scratch using existing src/trg files")
    print(f"Dataset folder: {args.dataset_dir}")
    print(f"Train src: {args.train_src_path}")
    print(f"Train trg: {args.train_trg_path}")
    print(f"Val src: {args.val_src_path}")
    print(f"Val trg: {args.val_trg_path}")

    if args.has_test:
        print(f"Test src: {args.test_src_path}")
        print(f"Test trg: {args.test_trg_path}")
    else:
        print("Test files: not found, skipping final test evaluation")

    print(f"Vocab: {args.vocab_path}")
    print(f"Training output folder: {training_dir}")

    for special_token_name, special_token in {
        "bos": args.bos_token,
        "eos": args.eos_token,
        "pad": args.pad_token,
        "unk": args.unk_token,
    }.items():
        if not vocab.contains(special_token):
            raise ValueError(
                f"The {special_token_name} token '{special_token}' is not present in the vocabulary."
            )

    train_pairs = read_parallel_pairs(
        args.train_src_path,
        args.train_trg_path,
        "train",
    )

    valid_pairs = read_parallel_pairs(
        args.val_src_path,
        args.val_trg_path,
        "val",
    )

    if args.has_test:
        test_pairs = read_parallel_pairs(
            args.test_src_path,
            args.test_trg_path,
            "test",
        )
    else:
        test_pairs = None

    train_data = PairSequenceDataset(
        pairs=train_pairs,
        split_name="train",
        tokenizer=tokenizer,
        vocab=vocab,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        max_seq_len=args.max_seq_len,
    )

    valid_data = PairSequenceDataset(
        pairs=valid_pairs,
        split_name="val",
        tokenizer=tokenizer,
        vocab=vocab,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        max_seq_len=args.max_seq_len,
    )

    if test_pairs is not None:
        test_data = PairSequenceDataset(
            pairs=test_pairs,
            split_name="test",
            tokenizer=tokenizer,
            vocab=vocab,
            bos_token=args.bos_token,
            eos_token=args.eos_token,
            max_seq_len=args.max_seq_len,
        )
    else:
        test_data = None

    dataset_stats = {
        "paths": {
            "train_src": str(args.train_src_path),
            "train_trg": str(args.train_trg_path),
            "val_src": str(args.val_src_path),
            "val_trg": str(args.val_trg_path),
            "test_src": str(args.test_src_path) if args.has_test else None,
            "test_trg": str(args.test_trg_path) if args.has_test else None,
        },
        "train": train_data.stats,
        "val": valid_data.stats,
        "test": test_data.stats if test_data is not None else None,
    }

    with (training_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)

    device = resolve_device(args.device)
    pad_idx = vocab.token_to_idx(args.pad_token)

    print(f"Device: {device}")
    print(f"Train file pairs: {len(train_pairs)}")
    print(f"Valid file pairs: {len(valid_pairs)}")

    if test_pairs is not None:
        print(f"Test file pairs: {len(test_pairs)}")

    print(f"Train examples kept after token checks: {len(train_data)}")
    print(f"Valid examples kept after token checks: {len(valid_data)}")

    if test_data is not None:
        print(f"Test examples kept after token checks: {len(test_data)}")

    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")

    collate_fn = make_collate_fn(pad_idx)
    pin_memory = device.type == "cuda"

    train_loader = build_loader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        loader_workers=args.loader_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        device=device,
    )

    valid_loader = build_loader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        loader_workers=args.loader_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        device=device,
    )

    if test_data is not None:
        test_loader = build_loader(
            dataset=test_data,
            batch_size=args.batch_size,
            shuffle=False,
            loader_workers=args.loader_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            device=device,
        )
    else:
        test_loader = None

    input_dim = len(vocab)
    output_dim = len(vocab)

    enc = seq2seq_attention.Encoder(
        input_dim,
        args.hid_dim,
        args.enc_layers,
        args.enc_heads,
        args.enc_pf_dim,
        args.enc_dropout,
        device,
        args.max_seq_len,
    )

    dec = seq2seq_attention.Decoder(
        output_dim,
        args.hid_dim,
        args.dec_layers,
        args.dec_heads,
        args.dec_pf_dim,
        args.dec_dropout,
        device,
        args.max_seq_len,
    )

    model = seq2seq_attention.Seq2Seq(
        enc,
        dec,
        pad_idx,
        pad_idx,
        device,
    ).to(device)

    if hasattr(seq2seq_attention, "count_parameters"):
        n_params = seq2seq_attention.count_parameters(model)
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    args_dict = make_jsonable_args(args)

    with (training_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, default=str)

    log_path = training_dir / "training_log.tsv"

    with log_path.open("w", encoding="utf-8") as f:
        f.write(
            "epoch\t"
            "train_loss\t"
            "valid_loss\t"
            "train_eval_loss\t"
            "best_valid_loss_before_epoch\t"
            "elapsed_sec\n"
        )

    best_valid_loss = math.inf
    best_epoch = None

    best_path = training_dir / "best_model.pt"
    best_resume_path = training_dir / "best_model_training.ckpt"

    train_loss = math.inf
    valid_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = seq2seq_attention.train(
            model,
            train_loader,
            optimizer,
            criterion,
            args.clip,
        )

        valid_loss = seq2seq_attention.evaluate(
            model,
            valid_loader,
            criterion,
        )

        if args.evaluate_train_split:
            train_eval_loss = seq2seq_attention.evaluate(
                model,
                train_loader,
                criterion,
            )
        else:
            train_eval_loss = float("nan")

        elapsed = time.time() - epoch_start

        with log_path.open("a", encoding="utf-8") as f:
            train_eval_str = (
                "nan" if math.isnan(train_eval_loss) else f"{train_eval_loss:.6f}"
            )

            best_str = (
                "inf" if math.isinf(best_valid_loss) else f"{best_valid_loss:.6f}"
            )

            f.write(
                f"{epoch}\t"
                f"{train_loss:.6f}\t"
                f"{valid_loss:.6f}\t"
                f"{train_eval_str}\t"
                f"{best_str}\t"
                f"{elapsed:.3f}\n"
            )

        msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"valid={valid_loss:.6f}"
        )

        if args.evaluate_train_split:
            msg += f" | train_eval={train_eval_loss:.6f}"

        msg += f" | elapsed={format_seconds(elapsed)}"

        print(msg)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

            save_weights_only(best_path, model)

            save_training_checkpoint(
                save_path=best_resume_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                test_loss=None,
                best_valid_loss=best_valid_loss,
                args_dict=args_dict,
                dataset_stats=dataset_stats,
            )

            print(f"  New best validation loss. Saved weights: {best_path}")
            print(f"  Saved resumable checkpoint: {best_resume_path}")

        if epoch % args.save_interval == 0:
            ckpt_path = training_dir / f"epoch_{epoch}_training.ckpt"

            save_training_checkpoint(
                save_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                test_loss=None,
                best_valid_loss=best_valid_loss,
                args_dict=args_dict,
                dataset_stats=dataset_stats,
            )

            print(f"  Saved periodic training checkpoint: {ckpt_path}")

    final_path = training_dir / "final_model.pt"
    final_resume_path = training_dir / "final_model_training.ckpt"

    save_weights_only(final_path, model)

    save_training_checkpoint(
        save_path=final_resume_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        train_loss=train_loss,
        valid_loss=valid_loss,
        test_loss=None,
        best_valid_loss=best_valid_loss,
        args_dict=args_dict,
        dataset_stats=dataset_stats,
    )

    print(f"Saved final weights: {final_path}")
    print(f"Saved final resumable checkpoint: {final_resume_path}")

    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state)
        print("Loaded best validation checkpoint for final evaluation.")

    best_model_valid_loss = seq2seq_attention.evaluate(
        model,
        valid_loader,
        criterion,
    )

    if test_loader is not None:
        best_model_test_loss = seq2seq_attention.evaluate(
            model,
            test_loader,
            criterion,
        )
    else:
        best_model_test_loss = None

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss recorded: {best_valid_loss:.6f}")
    print(f"Best-model validation loss after reload: {best_model_valid_loss:.6f}")

    if best_model_test_loss is not None:
        print(f"Best-model test loss after reload: {best_model_test_loss:.6f}")
    else:
        print("Best-model test loss after reload: skipped, no test files provided")

    print(f"Final weights file: {final_path}")

    final_best_resume_path = training_dir / "best_model_final_eval_training.ckpt"

    save_training_checkpoint(
        save_path=final_best_resume_path,
        model=model,
        optimizer=optimizer,
        epoch=best_epoch if best_epoch is not None else args.epochs,
        train_loss=train_loss,
        valid_loss=best_model_valid_loss,
        test_loss=best_model_test_loss,
        best_valid_loss=best_valid_loss,
        args_dict=args_dict,
        dataset_stats=dataset_stats,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_model_valid_loss": best_model_valid_loss,
        "best_model_test_loss": best_model_test_loss,
        "final_weights": str(final_path),
        "final_training_checkpoint": str(final_resume_path),
        "best_weights": str(best_path),
        "best_training_checkpoint": str(best_resume_path),
        "best_final_eval_training_checkpoint": str(final_best_resume_path),
        "log_file": str(log_path),
    }

    with (training_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (training_dir / "summary.txt").open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}\t{value}\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="ANNalog training from scratch using existing parallel src/trg files."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Folder containing train.src, train.trg, val.src, val.trg, and optionally test.src/test.trg.",
    )

    parser.add_argument(
        "--train-src",
        default="train.src",
        help="Train source file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--train-trg",
        default="train.trg",
        help="Train target file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--val-src",
        default="val.src",
        help="Validation source file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--val-trg",
        default="val.trg",
        help="Validation target file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--test-src",
        default="test.src",
        help="Optional test source file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--test-trg",
        default="test.trg",
        help="Optional test target file, relative to dataset-dir unless absolute.",
    )

    parser.add_argument(
        "--vocab-path",
        required=True,
        help="Path to ANNalog vocab pickle.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder for training results.",
    )

    parser.add_argument(
        "--repo-root",
        default=None,
        help="ANNalog repo root. This should be the parent folder containing the annalog package.",
    )

    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader workers for training.",
    )

    parser.add_argument("--bos-token", default="<sos>")
    parser.add_argument("--eos-token", default="<eos>")
    parser.add_argument("--pad-token", default="<pad>")
    parser.add_argument("--unk-token", default="<unk>")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--evaluate-train-split",
        type=str2bool,
        default=True,
    )

    parser.add_argument("--hid-dim", type=int, default=256)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--dec-heads", type=int, default=8)
    parser.add_argument("--enc-pf-dim", type=int, default=512)
    parser.add_argument("--dec-pf-dim", type=int, default=512)
    parser.add_argument("--enc-dropout", type=float, default=0.1)
    parser.add_argument("--dec-dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    args.repo_root_resolved = str(add_repo_root_to_syspath(args.repo_root))

    from annalog.model_files import vocabulary

    tokenizer = vocabulary.SMILESTokenizer()

    vocab_obj = load_vocab_pickle(Path(args.vocab_path).resolve())
    vocab = VocabAdapter(vocab_obj)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args.dataset_dir = str(Path(args.dataset_dir).resolve())
    dataset_dir = Path(args.dataset_dir)

    args.train_src_path = resolve_dataset_file(dataset_dir, args.train_src)
    args.train_trg_path = resolve_dataset_file(dataset_dir, args.train_trg)
    args.val_src_path = resolve_dataset_file(dataset_dir, args.val_src)
    args.val_trg_path = resolve_dataset_file(dataset_dir, args.val_trg)

    resolve_optional_test_paths(args, dataset_dir)

    args.vocab_path = str(Path(args.vocab_path).resolve())
    args.repo_root = str(args.repo_root_resolved)

    print(f"Loaded vocab adapter kind: {vocab.kind}")
    print(f"Using fixed random seed: {SEED}")
    print(f"Output directory: {output_dir}")

    training_summary = run_training_from_scratch(
        args,
        tokenizer,
        vocab,
        output_dir,
    )

    pipeline_summary = {
        "seed": SEED,
        "mode": "training_from_scratch_existing_src_trg_files",
        "has_test": args.has_test,
        "training": training_summary,
    }

    with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)

    print("\nPipeline finished.")
    print(f"Output directory: {output_dir}")
    print(f"Training summary: {output_dir / 'training' / 'summary.txt'}")
    print(f"Loss log: {output_dir / 'training' / 'training_log.tsv'}")


if __name__ == "__main__":
    main()