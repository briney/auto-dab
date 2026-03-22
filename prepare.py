"""
Data preparation and evaluation utilities for auto-dab.

Run once to prepare data:
    uv run prepare.py --data /path/to/antibodies.csv

Then imported by train.py for dataloader and evaluation.

This file is READ-ONLY during autoresearch -- the agent must not modify it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (importable by train.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
MAX_SEQ_LEN = 320
TIME_BUDGET = 300  # 5 minutes of training
EVAL_TIMESTEPS = 100  # fixed evaluation timesteps
MASK_TOKEN_ID = 31
PAD_TOKEN_ID = 1
CLS_TOKEN_ID = 0
EOS_TOKEN_ID = 2
AA_START_IDX = 4
AA_END_IDX = 30  # exclusive

CACHE_DIR = Path(os.environ.get("AUTO_DAB_CACHE", Path.home() / ".cache" / "auto-dab"))

# ---------------------------------------------------------------------------
# Inline tokenizer (32-token fixed vocabulary, no HuggingFace dependency)
# ---------------------------------------------------------------------------

VOCAB = [
    "<cls>",   # 0
    "<pad>",   # 1
    "<eos>",   # 2
    "<unk>",   # 3
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",  # 4-13
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",  # 14-23
    "X", "B", "U", "O", "Z",  # 24-28 (non-standard)
    ".", "-",  # 29-30 (insertion, gap)
    "<mask>",  # 31
]

_TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
_ID_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(VOCAB)}
_UNK_ID = _TOKEN_TO_ID["<unk>"]


def encode(sequence: str) -> list[int]:
    """Encode an amino acid string to token IDs (no special tokens)."""
    return [_TOKEN_TO_ID.get(ch, _UNK_ID) for ch in sequence]


def encode_paired(heavy: str, light: str) -> dict[str, list[int]]:
    """Encode a paired heavy/light chain with CLS and EOS tokens.

    Format: [CLS] heavy light [EOS]
    Chain IDs: 0 for CLS+heavy, 1 for light+EOS
    """
    heavy_ids = encode(heavy)
    light_ids = encode(light)

    token_ids = [CLS_TOKEN_ID] + heavy_ids + light_ids + [EOS_TOKEN_ID]
    chain_ids = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)
    special_mask = [1] + [0] * len(heavy_ids) + [0] * len(light_ids) + [1]

    return {
        "token_ids": token_ids,
        "chain_ids": chain_ids,
        "special_tokens_mask": special_mask,
    }


def decode_tokens(token_ids: list[int] | Tensor) -> str:
    """Decode token IDs back to a string."""
    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()
    return "".join(_ID_TO_TOKEN.get(i, "?") for i in token_ids)


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

def _parse_mask_string(mask_str: str) -> list[int]:
    """Parse a string of digits into a list of ints."""
    return [int(ch) for ch in mask_str]


def prepare_data(data_path: str, val_fraction: float = 0.05, seed: int = 42) -> None:
    """Process antibody data and save cached tensors.

    Args:
        data_path: Path to CSV/TSV/Parquet with heavy_chain and light_chain columns.
        val_fraction: Fraction of data to hold out for validation.
        seed: Random seed for train/val split.
    """
    path = Path(data_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Validate required columns
    for col in ("heavy_chain", "light_chain"):
        if col not in df.columns:
            print(f"Error: required column '{col}' not found in data", file=sys.stderr)
            sys.exit(1)

    # Drop rows with missing sequences
    df = df.dropna(subset=["heavy_chain", "light_chain"]).reset_index(drop=True)

    # Filter sequences that would exceed MAX_SEQ_LEN (including CLS + EOS)
    lengths = df["heavy_chain"].str.len() + df["light_chain"].str.len() + 2
    df = df[lengths <= MAX_SEQ_LEN].reset_index(drop=True)

    print(f"Loaded {len(df)} sequences from {path}")
    print(f"  Heavy chain lengths: {df['heavy_chain'].str.len().describe().to_dict()}")
    print(f"  Light chain lengths: {df['light_chain'].str.len().describe().to_dict()}")

    # Check for optional annotation columns
    has_cdr = "heavy_cdr_mask" in df.columns and "light_cdr_mask" in df.columns
    has_nt = (
        "heavy_non_templated_mask" in df.columns
        and "light_non_templated_mask" in df.columns
    )

    if has_cdr:
        print("  CDR masks: present")
    if has_nt:
        print("  Non-templated masks: present")

    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    val_size = max(1, int(len(df) * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    print(f"  Train: {len(train_indices)} sequences")
    print(f"  Val: {len(val_indices)} sequences")

    # Encode and cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
        split_df = df.iloc[split_indices].reset_index(drop=True)
        cache = _encode_split(split_df, has_cdr, has_nt)
        cache_path = CACHE_DIR / f"{split_name}_tokens.pt"
        torch.save(cache, cache_path)
        print(f"  Saved {split_name} cache to {cache_path}")


def _encode_split(
    df: pd.DataFrame, has_cdr: bool, has_nt: bool
) -> dict[str, list]:
    """Encode a dataframe split into lists of tensors."""
    all_token_ids = []
    all_chain_ids = []
    all_special_masks = []
    all_cdr_masks = [] if has_cdr else None
    all_nt_masks = [] if has_nt else None

    for _, row in df.iterrows():
        enc = encode_paired(row["heavy_chain"], row["light_chain"])
        all_token_ids.append(enc["token_ids"])
        all_chain_ids.append(enc["chain_ids"])
        all_special_masks.append(enc["special_tokens_mask"])

        if has_cdr:
            heavy_cdr = _parse_mask_string(row["heavy_cdr_mask"])
            light_cdr = _parse_mask_string(row["light_cdr_mask"])
            cdr = [0] + heavy_cdr + light_cdr + [0]
            all_cdr_masks.append(cdr)

        if has_nt:
            heavy_nt = _parse_mask_string(row["heavy_non_templated_mask"])
            light_nt = _parse_mask_string(row["light_non_templated_mask"])
            nt = [0] + heavy_nt + light_nt + [0]
            all_nt_masks.append(nt)

    cache = {
        "token_ids": all_token_ids,
        "chain_ids": all_chain_ids,
        "special_tokens_mask": all_special_masks,
    }
    if has_cdr:
        cache["cdr_mask"] = all_cdr_masks
    if has_nt:
        cache["non_templated_mask"] = all_nt_masks

    return cache


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class AntibodyDataset(Dataset):
    """Dataset wrapping pre-tokenized antibody sequences."""

    def __init__(self, cache: dict[str, list]) -> None:
        self.token_ids = cache["token_ids"]
        self.chain_ids = cache["chain_ids"]
        self.special_tokens_mask = cache["special_tokens_mask"]
        self.cdr_mask = cache.get("cdr_mask")
        self.non_templated_mask = cache.get("non_templated_mask")

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = {
            "token_ids": self.token_ids[idx],
            "chain_ids": self.chain_ids[idx],
            "special_tokens_mask": self.special_tokens_mask[idx],
        }
        if self.cdr_mask is not None:
            item["cdr_mask"] = self.cdr_mask[idx]
        if self.non_templated_mask is not None:
            item["non_templated_mask"] = self.non_templated_mask[idx]
        return item


def _collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, Tensor]:
    """Collate variable-length sequences with dynamic padding."""
    max_len = min(max(len(item["token_ids"]) for item in batch), MAX_SEQ_LEN)

    token_ids = []
    chain_ids = []
    attention_mask = []
    special_masks = []
    cdr_masks = []
    nt_masks = []

    has_cdr = "cdr_mask" in batch[0]
    has_nt = "non_templated_mask" in batch[0]

    for item in batch:
        seq_len = min(len(item["token_ids"]), max_len)
        pad_len = max_len - seq_len

        token_ids.append(item["token_ids"][:seq_len] + [PAD_TOKEN_ID] * pad_len)
        chain_ids.append(item["chain_ids"][:seq_len] + [0] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)
        special_masks.append(item["special_tokens_mask"][:seq_len] + [1] * pad_len)

        if has_cdr:
            cdr_masks.append(item["cdr_mask"][:seq_len] + [0] * pad_len)
        if has_nt:
            nt_masks.append(item["non_templated_mask"][:seq_len] + [0] * pad_len)

    result: dict[str, Tensor] = {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "special_tokens_mask": torch.tensor(special_masks, dtype=torch.bool),
    }
    if has_cdr:
        result["cdr_mask"] = torch.tensor(cdr_masks, dtype=torch.long)
    if has_nt:
        result["non_templated_mask"] = torch.tensor(nt_masks, dtype=torch.long)

    return result


def make_dataloader(
    batch_size: int,
    split: str = "train",
    num_workers: int = 2,
) -> DataLoader:
    """Create a DataLoader from cached pre-tokenized data.

    Args:
        batch_size: Batch size.
        split: "train" or "val".
        num_workers: Number of data loading workers.

    Returns:
        DataLoader yielding batched tensors.
    """
    cache_path = CACHE_DIR / f"{split}_tokens.pt"
    if not cache_path.exists():
        print(f"Error: cached data not found at {cache_path}", file=sys.stderr)
        print("Run: uv run prepare.py --data /path/to/data.csv", file=sys.stderr)
        sys.exit(1)

    cache = torch.load(cache_path, weights_only=False)
    dataset = AntibodyDataset(cache)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Evaluation (fixed protocol -- DO NOT MODIFY)
# ---------------------------------------------------------------------------

def _cosine_mask_rate(timestep: int, num_timesteps: int) -> float:
    """Fixed cosine schedule mask rate for evaluation."""
    t_norm = timestep / num_timesteps
    return 1.0 - math.cos(t_norm * math.pi / 2)


@torch.no_grad()
def evaluate(model: torch.nn.Module, batch_size: int = 64) -> dict[str, float]:
    """Evaluate model using a fixed protocol.

    Uses a cosine noise schedule with uniform masking across all timesteps.
    This gives a consistent val_loss regardless of training choices.

    Args:
        model: The model to evaluate. Must accept
            (token_ids, chain_ids, attention_mask) and return dict with "logits".
        batch_size: Evaluation batch size.

    Returns:
        Dict with val_loss and val_accuracy.
    """
    device = next(model.parameters()).device
    model.eval()

    val_loader = make_dataloader(batch_size, split="val", num_workers=0)

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for batch in val_loader:
        token_ids = batch["token_ids"].to(device)
        chain_ids = batch["chain_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_tokens_mask = batch["special_tokens_mask"].to(device)

        bsz, seq_len = token_ids.shape

        # Evaluate at every timestep and average
        for t in range(1, EVAL_TIMESTEPS + 1):
            mask_rate = _cosine_mask_rate(t, EVAL_TIMESTEPS)

            # Uniform masking: random mask at this rate
            maskable = attention_mask.bool() & ~special_tokens_mask
            rand = torch.rand(bsz, seq_len, device=device)
            mask_labels = (rand < mask_rate) & maskable

            # Skip if nothing is masked
            num_masked = mask_labels.sum().item()
            if num_masked == 0:
                continue

            masked_ids = token_ids.clone()
            masked_ids[mask_labels] = MASK_TOKEN_ID

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(masked_ids, chain_ids, attention_mask)
                logits = outputs["logits"]

            # Compute loss on masked positions
            logits_flat = logits.view(-1, VOCAB_SIZE)
            targets_flat = token_ids.view(-1)
            mask_flat = mask_labels.view(-1)

            loss_per_token = F.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            masked_loss = (loss_per_token * mask_flat.float()).sum().item()
            total_loss += masked_loss

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct = ((preds == token_ids) & mask_labels).sum().item()
            total_correct += correct
            total_masked += num_masked

    model.train()

    val_loss = total_loss / max(total_masked, 1)
    val_accuracy = total_correct / max(total_masked, 1)

    return {"val_loss": val_loss, "val_accuracy": val_accuracy}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare antibody data for auto-dab")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to CSV/TSV/Parquet with heavy_chain and light_chain columns",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="Fraction of data to hold out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    prepare_data(args.data, val_fraction=args.val_fraction, seed=args.seed)
    print("\nDone! Data cached at:", CACHE_DIR)
