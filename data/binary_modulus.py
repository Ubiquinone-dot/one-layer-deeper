"""Fixed-width binary modulus data for isolated ModBlock experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch

from .counting import TokenizedCountingDataset, write_dataset_config, write_split_files


TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "BIT_0": 1,
    "BIT_1": 2,
    "X": 3,
    "N": 4,
    "ANS": 5,
}
VOCAB_SIZE = len(TOKEN_IDS)
DEFAULT_BIT_WIDTH = 16
DEFAULT_TRAIN_EXAMPLES = 50_000
DEFAULT_VAL_EXAMPLES = 10_000
DEFAULT_MAX_SEQ_LEN = 2 + 4 * DEFAULT_BIT_WIDTH


class BinaryModulusTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed fixed-width binary remainder dataset."""


@dataclass(frozen=True)
class BinaryModulusGenerationConfig:
    output_dir: str
    bit_width: int = DEFAULT_BIT_WIDTH
    train_examples: int = DEFAULT_TRAIN_EXAMPLES
    val_examples: int = DEFAULT_VAL_EXAMPLES
    test_examples: int = 0
    train_moduli: int = 512
    val_moduli: int = 128
    test_moduli: int = 0
    seed: int = 46

    def __post_init__(self) -> None:
        if self.bit_width < 3:
            raise ValueError("bit_width must be at least 3")
        if self.train_examples < 1 or self.val_examples < 1:
            raise ValueError("train_examples and val_examples must be positive")
        if self.test_examples < 0:
            raise ValueError("test_examples must be nonnegative")
        if self.train_moduli < 1 or self.val_moduli < 1:
            raise ValueError("train_moduli and val_moduli must be positive")
        if self.test_moduli < 0:
            raise ValueError("test_moduli must be nonnegative")
        if (self.test_examples == 0) != (self.test_moduli == 0):
            raise ValueError(
                "test_examples and test_moduli must either both be zero or both positive"
            )
        available = 1 << (self.bit_width - 2)
        requested_moduli = self.train_moduli + self.val_moduli + self.test_moduli
        if requested_moduli > available:
            raise ValueError(
                "requested modulus identities exceed the number of full-width "
                f"odd {self.bit_width}-bit values ({available})"
            )


def _bit_tokens(value: int, width: int) -> list[int]:
    """Return fixed-width least-significant-bit-first token IDs."""

    if not 0 <= value < (1 << width):
        raise ValueError(f"value must fit in {width} bits")
    return [
        TOKEN_IDS["BIT_1"] if (value >> position) & 1 else TOKEN_IDS["BIT_0"]
        for position in range(width)
    ]


def tokenize_binary_modulus(
    x: int,
    modulus: int,
    bit_width: int,
) -> tuple[list[int], list[int]]:
    """Tokenize ``x mod modulus`` with a double-width value."""

    if not (1 << (bit_width - 1)) <= modulus < (1 << bit_width):
        raise ValueError("modulus must be full-width")
    if modulus % 2 == 0:
        raise ValueError("modulus must be odd")
    input_ids = [TOKEN_IDS["X"]]
    input_ids.extend(_bit_tokens(x, 2 * bit_width))
    input_ids.append(TOKEN_IDS["N"])
    input_ids.extend(_bit_tokens(modulus, bit_width))
    input_ids.extend([TOKEN_IDS["ANS"]] * bit_width)
    return input_ids, _bit_tokens(x % modulus, bit_width)


def collate_binary_modulus(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate rows through the benchmark's separate-target-position path."""

    max_input_len = max(len(item["input_ids"]) for item in batch)
    max_target_len = max(len(item["labels"]) for item in batch)
    input_ids = torch.full(
        (len(batch), max_input_len), TOKEN_IDS["PAD"], dtype=torch.long
    )
    labels = torch.full((len(batch), max_target_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.bool)
    target_positions = torch.full(
        (len(batch), max_target_len), -1, dtype=torch.long
    )
    for row, item in enumerate(batch):
        row_input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        row_labels = torch.tensor(item["labels"], dtype=torch.long)
        input_len = row_input_ids.numel()
        target_len = row_labels.numel()
        input_ids[row, :input_len] = row_input_ids
        labels[row, :target_len] = row_labels
        attention_mask[row, :input_len] = True
        target_positions[row, :target_len] = torch.arange(
            input_len - target_len,
            input_len,
            dtype=torch.long,
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
    }


def load_binary_modulus_dataset_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing binary modulus dataset config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_x(
    rng: random.Random,
    modulus: int,
    bit_width: int,
    source: str,
) -> int:
    limit = 1 << (2 * bit_width)
    if source == "uniform":
        return rng.randrange(limit)
    if source == "product":
        return rng.randrange(1 << bit_width) * rng.randrange(1 << bit_width)
    if source == "below_modulus":
        return rng.randrange(modulus)
    if source == "boundary":
        quotient = rng.randrange((limit - 1) // modulus + 1)
        delta = rng.choice((-1, 0, 1))
        return min(limit - 1, max(0, quotient * modulus + delta))
    raise ValueError(f"unknown source: {source}")


def generate_binary_modulus_dataset(
    config: BinaryModulusGenerationConfig,
) -> dict[str, Any]:
    """Generate train/validation rows with disjoint modulus identities."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    modulus_candidates = list(
        range((1 << (config.bit_width - 1)) + 1, 1 << config.bit_width, 2)
    )
    rng.shuffle(modulus_candidates)
    train_pool = modulus_candidates[: config.train_moduli]
    val_pool = modulus_candidates[
        config.train_moduli : config.train_moduli + config.val_moduli
    ]
    test_start = config.train_moduli + config.val_moduli
    test_pool = modulus_candidates[test_start : test_start + config.test_moduli]
    # Keep useful edge cases without allowing the exact-zero remainder from
    # boundary multiples to dominate whole-sequence accuracy.
    sources = (
        "uniform",
        "uniform",
        "uniform",
        "uniform",
        "product",
        "product",
        "product",
        "below_modulus",
        "below_modulus",
        "boundary",
    )

    records = []
    split_specs = [
        ("train", config.train_examples, train_pool),
        ("val", config.val_examples, val_pool),
    ]
    if config.test_examples:
        split_specs.append(("test", config.test_examples, test_pool))
    for split, examples, moduli in split_specs:
        seen: set[tuple[int, int]] = set()
        while len(seen) < examples:
            index = len(seen)
            modulus = moduli[index % len(moduli)]
            # Resample the source on retries. Keeping it tied to ``index`` can
            # deadlock narrow-bit datasets after exhausting the few distinct
            # boundary examples for one modulus.
            source = rng.choice(sources)
            x = _sample_x(rng, modulus, config.bit_width, source)
            key = (modulus, x)
            if key in seen:
                continue
            seen.add(key)
            input_ids, labels = tokenize_binary_modulus(
                x,
                modulus,
                config.bit_width,
            )
            records.append(
                {
                    "split": split,
                    "x": x,
                    "modulus": modulus,
                    "remainder": x % modulus,
                    "source": source,
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    write_split_files(output_dir, records)
    dataset_config = {
        "dataset_kind": "binary_modulus",
        "generator_config": asdict(config),
        "token_ids": TOKEN_IDS,
        "vocab_size": VOCAB_SIZE,
        "bit_width": config.bit_width,
        "x_bit_width": 2 * config.bit_width,
        "max_seq_len": 2 + 4 * config.bit_width,
        "num_examples": (
            config.train_examples + config.val_examples + config.test_examples
        ),
        "split_counts": {
            "train": config.train_examples,
            "val": config.val_examples,
            **({"test": config.test_examples} if config.test_examples else {}),
        },
        "split_modulus_counts": {
            "train": config.train_moduli,
            "val": config.val_moduli,
            **({"test": config.test_moduli} if config.test_moduli else {}),
        },
        "value_sources": sorted(set(sources)),
        "source_weights": {
            source: sources.count(source) / len(sources)
            for source in sorted(set(sources))
        },
        "data_format": "separate_input_output",
        "label_format": "fixed_width_lsb_first_binary_remainder",
    }
    write_dataset_config(output_dir, dataset_config)
    return dataset_config


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bit-width", type=int, default=DEFAULT_BIT_WIDTH)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument("--val-examples", type=int, default=DEFAULT_VAL_EXAMPLES)
    parser.add_argument("--test-examples", type=int, default=0)
    parser.add_argument("--train-moduli", type=int, default=512)
    parser.add_argument("--val-moduli", type=int, default=128)
    parser.add_argument("--test-moduli", type=int, default=0)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()
    result = generate_binary_modulus_dataset(
        BinaryModulusGenerationConfig(
            output_dir=args.output_dir,
            bit_width=args.bit_width,
            train_examples=args.train_examples,
            val_examples=args.val_examples,
            test_examples=args.test_examples,
            train_moduli=args.train_moduli,
            val_moduli=args.val_moduli,
            test_moduli=args.test_moduli,
            seed=args.seed,
        )
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    cli()
