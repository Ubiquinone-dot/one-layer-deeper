"""Fixed-width binary squaring data for isolated SquareBlock experiments."""

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
    "ANS": 4,
}
VOCAB_SIZE = len(TOKEN_IDS)
DEFAULT_BIT_WIDTH = 16
DEFAULT_TRAIN_EXAMPLES = 50_000
DEFAULT_VAL_EXAMPLES = 10_000
DEFAULT_MAX_SEQ_LEN = 1 + 3 * DEFAULT_BIT_WIDTH


class BinarySquaringTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed binary squaring dataset."""


@dataclass(frozen=True)
class BinarySquaringGenerationConfig:
    output_dir: str
    bit_width: int = DEFAULT_BIT_WIDTH
    train_examples: int = DEFAULT_TRAIN_EXAMPLES
    val_examples: int = DEFAULT_VAL_EXAMPLES
    seed: int = 45

    def __post_init__(self) -> None:
        if self.bit_width < 1:
            raise ValueError("bit_width must be positive")
        if self.train_examples < 1:
            raise ValueError("train_examples must be positive")
        if self.val_examples < 1:
            raise ValueError("val_examples must be positive")
        population = 1 << self.bit_width
        if self.train_examples + self.val_examples > population:
            raise ValueError(
                "train_examples + val_examples cannot exceed the number of "
                f"distinct {self.bit_width}-bit values ({population})"
            )


def _bit_tokens(value: int, width: int) -> list[int]:
    """Return a fixed-width, least-significant-bit-first token sequence."""

    if not 0 <= value < (1 << width):
        raise ValueError(f"value must fit in {width} bits")
    return [
        TOKEN_IDS["BIT_1"] if (value >> position) & 1 else TOKEN_IDS["BIT_0"]
        for position in range(width)
    ]


def tokenize_binary_square(value: int, bit_width: int) -> tuple[list[int], list[int]]:
    """Tokenize ``value`` and its square with fixed input/output widths."""

    square_width = 2 * bit_width
    input_ids = [TOKEN_IDS["X"]]
    input_ids.extend(_bit_tokens(value, bit_width))
    # Fixed query slots prevent the prompt length from revealing the square's
    # significant-bit length and provide one model position per target bit.
    input_ids.extend([TOKEN_IDS["ANS"]] * square_width)
    labels = _bit_tokens(value * value, square_width)
    return input_ids, labels


def collate_binary_squaring(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate separate-output rows using the benchmark's target-position path."""

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
        if target_len > input_len:
            raise ValueError("binary square output cannot exceed its query sequence")
        input_ids[row, :input_len] = row_input_ids
        labels[row, :target_len] = row_labels
        attention_mask[row, :input_len] = True
        target_positions[row, :target_len] = torch.arange(
            input_len - target_len, input_len, dtype=torch.long
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_positions": target_positions,
    }


def load_binary_squaring_dataset_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing binary squaring dataset config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_binary_squaring_dataset(
    config: BinarySquaringGenerationConfig,
) -> dict[str, Any]:
    """Generate disjoint train and validation splits without replacement."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    total = config.train_examples + config.val_examples
    values = rng.sample(range(1 << config.bit_width), total)

    records = []
    for index, value in enumerate(values):
        split = "train" if index < config.train_examples else "val"
        input_ids, labels = tokenize_binary_square(value, config.bit_width)
        records.append(
            {
                "split": split,
                "value": value,
                "square": value * value,
                "input_ids": input_ids,
                "labels": labels,
            }
        )

    write_split_files(output_dir, records)
    dataset_config = {
        "dataset_kind": "binary_squaring",
        "generator_config": asdict(config),
        "token_ids": TOKEN_IDS,
        "vocab_size": VOCAB_SIZE,
        "bit_width": config.bit_width,
        "square_bit_width": 2 * config.bit_width,
        "max_seq_len": 1 + 3 * config.bit_width,
        "num_examples": total,
        "split_counts": {
            "train": config.train_examples,
            "val": config.val_examples,
        },
        "data_format": "separate_input_output",
        "label_format": "fixed_width_lsb_first_binary_square",
    }
    write_dataset_config(output_dir, dataset_config)
    return dataset_config


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bit-width", type=int, default=DEFAULT_BIT_WIDTH)
    parser.add_argument(
        "--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES
    )
    parser.add_argument("--val-examples", type=int, default=DEFAULT_VAL_EXAMPLES)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    config = BinarySquaringGenerationConfig(
        output_dir=args.output_dir,
        bit_width=args.bit_width,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        seed=args.seed,
    )
    print(json.dumps(generate_binary_squaring_dataset(config), sort_keys=True))


if __name__ == "__main__":
    cli()
