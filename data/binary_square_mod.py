"""Fixed-width binary ``x squared mod N`` data for a single transition."""

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
DEFAULT_MAX_SEQ_LEN = 2 + 3 * DEFAULT_BIT_WIDTH


class BinarySquareModTokenizedDataset(TokenizedCountingDataset):
    """JSONL-backed binary one-step modular-squaring dataset."""


@dataclass(frozen=True)
class BinarySquareModGenerationConfig:
    output_dir: str
    bit_width: int = DEFAULT_BIT_WIDTH
    train_examples: int = 40_000
    val_examples: int = 2_000
    test_examples: int = 18_000
    train_moduli: int = 512
    val_moduli: int = 128
    test_moduli: int = 128
    seed: int = 49

    def __post_init__(self) -> None:
        if self.bit_width < 3:
            raise ValueError("bit_width must be at least 3")
        if min(self.train_examples, self.val_examples, self.test_examples) < 1:
            raise ValueError("all split sizes must be positive")
        if min(self.train_moduli, self.val_moduli, self.test_moduli) < 1:
            raise ValueError("all split modulus counts must be positive")
        available = 1 << (self.bit_width - 2)
        requested = self.train_moduli + self.val_moduli + self.test_moduli
        if requested > available:
            raise ValueError(
                "requested modulus identities exceed the number of full-width "
                f"odd {self.bit_width}-bit values ({available})"
            )
        capacity_by_split = (
            ("train", self.train_examples, self.train_moduli),
            ("val", self.val_examples, self.val_moduli),
            ("test", self.test_examples, self.test_moduli),
        )
        x_values = 1 << self.bit_width
        for split, examples, moduli in capacity_by_split:
            if examples > moduli * x_values:
                raise ValueError(f"{split}_examples exceeds distinct (N, x) capacity")


def _bit_tokens(value: int, width: int) -> list[int]:
    if not 0 <= value < (1 << width):
        raise ValueError(f"value must fit in {width} bits")
    return [
        TOKEN_IDS["BIT_1"] if (value >> position) & 1 else TOKEN_IDS["BIT_0"]
        for position in range(width)
    ]


def tokenize_binary_square_mod(
    x: int,
    modulus: int,
    bit_width: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one application of ``x -> x**2 mod modulus``."""

    if not 0 <= x < (1 << bit_width):
        raise ValueError("x must fit in bit_width bits")
    if not (1 << (bit_width - 1)) <= modulus < (1 << bit_width):
        raise ValueError("modulus must be full-width")
    if modulus % 2 == 0:
        raise ValueError("modulus must be odd")
    input_ids = [TOKEN_IDS["X"]]
    input_ids.extend(_bit_tokens(x, bit_width))
    input_ids.append(TOKEN_IDS["N"])
    input_ids.extend(_bit_tokens(modulus, bit_width))
    input_ids.extend([TOKEN_IDS["ANS"]] * bit_width)
    return input_ids, _bit_tokens((x * x) % modulus, bit_width)


def collate_binary_square_mod(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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
        row_input = torch.tensor(item["input_ids"], dtype=torch.long)
        row_labels = torch.tensor(item["labels"], dtype=torch.long)
        input_len = row_input.numel()
        target_len = row_labels.numel()
        input_ids[row, :input_len] = row_input
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


def load_binary_square_mod_dataset_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing binary square-mod dataset config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_binary_square_mod_dataset(
    config: BinarySquareModGenerationConfig,
) -> dict[str, Any]:
    """Generate disjoint-modulus train, validation, and untouched test splits."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    modulus_candidates = list(
        range((1 << (config.bit_width - 1)) + 1, 1 << config.bit_width, 2)
    )
    rng.shuffle(modulus_candidates)
    train_stop = config.train_moduli
    val_stop = train_stop + config.val_moduli
    test_stop = val_stop + config.test_moduli
    pools = {
        "train": modulus_candidates[:train_stop],
        "val": modulus_candidates[train_stop:val_stop],
        "test": modulus_candidates[val_stop:test_stop],
    }
    sizes = {
        "train": config.train_examples,
        "val": config.val_examples,
        "test": config.test_examples,
    }
    records = []
    for split in ("train", "val", "test"):
        seen: set[tuple[int, int]] = set()
        pool = pools[split]
        while len(seen) < sizes[split]:
            index = len(seen)
            modulus = pool[index % len(pool)]
            x = rng.randrange(1 << config.bit_width)
            key = (modulus, x)
            if key in seen:
                continue
            seen.add(key)
            input_ids, labels = tokenize_binary_square_mod(
                x,
                modulus,
                config.bit_width,
            )
            records.append(
                {
                    "split": split,
                    "x": x,
                    "modulus": modulus,
                    "result": (x * x) % modulus,
                    "time_steps": 1,
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    write_split_files(output_dir, records)
    dataset_config = {
        "dataset_kind": "binary_square_mod",
        "generator_config": asdict(config),
        "token_ids": TOKEN_IDS,
        "vocab_size": VOCAB_SIZE,
        "bit_width": config.bit_width,
        "max_seq_len": 2 + 3 * config.bit_width,
        "num_examples": sum(sizes.values()),
        "split_counts": sizes,
        "split_modulus_counts": {
            "train": config.train_moduli,
            "val": config.val_moduli,
            "test": config.test_moduli,
        },
        "time_steps": 1,
        "data_format": "separate_input_output",
        "label_format": "fixed_width_lsb_first_binary_square_mod",
    }
    write_dataset_config(output_dir, dataset_config)
    return dataset_config


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bit-width", type=int, default=DEFAULT_BIT_WIDTH)
    parser.add_argument("--train-examples", type=int, default=40_000)
    parser.add_argument("--val-examples", type=int, default=2_000)
    parser.add_argument("--test-examples", type=int, default=18_000)
    parser.add_argument("--train-moduli", type=int, default=512)
    parser.add_argument("--val-moduli", type=int, default=128)
    parser.add_argument("--test-moduli", type=int, default=128)
    parser.add_argument("--seed", type=int, default=49)
    args = parser.parse_args()
    result = generate_binary_square_mod_dataset(
        BinarySquareModGenerationConfig(
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
