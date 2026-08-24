from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import DataConfig, infer_max_seq_len, infer_vocab_size, make_dataloaders
from data.binary_squaring import (
    BinarySquaringGenerationConfig,
    TOKEN_IDS,
    generate_binary_squaring_dataset,
    tokenize_binary_square,
)


class BinarySquaringTests(unittest.TestCase):
    def test_tokenization_is_fixed_width_lsb_first(self) -> None:
        input_ids, labels = tokenize_binary_square(5, bit_width=3)
        self.assertEqual(
            input_ids,
            [
                TOKEN_IDS["X"],
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_1"],
                *([TOKEN_IDS["ANS"]] * 6),
            ],
        )
        # 25 = 011001 when written MSB-first at six-bit width.
        self.assertEqual(
            labels,
            [
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_0"],
            ],
        )

    def test_generation_makes_disjoint_train_and_val_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = generate_binary_squaring_dataset(
                BinarySquaringGenerationConfig(
                    output_dir=directory,
                    bit_width=5,
                    train_examples=20,
                    val_examples=8,
                    seed=19,
                )
            )
            self.assertEqual(
                {path.name for path in root.glob("*.jsonl")},
                {"train.jsonl", "val.jsonl"},
            )
            train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            val = [json.loads(line) for line in (root / "val.jsonl").read_text().splitlines()]
            self.assertEqual({row["value"] for row in train} & {row["value"] for row in val}, set())
            self.assertTrue(all(row["square"] == row["value"] ** 2 for row in train + val))
            self.assertEqual(config["split_counts"], {"train": 20, "val": 8})

    def test_factory_preserves_separate_output_loss_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_binary_squaring_dataset(
                BinarySquaringGenerationConfig(
                    output_dir=directory,
                    bit_width=4,
                    train_examples=10,
                    val_examples=6,
                    seed=7,
                )
            )
            config = DataConfig(
                kind="binary_squaring",
                data_root=directory,
                batch_size=5,
                eval_batch_size=6,
                drop_last=True,
                pin_memory=False,
            )
            loaders = make_dataloaders(config, device=torch.device("cpu"))
            self.assertEqual(set(loaders), {"train", "val"})
            batch = next(iter(loaders["val"]))
            self.assertEqual(tuple(batch["input_ids"].shape), (6, 13))
            self.assertEqual(tuple(batch["labels"].shape), (6, 8))
            self.assertTrue(
                torch.equal(
                    batch["target_positions"][0],
                    torch.arange(5, 13),
                )
            )
            self.assertEqual(infer_vocab_size(config), 5)
            self.assertEqual(infer_max_seq_len(config), 13)

    def test_generation_rejects_more_examples_than_distinct_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            BinarySquaringGenerationConfig(
                output_dir="unused",
                bit_width=3,
                train_examples=7,
                val_examples=2,
            )


if __name__ == "__main__":
    unittest.main()
