from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import DataConfig, infer_max_seq_len, infer_vocab_size, make_dataloaders
from data.binary_square_mod import (
    BinarySquareModGenerationConfig,
    TOKEN_IDS,
    generate_binary_square_mod_dataset,
    tokenize_binary_square_mod,
)


class BinarySquareModTests(unittest.TestCase):
    def test_tokenization_is_direct_t1_binary_transition(self) -> None:
        input_ids, labels = tokenize_binary_square_mod(5, modulus=13, bit_width=4)
        self.assertEqual(input_ids[0], TOKEN_IDS["X"])
        self.assertEqual(input_ids[5], TOKEN_IDS["N"])
        self.assertEqual(input_ids[-4:], [TOKEN_IDS["ANS"]] * 4)
        # 5**2 mod 13 = 12, represented least-significant-bit first.
        self.assertEqual(
            labels,
            [
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_1"],
            ],
        )

    def test_generation_and_factory_keep_test_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = generate_binary_square_mod_dataset(
                BinarySquareModGenerationConfig(
                    output_dir=directory,
                    bit_width=4,
                    train_examples=20,
                    val_examples=10,
                    test_examples=10,
                    train_moduli=2,
                    val_moduli=1,
                    test_moduli=1,
                    seed=9,
                )
            )
            root = Path(directory)
            rows = {
                split: [
                    json.loads(line)
                    for line in (root / f"{split}.jsonl").read_text().splitlines()
                ]
                for split in ("train", "val", "test")
            }
            modulus_sets = {
                split: {row["modulus"] for row in split_rows}
                for split, split_rows in rows.items()
            }
            self.assertTrue(modulus_sets["train"].isdisjoint(modulus_sets["val"]))
            self.assertTrue(modulus_sets["train"].isdisjoint(modulus_sets["test"]))
            self.assertTrue(modulus_sets["val"].isdisjoint(modulus_sets["test"]))
            self.assertTrue(
                all(
                    row["result"] == row["x"] * row["x"] % row["modulus"]
                    for split_rows in rows.values()
                    for row in split_rows
                )
            )
            self.assertEqual(
                config["split_counts"],
                {"train": 20, "val": 10, "test": 10},
            )

            data_config = DataConfig(
                kind="binary_square_mod",
                data_root=directory,
                batch_size=5,
                eval_batch_size=10,
                pin_memory=False,
            )
            loaders = make_dataloaders(data_config, device=torch.device("cpu"))
            self.assertEqual(set(loaders), {"train", "val", "test"})
            batch = next(iter(loaders["val"]))
            self.assertEqual(tuple(batch["input_ids"].shape), (10, 14))
            self.assertEqual(tuple(batch["labels"].shape), (10, 4))
            self.assertEqual(infer_vocab_size(data_config), 6)
            self.assertEqual(infer_max_seq_len(data_config), 14)


if __name__ == "__main__":
    unittest.main()
