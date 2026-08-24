from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from data import DataConfig, infer_max_seq_len, infer_vocab_size, make_dataloaders
from data.binary_modulus import (
    BinaryModulusGenerationConfig,
    TOKEN_IDS,
    generate_binary_modulus_dataset,
    tokenize_binary_modulus,
)


class BinaryModulusTests(unittest.TestCase):
    def test_tokenization_has_only_x_n_and_remainder(self) -> None:
        input_ids, labels = tokenize_binary_modulus(25, modulus=13, bit_width=4)
        self.assertEqual(input_ids[0], TOKEN_IDS["X"])
        self.assertEqual(input_ids[9], TOKEN_IDS["N"])
        self.assertEqual(input_ids[-4:], [TOKEN_IDS["ANS"]] * 4)
        # 25 mod 13 = 12 = 1100 MSB-first, hence 0011 LSB-first.
        self.assertEqual(
            labels,
            [
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_0"],
                TOKEN_IDS["BIT_1"],
                TOKEN_IDS["BIT_1"],
            ],
        )

    def test_generation_separates_modulus_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = generate_binary_modulus_dataset(
                BinaryModulusGenerationConfig(
                    output_dir=directory,
                    bit_width=6,
                    train_examples=40,
                    val_examples=20,
                    train_moduli=10,
                    val_moduli=5,
                    seed=8,
                )
            )
            root = Path(directory)
            train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            val = [json.loads(line) for line in (root / "val.jsonl").read_text().splitlines()]
            train_moduli = {row["modulus"] for row in train}
            val_moduli = {row["modulus"] for row in val}
            self.assertTrue(train_moduli.isdisjoint(val_moduli))
            self.assertTrue(all(row["remainder"] == row["x"] % row["modulus"] for row in train + val))
            self.assertEqual(config["split_counts"], {"train": 40, "val": 20})

    def test_factory_preserves_separate_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_binary_modulus_dataset(
                BinaryModulusGenerationConfig(
                    output_dir=directory,
                    bit_width=5,
                    train_examples=20,
                    val_examples=10,
                    train_moduli=5,
                    val_moduli=3,
                    seed=11,
                )
            )
            config = DataConfig(
                kind="binary_modulus",
                data_root=directory,
                batch_size=5,
                eval_batch_size=10,
                drop_last=True,
                pin_memory=False,
            )
            loaders = make_dataloaders(config, device=torch.device("cpu"))
            batch = next(iter(loaders["val"]))
            self.assertEqual(tuple(batch["input_ids"].shape), (10, 22))
            self.assertEqual(tuple(batch["labels"].shape), (10, 5))
            self.assertTrue(
                torch.equal(batch["target_positions"][0], torch.arange(17, 22))
            )
            self.assertEqual(infer_vocab_size(config), 6)
            self.assertEqual(infer_max_seq_len(config), 22)

    def test_generation_rejects_too_many_moduli(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceed"):
            BinaryModulusGenerationConfig(
                output_dir="unused",
                bit_width=5,
                train_moduli=5,
                val_moduli=4,
            )


if __name__ == "__main__":
    unittest.main()
