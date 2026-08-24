"""Data loading configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DatasetKind = Literal["squaring_mod", "binary_squaring", "binary_modulus"]


@dataclass(frozen=True)
class DataConfig:
    kind: DatasetKind = "squaring_mod"
    data_root: str | None = None
    batch_size: int = 2_500
    eval_batch_size: int | None = None
    shuffle_train: bool = True
    shuffle_eval: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    seed: int = 45

    def __post_init__(self) -> None:
        if self.kind not in ("squaring_mod", "binary_squaring", "binary_modulus"):
            raise ValueError(
                "data kind must be squaring_mod, binary_squaring, or binary_modulus"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.eval_batch_size is not None and self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive when provided")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def infer_vocab_size(config: DataConfig) -> int:
    if config.kind == "binary_modulus":
        from .binary_modulus import VOCAB_SIZE, load_binary_modulus_dataset_config

        if config.data_root is None:
            return VOCAB_SIZE
        return int(load_binary_modulus_dataset_config(config.data_root)["vocab_size"])
    if config.kind == "binary_squaring":
        from .binary_squaring import (
            VOCAB_SIZE,
            load_binary_squaring_dataset_config,
        )

        if config.data_root is None:
            return VOCAB_SIZE
        return int(load_binary_squaring_dataset_config(config.data_root)["vocab_size"])

    from .squaring_mod import VOCAB_SIZE, load_squaring_mod_dataset_config

    if config.data_root is None:
        return VOCAB_SIZE
    return int(load_squaring_mod_dataset_config(config.data_root)["vocab_size"])


def infer_max_seq_len(config: DataConfig) -> int:
    if config.kind == "binary_modulus":
        from .binary_modulus import (
            DEFAULT_MAX_SEQ_LEN,
            load_binary_modulus_dataset_config,
        )

        if config.data_root is None:
            return DEFAULT_MAX_SEQ_LEN
        return int(load_binary_modulus_dataset_config(config.data_root)["max_seq_len"])
    if config.kind == "binary_squaring":
        from .binary_squaring import (
            DEFAULT_MAX_SEQ_LEN,
            load_binary_squaring_dataset_config,
        )

        if config.data_root is None:
            return DEFAULT_MAX_SEQ_LEN
        return int(
            load_binary_squaring_dataset_config(config.data_root)["max_seq_len"]
        )

    from .squaring_mod import SMOKE_MAX_SEQ_LEN, load_squaring_mod_dataset_config

    if config.data_root is None:
        return SMOKE_MAX_SEQ_LEN
    return int(load_squaring_mod_dataset_config(config.data_root)["max_seq_len"])
