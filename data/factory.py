"""Factories for datasets and dataloaders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from .config import DataConfig


def _make_tokenized_counting_dataloaders(
    *,
    config: DataConfig,
    root: str | Path,
    dataset_class,
    collate_fn,
    device: torch.device | None = None,
) -> dict[str, DataLoader]:
    train_dataset = dataset_class(root, "train")
    extra_datasets = {}
    root_path = Path(root)
    for split_path in sorted(root_path.glob("*.jsonl")):
        split = split_path.stem
        if split in ("train", "eval"):
            continue
        candidate_dataset = dataset_class(root, split)
        if len(candidate_dataset) > 0:
            extra_datasets[split] = candidate_dataset

    train_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    pin_memory_device = device or torch.get_default_device()
    pin_memory = config.pin_memory and pin_memory_device.type == "cuda"
    eval_batch_size = config.eval_batch_size or config.batch_size

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            drop_last=config.drop_last,
            generator=train_generator,
        ),
    }
    for split, dataset in extra_datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=config.shuffle_eval,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return dataloaders


def _make_squaring_mod_dataloaders(
    config: DataConfig, device: torch.device | None = None
) -> dict[str, DataLoader]:
    from .squaring_mod import (
        SquaringModTokenizedDataset,
        collate_squaring_mod,
        generate_squaring_mod_smoke_dataset,
    )

    if config.data_root is None:
        # The JSONL-backed datasets eagerly load their records, so the temporary
        # files can be removed after constructing the dataloaders.
        with tempfile.TemporaryDirectory(prefix="squaring-mod-smoke-") as root:
            generate_squaring_mod_smoke_dataset(root, seed=config.seed)
            return _make_tokenized_counting_dataloaders(
                config=config,
                root=root,
                dataset_class=SquaringModTokenizedDataset,
                collate_fn=collate_squaring_mod,
                device=device,
            )
    return _make_tokenized_counting_dataloaders(
        config=config,
        root=config.data_root,
        dataset_class=SquaringModTokenizedDataset,
        collate_fn=collate_squaring_mod,
        device=device,
    )


def _make_binary_squaring_dataloaders(
    config: DataConfig, device: torch.device | None = None
) -> dict[str, DataLoader]:
    from .binary_squaring import (
        BinarySquaringTokenizedDataset,
        collate_binary_squaring,
    )

    if config.data_root is None:
        raise ValueError("binary_squaring requires a generated data_root")
    return _make_tokenized_counting_dataloaders(
        config=config,
        root=config.data_root,
        dataset_class=BinarySquaringTokenizedDataset,
        collate_fn=collate_binary_squaring,
        device=device,
    )


def make_dataloaders(
    config: DataConfig, device: torch.device | None = None
) -> Mapping[str, DataLoader]:
    if config.kind == "binary_squaring":
        return _make_binary_squaring_dataloaders(config, device=device)
    return _make_squaring_mod_dataloaders(config, device=device)
