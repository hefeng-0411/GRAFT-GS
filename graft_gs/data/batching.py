"""View-homogeneous object batching for variable-topology scenes."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterator, Protocol, Sequence

from torch.utils.data import Sampler


class ViewCountDataset(Protocol):
    def __len__(self) -> int: ...

    def view_count(self, index: int) -> int: ...


def _group_indices_by_view_count(
    dataset: ViewCountDataset,
    indices: Sequence[int],
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        count = int(dataset.view_count(int(index)))
        if count < 1:
            raise ValueError(f"dataset item {index} has a non-positive view count")
        groups[count].append(int(index))
    return dict(groups)


class ViewCountBatchSampler(Sampler[list[int]]):
    """Batch local indices without ever padding the VGGT view dimension."""

    def __init__(
        self,
        dataset: ViewCountDataset,
        batch_size: int,
        *,
        indices: Sequence[int] | None = None,
        shuffle: bool = False,
        seed: int = 17,
    ) -> None:
        if batch_size < 1:
            raise ValueError("object batch size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.indices = tuple(
            range(len(dataset)) if indices is None else (int(value) for value in indices)
        )
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = _group_indices_by_view_count(self.dataset, self.indices)
        full_batches: list[list[int]] = []
        partial_batches: list[list[int]] = []
        for view_count in sorted(groups):
            values = groups[view_count]
            if self.shuffle:
                generator.shuffle(values)
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                (
                    full_batches
                    if len(batch) == self.batch_size
                    else partial_batches
                ).append(batch)
        if self.shuffle:
            generator.shuffle(full_batches)
            generator.shuffle(partial_batches)
        batches = full_batches + partial_batches
        yield from batches

    def __len__(self) -> int:
        groups = _group_indices_by_view_count(self.dataset, self.indices)
        return sum(
            math.ceil(len(values) / self.batch_size) for values in groups.values()
        )


class DistributedViewCountBatchSampler(Sampler[list[int]]):
    """Build identical-shape DDP batches while sharding objects across ranks.

    Every view-count bucket is padded deterministically to a multiple of the
    global object batch. This mirrors ``DistributedSampler``: a few objects can
    repeat at an epoch boundary, but ranks execute the same number of gradient
    collectives and no fake camera/image views enter VGGT.
    """

    def __init__(
        self,
        dataset: ViewCountDataset,
        batch_size: int,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 17,
    ) -> None:
        if batch_size < 1:
            raise ValueError("object batch size must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_batches(self) -> list[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = _group_indices_by_view_count(
            self.dataset, tuple(range(len(self.dataset)))
        )
        global_batch_size = self.batch_size * self.num_replicas
        rank_batches: list[list[int]] = []
        for view_count in sorted(groups):
            values = groups[view_count]
            if self.shuffle:
                generator.shuffle(values)
            padded_count = (
                math.ceil(len(values) / global_batch_size) * global_batch_size
            )
            values = values + [
                values[index % len(values)]
                for index in range(padded_count - len(values))
            ]
            for start in range(0, padded_count, global_batch_size):
                global_batch = values[start : start + global_batch_size]
                rank_start = self.rank * self.batch_size
                rank_batches.append(
                    global_batch[rank_start : rank_start + self.batch_size]
                )
        if self.shuffle:
            generator.shuffle(rank_batches)
        return rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches()

    def __len__(self) -> int:
        groups = _group_indices_by_view_count(
            self.dataset, tuple(range(len(self.dataset)))
        )
        global_batch_size = self.batch_size * self.num_replicas
        return sum(
            math.ceil(len(values) / global_batch_size)
            for values in groups.values()
        )


__all__ = [
    "DistributedViewCountBatchSampler",
    "ViewCountBatchSampler",
    "ViewCountDataset",
]
