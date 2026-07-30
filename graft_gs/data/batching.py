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


def _estimated_work(dataset: ViewCountDataset, index: int) -> float:
    """Return a deterministic positive object-cost proxy."""

    estimator = getattr(dataset, "estimated_work", None)
    value = (
        float(estimator(int(index)))
        if callable(estimator)
        else float(dataset.view_count(int(index)))
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"dataset item {index} has an invalid estimated work value {value}"
        )
    return value


def _cost_ordered(
    dataset: ViewCountDataset,
    indices: Sequence[int],
    *,
    generator: random.Random,
    shuffle_ties: bool,
) -> list[int]:
    decorated = [
        (
            _estimated_work(dataset, int(index)),
            generator.random() if shuffle_ties else float(int(index)),
            int(index),
        )
        for index in indices
    ]
    decorated.sort()
    return [index for _, _, index in decorated]


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


def cost_balanced_distributed_indices(
    dataset: ViewCountDataset,
    *,
    num_replicas: int,
    rank: int,
) -> tuple[int, ...]:
    """Shard independent evaluation objects by deterministic greedy cost."""

    if num_replicas < 1 or not 0 <= rank < num_replicas:
        raise ValueError("invalid distributed sampler rank/world size")
    shards: list[list[int]] = [[] for _ in range(num_replicas)]
    costs = [0.0 for _ in range(num_replicas)]
    descending = sorted(
        range(len(dataset)),
        key=lambda index: (_estimated_work(dataset, index), -index),
        reverse=True,
    )
    for index in descending:
        selected = min(
            range(num_replicas),
            key=lambda rank_value: (
                costs[rank_value],
                len(shards[rank_value]),
                rank_value,
            ),
        )
        shards[selected].append(index)
        costs[selected] += _estimated_work(dataset, index)
    return tuple(shards[rank])


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
        largest_first: bool = False,
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
        self.largest_first = bool(largest_first)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = _group_indices_by_view_count(self.dataset, self.indices)
        full_batches: list[tuple[float, list[int]]] = []
        partial_batches: list[tuple[float, list[int]]] = []
        for view_count in sorted(groups):
            values = _cost_ordered(
                self.dataset,
                groups[view_count],
                generator=generator,
                shuffle_ties=self.shuffle,
            )
            batch_count = math.ceil(len(values) / self.batch_size)
            balanced_values: list[list[int]] = [
                [] for _ in range(batch_count)
            ]
            balanced_cost = [0.0 for _ in range(batch_count)]
            for index in reversed(values):
                available = [
                    batch_index
                    for batch_index in range(batch_count)
                    if len(balanced_values[batch_index]) < self.batch_size
                ]
                selected_batch = min(
                    available,
                    key=lambda batch_index: (
                        balanced_cost[batch_index],
                        len(balanced_values[batch_index]),
                        batch_index,
                    ),
                )
                balanced_values[selected_batch].append(index)
                balanced_cost[selected_batch] += _estimated_work(
                    self.dataset, index
                )
            for cost, batch in zip(balanced_cost, balanced_values):
                (
                    full_batches
                    if len(batch) == self.batch_size
                    else partial_batches
                ).append((cost, batch))
        if self.shuffle:
            generator.shuffle(full_batches)
            generator.shuffle(partial_batches)
        if self.largest_first:
            full_batches.sort(key=lambda item: item[0], reverse=True)
            partial_batches.sort(key=lambda item: item[0], reverse=True)
        batches = full_batches + partial_batches
        yield from (batch for _, batch in batches)

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
        largest_first: bool = False,
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
        self.largest_first = bool(largest_first)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_batches(self) -> list[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        groups = _group_indices_by_view_count(
            self.dataset, tuple(range(len(self.dataset)))
        )
        global_batch_size = self.batch_size * self.num_replicas
        distributed_batches: list[tuple[float, list[list[int]]]] = []
        for view_count in sorted(groups):
            source_values = list(groups[view_count])
            padded_count = (
                math.ceil(len(source_values) / global_batch_size)
                * global_batch_size
            )
            # Choose padding uniformly before cost ordering. This retains
            # DistributedSampler's expected repeat policy without always
            # duplicating one complexity tail.
            padding_source = list(source_values)
            if self.shuffle:
                generator.shuffle(padding_source)
            values = source_values + [
                padding_source[index % len(padding_source)]
                for index in range(padded_count - len(source_values))
            ]
            values = _cost_ordered(
                self.dataset,
                values,
                generator=generator,
                shuffle_ties=self.shuffle,
            )
            # First build balanced local object batches over the entire exact-
            # view bucket. With batch size one these are individual objects;
            # with larger batches, LPT-style placement pairs heavy and light
            # objects so tail memory and runtime do not stack on one rank.
            local_batch_count = padded_count // self.batch_size
            local_values: list[list[int]] = [
                [] for _ in range(local_batch_count)
            ]
            local_cost = [0.0 for _ in range(local_batch_count)]
            for index in reversed(values):
                available = [
                    batch_index
                    for batch_index in range(local_batch_count)
                    if len(local_values[batch_index]) < self.batch_size
                ]
                selected_batch = min(
                    available,
                    key=lambda batch_index: (
                        local_cost[batch_index],
                        len(local_values[batch_index]),
                        batch_index,
                    ),
                )
                local_values[selected_batch].append(index)
                local_cost[selected_batch] += _estimated_work(
                    self.dataset, index
                )
            ordered_local = sorted(
                zip(local_cost, local_values),
                key=lambda item: item[0],
            )
            # Adjacent balanced local batches form one global cohort. The
            # rank permutation changes by epoch, but every process constructs
            # the same mapping and collective count.
            for start in range(0, local_batch_count, self.num_replicas):
                cohort = ordered_local[start : start + self.num_replicas]
                rank_priority = list(range(self.num_replicas))
                if self.shuffle:
                    generator.shuffle(rank_priority)
                rank_values: list[list[int]] = [
                    [] for _ in range(self.num_replicas)
                ]
                for offset, rank_value in enumerate(rank_priority):
                    rank_values[rank_value] = cohort[offset][1]
                distributed_batches.append(
                    (sum(cost for cost, _ in cohort), rank_values)
                )
        if self.shuffle:
            generator.shuffle(distributed_batches)
        if self.largest_first:
            distributed_batches.sort(key=lambda item: item[0], reverse=True)
        return [
            rank_values[self.rank]
            for _, rank_values in distributed_batches
        ]

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
    "cost_balanced_distributed_indices",
]
