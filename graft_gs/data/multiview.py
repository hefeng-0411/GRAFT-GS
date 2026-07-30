"""Folder-based real multiview dataset; no synthetic geometry placeholders."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class FolderMultiviewDataset(Dataset):
    """Each child directory is one static object with two or more RGB views."""

    def __init__(self, root: str | Path, maximum_views: int = 8, require_target_state: bool = False) -> None:
        self.root = Path(root)
        self.maximum_views = maximum_views
        self.require_target_state = require_target_state
        self.objects = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
            target = directory / "target_state.pt"
            if len(images) >= 2 and (not require_target_state or target.exists()):
                self.objects.append((directory, images[:maximum_views], target))
        if not self.objects:
            raise ValueError(f"no valid multiview objects found under {self.root}")

    def __len__(self) -> int:
        return len(self.objects)

    def view_count(self, index: int) -> int:
        """Return the exact joint-attention view count without loading images."""

        return len(self.objects[index][1])

    def __getitem__(self, index: int) -> dict[str, object]:
        directory, paths, target = self.objects[index]
        # Resolve installed packages and explicit server checkouts through the
        # same provenance-aware boundary used by the production adapter.
        from ..integration.external import import_external_module

        load_and_preprocess_images = getattr(
            import_external_module("vggt.utils.load_fn"),
            "load_and_preprocess_images",
        )

        result: dict[str, object] = {
            "object_id": directory.name,
            "images": load_and_preprocess_images([str(path) for path in paths]),
        }
        if self.require_target_state:
            result["target_states"] = [torch.load(target, map_location="cpu", weights_only=False)]
            result["target_state_provenance"] = "explicit_serialized_manifold_target"
            result["target_state_confidence"] = torch.tensor([1.0], dtype=torch.float32)
        return result


def single_object_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Preserve variable topology by enforcing one object per process step."""

    if len(batch) != 1:
        raise ValueError("GRAFT-GS reference training uses batch size one per rank; use gradient accumulation for larger batches")
    item = dict(batch[0])
    item["images"] = item["images"][None]
    return item


def folder_object_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate objects only when their VGGT joint-view dimensions agree."""

    if not batch:
        raise ValueError("cannot collate an empty object batch")
    if len(batch) == 1:
        return single_object_collate(batch)
    view_counts = {int(torch.as_tensor(item["images"]).shape[0]) for item in batch}
    if len(view_counts) != 1:
        raise ValueError(
            "VGGT object batches must have identical view counts; use "
            "ViewCountBatchSampler instead of padding joint-attention inputs"
        )
    result: dict[str, object] = {
        "object_id": [str(item["object_id"]) for item in batch],
        "images": torch.stack(
            [torch.as_tensor(item["images"]) for item in batch], dim=0
        ),
    }
    if any("target_states" in item for item in batch):
        if not all("target_states" in item for item in batch):
            raise ValueError("folder target-state supervision is incomplete in batch")
        result["target_states"] = [
            item["target_states"][0] for item in batch
        ]
        provenance = [item.get("target_state_provenance") for item in batch]
        if len(set(provenance)) != 1:
            raise ValueError("folder target-state provenance differs within batch")
        result["target_state_provenance"] = provenance[0]
        result["target_state_confidence"] = torch.cat(
            [
                torch.as_tensor(item.get("target_state_confidence", [1.0])).reshape(-1)
                for item in batch
            ]
        )
    return result


__all__ = [
    "FolderMultiviewDataset",
    "folder_object_collate",
    "single_object_collate",
]
