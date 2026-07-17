"""Folder-based real multiview dataset; no synthetic geometry placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

    def __getitem__(self, index: int) -> dict[str, object]:
        directory, paths, target = self.objects[index]
        from vggt.utils.load_fn import load_and_preprocess_images

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


__all__ = ["FolderMultiviewDataset", "single_object_collate"]
