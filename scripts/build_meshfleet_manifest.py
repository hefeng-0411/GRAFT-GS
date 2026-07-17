"""Build the physical-file-verified MeshFleet/TRELLIS object manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_manifest_builder():
    source = Path(__file__).resolve().parents[1] / "graft_gs" / "data" / "meshfleet.py"
    spec = importlib.util.spec_from_file_location("graft_gs_meshfleet_manifest", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load manifest implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_meshfleet_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "test"))
    parser.add_argument(
        "--skip-image-headers",
        action="store_true",
        help="inventory physical frames without opening image headers",
    )
    parser.add_argument(
        "--object-id",
        action="append",
        dest="object_ids",
        help="audit only these object IDs; repeat for multiple objects",
    )
    args = parser.parse_args()
    summary = _load_manifest_builder()(
        args.dataset_root,
        args.output,
        splits=args.splits,
        inspect_image_headers=not args.skip_image_headers,
        object_ids=args.object_ids,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
