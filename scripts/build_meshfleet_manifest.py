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
    return module


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
    parser.add_argument(
        "--object-id-file",
        type=Path,
        help="audit the validated 64-hex ID catalog across all selected splits",
    )
    parser.add_argument(
        "--primary-modality",
        action="append",
        dest="primary_modalities",
        help="candidate-ID source modality; repeat to form a union",
    )
    parser.add_argument(
        "--required-modality",
        action="append",
        dest="required_modalities",
        help="modality required for admission to the manifest; repeat as needed",
    )
    parser.add_argument(
        "--optional-modality",
        action="append",
        dest="optional_modalities",
        help="audited modality whose absence does not reject an object",
    )
    args = parser.parse_args()
    module = _load_manifest_builder()
    required_modalities = (
        tuple(args.required_modalities)
        if args.required_modalities
        else module.DEFAULT_REQUIRED_MODALITIES
    )
    optional_modalities = (
        tuple(args.optional_modalities)
        if args.optional_modalities
        else tuple(
            name
            for name in module.MESHFLEET_MODALITIES
            if name not in required_modalities
        )
    )
    summary = module.build_meshfleet_manifest(
        args.dataset_root,
        args.output,
        splits=args.splits,
        inspect_image_headers=not args.skip_image_headers,
        object_ids=args.object_ids,
        object_id_file=args.object_id_file,
        primary_modalities=(
            args.primary_modalities or module.DEFAULT_PRIMARY_MODALITIES
        ),
        required_modalities=(
            required_modalities
        ),
        optional_modalities=(
            optional_modalities
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
