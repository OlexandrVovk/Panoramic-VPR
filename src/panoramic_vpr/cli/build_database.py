"""CLI: Build hyperbolic descriptor database from panoramas."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hyperbolic descriptor database from panoramic images."
    )
    parser.add_argument("--dataset", required=True, choices=["pitts250k", "yq360"],
                        help="Dataset name.")
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--data-root", default="data/dataset",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--output", required=True,
                        help="Output path for the database .pt file.")
    parser.add_argument("--num-views", type=int, default=8,
                        help="Number of perspective views per panorama (default: 8).")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="Field of view in degrees (default: 90).")
    parser.add_argument("--aggregation", default="standard",
                        choices=["standard", "weighted"],
                        help="Einstein midpoint variant (default: standard).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Backbone batch size (default: 32).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ..backbone.megaloc import MegaLocBackbone
    from ..config import PipelineConfig
    from ..database.builder import DatabaseBuilder
    from ..database.store import save_database
    from ..datasets.registry import get_dataset

    config = PipelineConfig(
        num_views=args.num_views,
        fov=args.fov,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Resolve dataset root
    data_root = Path(args.data_root)
    if args.dataset == "pitts250k":
        dataset_root = data_root / "pitts250k" / "pitts250k"
    else:
        dataset_root = data_root / args.dataset

    dataset = get_dataset(args.dataset, dataset_root, args.split)

    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"  Panoramas: {len(dataset.get_database_paths())}")
    print(f"  Views per pano: {config.num_views}, FOV: {config.fov}")
    print(f"  Aggregation: {config.aggregation}")
    print(f"  Device: {config.device}")

    backbone = MegaLocBackbone(device=config.device)
    builder = DatabaseBuilder(config, backbone)

    print("\nBuilding database...")
    database = builder.build(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_database(database, output_path)

    n_panos = database["root_descriptors"].shape[0]
    print(f"\nDone! Saved {n_panos} panorama descriptors to {output_path}")


if __name__ == "__main__":
    main()
