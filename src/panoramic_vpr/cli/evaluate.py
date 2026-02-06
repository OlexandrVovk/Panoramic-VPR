"""CLI: Batch evaluation with Recall@K metrics."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate panoramic VPR retrieval with Recall@K."
    )
    parser.add_argument("--dataset", required=True, choices=["pitts250k", "yq360"],
                        help="Dataset name.")
    parser.add_argument("--split", default="test",
                        help="Dataset split (default: test).")
    parser.add_argument("--data-root", default="data/dataset",
                        help="Root directory containing dataset folders.")
    parser.add_argument("--database", required=True,
                        help="Path to .pt database file.")
    parser.add_argument("--top-k-coarse", type=int, default=5,
                        help="Number of coarse candidates (default: 5).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Query encoding batch size (default: 32).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda).")
    parser.add_argument("--recall-ks", default="1,5,10",
                        help="Comma-separated K values for Recall@K (default: 1,5,10).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ..backbone.megaloc import MegaLocBackbone
    from ..config import PipelineConfig
    from ..datasets.registry import get_dataset
    from ..evaluation.evaluator import Evaluator

    recall_ks = [int(k) for k in args.recall_ks.split(",")]

    config = PipelineConfig(
        top_k_coarse=args.top_k_coarse,
        batch_size=args.batch_size,
        device=args.device,
        recall_ks=recall_ks,
    )

    data_root = Path(args.data_root)
    if args.dataset == "pitts250k":
        dataset_root = data_root / "pitts250k" / "pitts250k"
    else:
        dataset_root = data_root / args.dataset

    dataset = get_dataset(args.dataset, dataset_root, args.split)

    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"  Panoramas: {len(dataset.get_database_paths())}")
    print(f"  Queries: {len(dataset.get_query_paths())}")
    print(f"  Top-K coarse: {config.top_k_coarse}")
    print(f"  Recall@K: {recall_ks}")

    backbone = MegaLocBackbone(device=config.device)
    evaluator = Evaluator(config, backbone)

    results = evaluator.evaluate(dataset, Path(args.database))

    print("\n" + "=" * 40)
    print("Results:")
    print("=" * 40)
    for k in recall_ks:
        key = f"R@{k}"
        print(f"  {key}: {results[key]:.4f} ({results[key]*100:.1f}%)")
    print(f"  Queries evaluated: {results['num_queries']}")
    print(f"  Total retrieval time: {results['retrieval_time_s']:.3f}s")
    print(f"  Avg per query: {results['avg_time_per_query_ms']:.2f}ms")


if __name__ == "__main__":
    main()
