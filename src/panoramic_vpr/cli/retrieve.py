"""CLI: Single-query retrieval against a built database."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve matching panoramas for a query image."
    )
    parser.add_argument("--query", required=True,
                        help="Path to query perspective image.")
    parser.add_argument("--database", required=True,
                        help="Path to .pt database file.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of coarse candidates (default: 5).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import cv2

    from ..backbone.megaloc import MegaLocBackbone
    from ..config import PipelineConfig
    from ..database.store import load_database
    from ..hyperbolic.poincare import exp_map_zero
    from ..retrieval.engine import RetrievalEngine

    config = PipelineConfig(top_k_coarse=args.top_k, device=args.device)

    print("Loading database...")
    database = load_database(args.database)

    print("Loading backbone...")
    backbone = MegaLocBackbone(device=config.device)

    print("Encoding query...")
    query_img = cv2.imread(args.query)
    if query_img is None:
        raise RuntimeError(f"Failed to load query image: {args.query}")
    query_desc = backbone.extract_single(query_img)
    query_hyp = exp_map_zero(query_desc)

    print("Retrieving...")
    engine = RetrievalEngine(database, config)
    results = engine.retrieve(query_hyp, args.top_k)

    # Print results (deduplicated by panorama, showing best view)
    seen_panos = set()
    print(f"\n{'Rank':<6}{'Pano Index':<12}{'Best View':<10}{'Yaw':<8}{'Distance':<12}{'Path'}")
    print("-" * 80)
    rank = 0
    for r in results:
        if r["pano_idx"] in seen_panos:
            continue
        seen_panos.add(r["pano_idx"])
        rank += 1
        path = Path(r["pano_metadata"].get("path", "N/A")).name
        print(f"{rank:<6}{r['pano_idx']:<12}{r['view_idx']:<10}{r['yaw']:<8.1f}"
              f"{r['fine_distance']:<12.4f}{path}")
        if rank >= args.top_k:
            break


if __name__ == "__main__":
    main()
