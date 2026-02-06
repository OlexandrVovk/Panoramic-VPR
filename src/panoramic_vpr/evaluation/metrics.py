"""Recall@K metrics for VPR evaluation."""


def compute_recall_at_k(
    predictions: list[list[int]],
    ground_truth: dict[int, list[int]],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute Recall@K for visual place recognition.

    For each query with ground truth, checks if ANY correct panorama
    appears in the top-K predictions.

    Args:
        predictions: For each query index, a ranked list of predicted pano indices.
        ground_truth: query_idx -> list of correct pano indices.
        ks: List of K values to compute recall for (default: [1, 5, 10]).

    Returns:
        Dict like {"R@1": 0.85, "R@5": 0.93, "R@10": 0.96, "num_queries": 250}.
    """
    if ks is None:
        ks = [1, 5, 10]

    correct_at_k = {k: 0 for k in ks}
    num_evaluated = 0

    for qi, preds in enumerate(predictions):
        if qi not in ground_truth:
            continue
        positives = set(ground_truth[qi])
        num_evaluated += 1

        for k in ks:
            top_k_preds = set(preds[:k])
            if top_k_preds & positives:
                correct_at_k[k] += 1

    results = {}
    for k in ks:
        results[f"R@{k}"] = correct_at_k[k] / max(num_evaluated, 1)
    results["num_queries"] = num_evaluated
    return results
