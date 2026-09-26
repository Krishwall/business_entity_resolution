import numpy as np


def compute_entity_f05(y_true: list[str], y_pred: list[str]) -> float:
    # Singletons
    if not y_true and not y_pred:
        return 1.0
    if not y_true and y_pred:
        return 0.0
    if y_true and not y_pred:
        return 0.0

    true_set = set(y_true)
    pred_set = set(y_pred)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    all_s1_ids: list[str],
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> float:
    scores = [
        compute_entity_f05(ground_truth.get(s1_id, []), predictions.get(s1_id, []))
        for s1_id in all_s1_ids
    ]
    return float(np.mean(scores))