import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from features import FEATURE_NAMES, extract_features_parallel


def evaluate_f05_fast(
    all_s1: list[str],
    gt: dict[str, list[str]],
    pair_ids: list[tuple[str, str]],
    probs: np.ndarray,
    threshold: float,
) -> float:
    preds = {s1: [] for s1 in all_s1}
    for idx, (s1, cand) in enumerate(pair_ids):
        if probs[idx] >= threshold:
            preds[s1].append(cand)

    scores = []
    for s1 in all_s1:
        y_true = gt.get(s1, [])
        y_pred = preds.get(s1, [])

        if not y_true and not y_pred:
            scores.append(1.0)
            continue
        if (not y_true and y_pred) or (y_true and not y_pred):
            scores.append(0.0)
            continue

        true_set = set(y_true)
        pred_set = set(y_pred)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 0.25 * p + r
        scores.append((1.25 * p * r) / denom if denom > 0 else 0.0)

    return float(np.mean(scores))


def train_lightgbm(
    train_cands: dict[str, list[str]],
    gt: dict[str, list[str]],
    names: dict[str, str],
    addrs: dict[str, str],
    nums: dict[str, set[str]],
    negative_ratio: int = 5,
) -> tuple[lgb.Booster, float]:
    """Subsamples negatives, trains LightGBM, and tunes decision threshold."""
    random.seed(42)
    training_pairs = []
    labels = []

    print("Building subsampled training pair set...")
    for s1_id, cands in train_cands.items():
        true_matches = set(gt.get(s1_id, []))
        negatives = [c for c in cands if c not in true_matches]

        # Include all positives
        for m in true_matches:
            if m in names:  # ensure it exists in pool
                training_pairs.append((s1_id, m))
                labels.append(1)

        # Subsample negatives to prevent memory explosion
        sample_k = min(len(negatives), max(len(true_matches) * negative_ratio, 3))
        sampled_negatives = random.sample(negatives, sample_k)
        for neg in sampled_negatives:
            training_pairs.append((s1_id, neg))
            labels.append(0)

    print(
        f"Total balanced training pairs: {len(training_pairs)} ({sum(labels)} positives)"
    )

    pairs, X = extract_features_parallel(training_pairs, names, addrs, nums)
    y = np.array(labels, dtype=np.int32)

    # Stratified validation split by S1 ID
    unique_s1 = list(train_cands.keys())
    random.shuffle(unique_s1)
    val_split = int(0.20 * len(unique_s1))
    val_s1_set = set(unique_s1[:val_split])

    train_idx = [i for i, (s1, _) in enumerate(pairs) if s1 not in val_s1_set]
    val_idx = [i for i, (s1, _) in enumerate(pairs) if s1 in val_s1_set]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    val_pairs = [pairs[i] for i in val_idx]

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": -1,
    }

    evals_result = {}
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    # Tune threshold specifically for macro F_0.5 on validation entities
    val_probs = booster.predict(X_val)
    val_s1_list = list(val_s1_set)

    best_thresh = 0.50
    best_score = -1.0
    for t in np.linspace(0.40, 0.90, 51):
        score = evaluate_f05_fast(val_s1_list, gt, val_pairs, val_probs, t)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(
        f"Validation Complete | Best Threshold: {best_thresh:.3f} | Macro F_0.5: {best_score:.4f}"
    )
    return booster, best_thresh