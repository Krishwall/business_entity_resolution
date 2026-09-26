import csv
import os
import subprocess
from blocking import generate_candidates
from features import extract_features_parallel
import numpy as np
import pandas as pd
from preprocess import load_source_table
from train import train_lightgbm


def load_gt_map(gt_path: str) -> dict[str, list[str]]:
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in df.iterrows():
        s1 = row["source1_entity_id"]
        val = row["matched_entity_ids"].strip()
        gt[s1] = val.split(",") if val else []
    return gt


def main():
    print("[1/5] Loading and cleaning training data...")
    tr_s1_names, tr_s1_addr, tr_s1_country, tr_s1_nums = load_source_table(
        "dataset/train/train_source1.tsv"
    )
    tr_s2_names, tr_s2_addr, tr_s2_country, tr_s2_nums = load_source_table(
        "dataset/train/train_source2.tsv"
    )
    tr_s3_names, tr_s3_addr, tr_s3_country, tr_s3_nums = load_source_table(
        "dataset/train/train_source3.tsv"
    )

    pool_names = {**tr_s2_names, **tr_s3_names}
    pool_addr = {**tr_s2_addr, **tr_s3_addr}
    pool_country = {**tr_s2_country, **tr_s3_country}
    pool_nums = {**tr_s2_nums, **tr_s3_nums}

    all_names = {**tr_s1_names, **pool_names}
    all_addr = {**tr_s1_addr, **pool_addr}
    all_nums = {**tr_s1_nums, **pool_nums}
    all_country = {**tr_s1_country, **pool_country}

    gt = load_gt_map("dataset/train/train_ground_truth.tsv")

    print("[2/5] Running country-partitioned blocking on train set...")
    s1_train_ids = list(tr_s1_names.keys())
    pool_train_ids = list(pool_names.keys())

    train_cands = generate_candidates(
        s1_train_ids,
        pool_train_ids,
        all_names,
        all_country,
        top_k=25,
        threshold=0.25,
    )

    print("[3/5] Subsampling negatives and training LightGBM...")
    model, threshold = train_lightgbm(
        train_cands, gt, all_names, all_addr, all_nums, negative_ratio=5
    )

    # Free training structures
    del tr_s1_names, tr_s2_names, tr_s3_names, pool_names
    del all_names, all_addr, all_nums, all_country, train_cands

    print("[4/5] Loading test sources (US, India, and France)...")
    te_s1_names, te_s1_addr, te_s1_country, te_s1_nums = load_source_table(
        "dataset/test/test_source1.tsv"
    )
    te_s2_names, te_s2_addr, te_s2_country, te_s2_nums = load_source_table(
        "dataset/test/test_source2.tsv"
    )
    te_s3_names, te_s3_addr, te_s3_country, te_s3_nums = load_source_table(
        "dataset/test/test_source3.tsv"
    )

    te_pool_names = {**te_s2_names, **te_s3_names}
    te_pool_addr = {**te_s2_addr, **te_s3_addr}
    te_pool_country = {**te_s2_country, **te_s3_country}
    te_pool_nums = {**te_s2_nums, **te_s3_nums}

    te_all_names = {**te_s1_names, **te_pool_names}
    te_all_addr = {**te_s1_addr, **te_pool_addr}
    te_all_nums = {**te_s1_nums, **te_pool_nums}
    te_all_country = {**te_s1_country, **te_pool_country}

    os.makedirs("output", exist_ok=True)
    cand_file = "output/candidate_pairs.tsv"
    match_file = "output/matching_results.tsv"

    # Initialize TSV files with headers
    with open(cand_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerow(
            ["source1_entity_id", "candidate_entity_ids"]
        )
    with open(match_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerow(
            ["source1_entity_id", "matched_entity_ids"]
        )

    print("[5/5] Streaming test inference in chunks to protect RAM...")
    test_s1_ids = list(te_s1_names.keys())
    pool_test_ids = list(te_pool_names.keys())
    chunk_size = 15_000

    for chunk_start in range(0, len(test_s1_ids), chunk_size):
        chunk_s1 = test_s1_ids[chunk_start : chunk_start + chunk_size]
        print(
            f"  Processing S1 batch {chunk_start} to {chunk_start + len(chunk_s1)} of {len(test_s1_ids)}..."
        )

        # 1. Block candidates for this chunk only
        chunk_cands = generate_candidates(
            chunk_s1,
            pool_test_ids,
            te_all_names,
            te_all_country,
            top_k=25,
            threshold=0.25,
        )

        # 2. Flatten pairs for feature extraction
        flat_pairs = []
        for s1_id in chunk_s1:
            clean_cands = [
                c
                for c in dict.fromkeys(chunk_cands.get(s1_id, []))
                if c.startswith(("S2-", "S3-")) and c != s1_id
            ]
            for c_id in clean_cands:
                flat_pairs.append((s1_id, c_id))

        # 3. Extract features & run model
        if flat_pairs:
            eval_pairs, X_chunk = extract_features_parallel(
                flat_pairs,
                te_all_names,
                te_all_addr,
                te_all_nums,
                workers=4,
            )
            probs = model.predict(X_chunk)
            matched_dict = {s1: [] for s1 in chunk_s1}
            for (s1, cand), p in zip(eval_pairs, probs):
                if p >= threshold:
                    matched_dict[s1].append(cand)
        else:
            matched_dict = {s1: [] for s1 in chunk_s1}

        # 4. Stream directly to disk (append mode)
        with open(cand_file, "a", newline="", encoding="utf-8") as fc:
            cw = csv.writer(fc, delimiter="\t", lineterminator="\n")
            for s1_id in chunk_s1:
                c_list = [
                    c
                    for c in dict.fromkeys(chunk_cands.get(s1_id, []))
                    if c.startswith(("S2-", "S3-")) and c != s1_id
                ]
                cw.writerow([s1_id, ",".join(c_list)])

        with open(match_file, "a", newline="", encoding="utf-8") as fm:
            mw = csv.writer(fm, delimiter="\t", lineterminator="\n")
            for s1_id in chunk_s1:
                clean_matches = [
                    c
                    for c in dict.fromkeys(matched_dict[s1_id])
                    if c.startswith(("S2-", "S3-")) and c != s1_id
                ]
                mw.writerow([s1_id, ",".join(clean_matches)])

    print("\nRunning compliance validation script...")
    res = subprocess.run(
        [
            "python3",
            "utils/validate_submission.py",
            "--matching",
            match_file,
            "--candidate",
            cand_file,
            "--test-dir",
            "dataset/test",
        ],
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    if res.stderr:
        print("Validation errors:\n", res.stderr)


if __name__ == "__main__":
    main()