from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def block_single_country(
    s1_ids: list[str],
    pool_ids: list[str],
    names_dict: dict[str, str],
    top_k: int = 25,
    threshold: float = 0.25,
    batch_size: int = 5000,
) -> dict[str, list[str]]:
    """Runs fast chunked matrix multiplication for one country partition."""
    candidates = {s1_id: [] for s1_id in s1_ids}
    if not s1_ids or not pool_ids:
        return candidates

    s1_texts = [names_dict[eid] for eid in s1_ids]
    pool_texts = [names_dict[eid] for eid in pool_ids]

    # Char n-grams capture typos, suffixes, and OCR errors
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_features=250_000,
        dtype=np.float32,
    )
    vec.fit(s1_texts + pool_texts)

    s1_mat = vec.transform(s1_texts)
    pool_mat = vec.transform(pool_texts).T.tocsr()  # Transpose for dot product

    pool_ids_arr = np.array(pool_ids)

    # Process S1 in slices to keep intermediate matrices small
    for start_idx in range(0, s1_mat.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, s1_mat.shape[0])
        sims: csr_matrix = s1_mat[start_idx:end_idx].dot(pool_mat)

        for row_offset in range(sims.shape[0]):
            global_s1_id = s1_ids[start_idx + row_offset]
            row = sims.getrow(row_offset)

            if row.nnz == 0:
                continue

            cols = row.indices
            vals = row.data

            # Filter by minimum threshold
            mask = vals >= threshold
            valid_cols = cols[mask]
            valid_vals = vals[mask]

            if len(valid_cols) > top_k:
                top_part = np.argpartition(valid_vals, -top_k)[-top_k:]
                valid_cols = valid_cols[top_part]

            candidates[global_s1_id] = pool_ids_arr[valid_cols].tolist()

    return candidates


def generate_candidates(
    s1_ids: list[str],
    pool_ids: list[str],
    names_dict: dict[str, str],
    country_dict: dict[str, str],
    top_k: int = 25,
    threshold: float = 0.25,
) -> dict[str, list[str]]:
    """Partitions records by country, runs blocking, and merges results."""
    # Partition IDs by country
    s1_by_country = defaultdict(list)
    pool_by_country = defaultdict(list)

    for eid in s1_ids:
        s1_by_country[country_dict[eid]].append(eid)
    for eid in pool_ids:
        pool_by_country[country_dict[eid]].append(eid)

    all_candidates = {}
    for country, s1_c_ids in s1_by_country.items():
        pool_c_ids = pool_by_country.get(country, [])
        cand_map = block_single_country(
            s1_c_ids,
            pool_c_ids,
            names_dict,
            top_k=top_k,
            threshold=threshold,
        )
        all_candidates.update(cand_map)

    # Ensure every S1 ID is present
    for eid in s1_ids:
        if eid not in all_candidates:
            all_candidates[eid] = []

    return all_candidates