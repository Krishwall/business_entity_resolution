from concurrent.futures import ProcessPoolExecutor
import numpy as np
from rapidfuzz import distance, fuzz

FEATURE_NAMES = [
    "lev_ratio",
    "partial_ratio",
    "token_sort",
    "token_set",
    "jw_sim",
    "addr_lev",
    "addr_token_sort",
    "addr_token_set",
    "jaccard_tokens",
    "num_match",
    "num_intersection",
    "name_len_diff",
    "name_len_ratio",
    "prefix_match",
]


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_nums: set[str],
    cand_name: str,
    cand_addr: str,
    cand_nums: set[str],
) -> list[float]:
    """Pure C-speed feature extraction per entity pair."""
    # Names
    lev = fuzz.ratio(s1_name, cand_name) / 100.0
    partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    t_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    t_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    jw = distance.JaroWinkler.similarity(s1_name, cand_name)

    # Addresses
    addr_lev = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

    # Token overlap
    t1 = set(s1_name.split())
    t2 = set(cand_name.split())
    union = t1 | t2
    jaccard = len(t1 & t2) / len(union) if union else 0.0

    # Number / PIN overlap
    intersect = len(s1_nums & cand_nums)
    if intersect > 0:
        num_m = 1.0
    elif len(s1_nums) > 0 and len(cand_nums) > 0:
        num_m = 0.0
    else:
        num_m = 0.5

    # Length disparities
    len1, len2 = len(s1_name), len(cand_name)
    len_diff = abs(len1 - len2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
    prefix_m = (
        1.0
        if (s1_name.startswith(cand_name) or cand_name.startswith(s1_name))
        and min(len1, len2) >= 4
        else 0.0
    )

    return [
        lev,
        partial,
        t_sort,
        t_set,
        jw,
        addr_lev,
        addr_tsort,
        addr_tset,
        jaccard,
        num_m,
        float(intersect),
        float(len_diff),
        len_ratio,
        prefix_m,
    ]


def _worker_batch_features(batch_data):
    """Worker task for multiprocessing."""
    results = []
    for item in batch_data:
        s1_id, c_id, n1, a1, num1, n2, a2, num2 = item
        feats = extract_pair_features(n1, a1, num1, n2, a2, num2)
        results.append((s1_id, c_id, feats))
    return results


def extract_features_parallel(
    pairs: list[tuple[str, str]],
    names: dict[str, str],
    addrs: dict[str, str],
    nums: dict[str, set[str]],
    workers: int = 4,
    batch_size: int = 20_000,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Processes candidate pairs across cores without building intermediate DataFrames."""
    if not pairs:
        return [], np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    # Pack raw strings into batches to avoid cross-process dict lookups
    batches = []
    current_batch = []
    for s1_id, c_id in pairs:
        current_batch.append(
            (
                s1_id,
                c_id,
                names[s1_id],
                addrs[s1_id],
                nums[s1_id],
                names[c_id],
                addrs[c_id],
                nums[c_id],
            )
        )
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    out_pairs = []
    out_features = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_res in executor.map(_worker_batch_features, batches):
            for s1_id, c_id, f in batch_res:
                out_pairs.append((s1_id, c_id))
                out_features.append(f)

    return out_pairs, np.array(out_features, dtype=np.float32)