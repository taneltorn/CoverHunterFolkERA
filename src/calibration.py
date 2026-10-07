#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score calibration for work identification embeddings.

Given a set of reference embeddings already grouped by work, sample
cross-work (unrelated) performance pairs, compute their cosine distances,
and summarize the resulting "negative distribution." The summary lets us
convert a raw cosine distance into a model-portable score:

    z_score(d, cal) = (d - mu) / sigma
        Standard score under the negative distribution.
        Strongly negative -> match-like distance.

    pvalue(d, cal)   ~ ECDF_neg(d)
        Empirical P(D <= d | unrelated), interpolated between stored
        quantile anchors. Small -> unlikely under null of "unrelated
        tunes" -> strong match. Distances below the smallest sampled
        negative return the resolution floor, ~1/n_pairs: read such a
        value as "p <= 1/n_pairs", not as an exact probability.

What this null is, and is not: it is the distribution of distances
between two random REFERENCE performances of different works, sampled
uniformly over performances. It is not the distribution of query-to-
centroid distances (centroid averaging pulls unrelated distances in,
more so for works with many performances), and it is not a null for the
best match across a library (multiple comparisons).

Calibration is meaningful only in the geometry actually used at query
time. This module assumes unit-normalized embeddings and cosine distance
(1 - dot). Do not call compute_calibration on raw embeddings; the result
would not transfer to inference.

Calibration record schema (dict, pickle-friendly):

    {
        "mu":              float,        # mean of negative-pair distances
        "sigma":           float,        # std of negative-pair distances
        "ecdf_quantiles":  np.ndarray,   # shape (n_quantiles,), float32,
                                         # sorted ascending; the i-th entry
                                         # is the distance value at which
                                         # the empirical CDF equals
                                         # ecdf_probs[i].
        "ecdf_probs":      np.ndarray,   # shape (n_quantiles,), float32,
                                         # exact empirical CDF values
                                         # (k+1)/n_pairs at order
                                         # statistics chosen log-spaced
                                         # toward 0 and 1 for tail
                                         # resolution; first anchor is
                                         # 1/n_pairs (the resolution
                                         # floor), last ~1. Strictly
                                         # increasing, as are the paired
                                         # ecdf_quantiles.
        "n_pairs":         int,          # how many pairs were sampled
        "n_quantiles":     int,          # len(ecdf_quantiles); at most the
                                         # requested count (duplicate
                                         # anchors near the floor removed)
        "seed":            int,          # RNG seed used for sampling
        "sampling":        str,          # description of the pair design
        "n_perfs":         int,          # reference perfs available
        "n_works":         int,          # distinct works among them
    }

The last four keys were added 2026-10-05.

@author: alanngnet with Claude Opus 4.7 2026-05-03

Revised with Claude Opus 4.8 2026-07-17

Revised 2026-10-05 by alanngnet with Claude (Opus 5.5), following a
review of calibration validity on singleton-heavy corpora:
  - ECDF resolution floor now tracks the sample size: the lowest
    probability anchor is 1/n_pairs instead of a hard-coded 1e-6.
    Previously, for n_pairs < 1e6, ~10% of the anchors collapsed onto
    the single smallest sampled distance and any closer distance was
    reported as p = 1e-6, a resolution the sample did not have.
  - Anchors store exact empirical CDF values (k+1)/n at deduplicated
    order statistics, so np.interp always receives strictly increasing
    x (log-spaced targets finer than one sample previously produced
    repeated x-values, for which np.interp's result is undefined).
  - Pair distances are computed in fixed-size chunks, so large n_pairs
    (tools.make_embeds now defaults to 1,000,000) never materialize
    multi-GB gathered matrices. The rejection-sampling iteration cap
    scales with n_pairs accordingly.
  - The calibration record now includes provenance fields
  	(seed, sampling, n_perfs, n_works).
  - This header docstring now states what the null does and does not model.
"""

import numpy as np
from typing import Dict, List

# Rows per chunk for pairwise dot products: 200k x 256 float32 = ~200 MB
# per side, comfortably within unified memory on the target hardware 
# (personal desktop/laptop).
_PAIR_CHUNK = 200_000


def _log_spaced_probs(n: int, n_pairs: int) -> np.ndarray:
    """Probabilities for ECDF anchors, log-spaced toward both 0 and 1.

    Half the points concentrate near 0 (log-spaced from the resolution
    floor 1/n_pairs up to 0.5); the other half mirror them near 1. This
    puts resolution where we need it for confidence reporting (very small
    p, very large p) and saves storage in the bulk. Anchors below
    1/n_pairs would all map to the single smallest sampled distance, so
    the floor is tied to the sample size.
    """
    half = n // 2
    floor = min(1.0 / max(n_pairs, 1), 0.25)
    lower = np.logspace(np.log10(floor), np.log10(0.5), half, endpoint=False)
    upper = 1.0 - lower[::-1]
    middle = np.array([0.5])
    return np.concatenate([lower, middle, upper])


def _pair_distances(matrix: np.ndarray, ii: np.ndarray,
                    jj: np.ndarray) -> np.ndarray:
    """Cosine distances 1 - <m_i, m_j> for index pairs, chunked.

    Dot products accumulate in float64 for ECDF accuracy at the tails;
    the clip guards against floating-point overshoot of [-1, 1] for
    near-identical or near-antipodal unit vectors.
    """
    out = np.empty(len(ii), dtype=np.float32)
    for s in range(0, len(ii), _PAIR_CHUNK):
        e = min(s + _PAIR_CHUNK, len(ii))
        dots = np.einsum(
            "nd,nd->n", matrix[ii[s:e]], matrix[jj[s:e]], dtype=np.float64
        )
        out[s:e] = (1.0 - np.clip(dots, -1.0, 1.0)).astype(np.float32)
    return out


def compute_calibration(
    embeddings: Dict[str, np.ndarray],
    work_to_perfs: Dict[str, List[str]],
    n_pairs: int = 1_000_000,
    n_quantiles: int = 1001,
    seed: int = 12345,
) -> Dict:
    """Sample cross-work performance pairs, compute their cosine distances,
    and return a calibration record.

    Assumes embeddings are already L2-normalized (unit vectors). Cosine
    distance is computed as 1 - dot.

    Args:
        embeddings: dict perf_id -> unit-normalized embedding vector.
        work_to_perfs: dict work_id -> list of perf_ids belonging to that
            work. Used to ensure sampled pairs are cross-work.
        n_pairs: target number of cross-work pairs to sample. The ECDF
            resolution floor is 1/n_pairs; at 1e6 this costs a few
            seconds and supports p-values down to ~1e-5 with ~10 samples
            behind them.
        n_quantiles: resolution of the stored empirical CDF. 1001 gives
            fine tail resolution at ~8 KB.
        seed: RNG seed for reproducibility. Fixing it keeps the sampled
            negative distribution stable across runs on the same reference
            set.

    Returns:
        Calibration record dict per the schema in the module docstring.
    """
    rng = np.random.default_rng(seed)

    # Stack perfs into a contiguous matrix; remember work assignment per row
    perf_ids = list(embeddings.keys())
    n_perfs = len(perf_ids)
    if n_perfs < 2:
        raise ValueError(
            f"Cannot calibrate from {n_perfs} perfs; need at least 2."
        )

    perf_to_work = {}
    for work_id, perfs in work_to_perfs.items():
        for p in perfs:
            perf_to_work[p] = work_id
    works_arr = np.array([perf_to_work[p] for p in perf_ids], dtype=object)

    # If every perf belongs to the same work, calibration is impossible.
    n_distinct_works = len(set(works_arr.tolist()))
    if n_distinct_works < 2:
        raise ValueError(
            "Cannot calibrate: all reference perfs belong to one work."
        )

    # Stack embeddings as (N, D) float32 matrix for fast dot products.
    D = len(embeddings[perf_ids[0]])
    matrix = np.empty((n_perfs, D), dtype=np.float32)
    for i, p in enumerate(perf_ids):
        matrix[i] = embeddings[p]

    # Small reference sets: enumerate all cross-work pairs, then sample
    # without replacement. Large ones: rejection-sample index pairs and
    # keep cross-work pairs (yield ~1 - sum_w (n_w/N)^2, near 1 for any
    # realistic CSI library), with an iteration cap so a pathological
    # work distribution fails loud instead of spinning.
    total_pairs_possible = n_perfs * (n_perfs - 1) // 2
    if total_pairs_possible <= max(n_pairs * 4, 100_000):
        ii_all, jj_all = np.triu_indices(n_perfs, k=1)
        cross_mask = works_arr[ii_all] != works_arr[jj_all]
        ii_all = ii_all[cross_mask]
        jj_all = jj_all[cross_mask]
        n_available = len(ii_all)
        if n_available == 0:
            raise ValueError(
                "Cannot calibrate: no cross-work pairs exist in the "
                "reference set. Check that work_to_perfs covers more "
                "than one work."
            )
        actual_n = min(n_pairs, n_available)
        if actual_n < n_pairs:
            print(
                f"Warning: requested {n_pairs} calibration pairs but "
                f"only {n_available} cross-work pairs exist; using "
                f"{actual_n}."
            )
        sel = rng.choice(n_available, size=actual_n, replace=False)
        distances = _pair_distances(matrix, ii_all[sel], jj_all[sel])
        n_pairs = actual_n  # update for record-keeping
        sampling = ("all cross-work perf pairs enumerated, then sampled "
                    "uniformly without replacement")
    else:
        distances = np.empty(n_pairs, dtype=np.float32)
        filled = 0
        max_iterations = 100 * (1 + n_pairs // _PAIR_CHUNK)
        iteration = 0
        while filled < n_pairs:
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError(
                    f"Calibration sampling failed: only {filled} of "
                    f"{n_pairs} cross-work pairs sampled after "
                    f"{max_iterations} iterations. The reference set "
                    f"may be severely imbalanced (one work dominating). "
                    f"Inspect work_to_perfs distribution."
                )
            need = n_pairs - filled
            draw = min(max(need * 2, 1024), _PAIR_CHUNK)
            i = rng.integers(0, n_perfs, size=draw)
            j = rng.integers(0, n_perfs, size=draw)
            keep_mask = (i != j) & (works_arr[i] != works_arr[j])
            ii = i[keep_mask][:need]
            jj = j[keep_mask][:need]
            if len(ii) == 0:
                continue
            distances[filled : filled + len(ii)] = _pair_distances(
                matrix, ii, jj
            )
            filled += len(ii)
        sampling = ("uniform random cross-work perf pairs (with "
                    "replacement), rejection-sampled")

    mu = float(distances.mean())
    sigma = float(distances.std(ddof=1))

    # ECDF stored as sorted quantiles with log-spaced probabilities so that
    # the distribution is dense at both tails (high confidence and no-match).
    # Lookup at query time is a binary search.
    # Anchors are order statistics chosen at log-spaced probabilities;
    # each stores its exact empirical CDF value P(D <= d_(k)) = (k+1)/n.
    # Near the floor the log spacing is finer than one sample, so several
    # targets land on the same order statistic; duplicates (and tied
    # distance values, keeping the largest CDF value as the ECDF is
    # right-continuous) are removed so np.interp sees strictly increasing
    # x. The stored length can therefore be less than n_quantiles.
    sorted_d = np.sort(distances)
    probs = _log_spaced_probs(n_quantiles, n_pairs)
    quantile_idx = np.unique(
        np.clip(
            np.ceil(probs * n_pairs).astype(np.int64) - 1, 0, n_pairs - 1
        )
    )
    q_vals = sorted_d[quantile_idx]
    # For tied values keep the last (highest) index.
    last = np.r_[q_vals[1:] != q_vals[:-1], True]
    quantile_idx = quantile_idx[last]
    ecdf_quantiles = sorted_d[quantile_idx].astype(np.float32)
    ecdf_probs = ((quantile_idx + 1) / n_pairs).astype(np.float32)

    return {
        "mu": mu,
        "sigma": sigma,
        "ecdf_quantiles": ecdf_quantiles,
        "ecdf_probs": ecdf_probs,
        "n_pairs": int(n_pairs),
        "n_quantiles": int(len(ecdf_probs)),
        "seed": int(seed),
        "sampling": sampling,
        "n_perfs": int(n_perfs),
        "n_works": int(n_distinct_works),
    }


def z_score(distance, calibration: Dict):
    """Standard score under the negative distribution.

    Strongly negative values mean the distance is much smaller than
    typical for unrelated pairs - i.e. a match-like distance.
    A z-score of 0 means the query looks like a random unrelated pair.

    Accepts a scalar or numpy array of distances; returns the same shape.
    """
    d = np.asarray(distance)
    return (d - calibration["mu"]) / calibration["sigma"]


def pvalue(distance, calibration: Dict):
    """Empirical P(D <= distance | unrelated pair).

    Small p-value means the distance is unlikely under the null of
    unrelated works - i.e. a strong match. Bounded in [0, 1]. Distances
    below the smallest stored quantile return the lowest anchor
    (~1/n_pairs), which is a bound, not an estimate.

    Accepts a scalar or numpy array of distances; returns the same shape.
    """
    quantiles = calibration["ecdf_quantiles"]
    probs = calibration["ecdf_probs"]
    d = np.asarray(distance)
    # np.interp does linear interpolation between anchors; both arrays
    # must be ascending, which they are by construction.
    return np.clip(np.interp(d, quantiles, probs), 0.0, 1.0)
