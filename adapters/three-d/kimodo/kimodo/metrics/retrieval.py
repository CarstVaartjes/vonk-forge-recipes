# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval (R@K, MedR) and FID math shared by the TMR metric implementations."""

import logging

import numpy as np
from scipy import linalg

log = logging.getLogger(__name__)


def get_score_matrix_unit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity rescaled to ``[0, 1]`` (assumes inputs are unit vectors)."""
    sim_matrix = np.einsum("b i, c i -> b c", x, y)
    return sim_matrix / 2 + 0.5


def get_scores_unit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise (paired) cosine similarity rescaled to ``[0, 1]``."""
    similarity = np.einsum("... i, ... i", x, y)
    return similarity / 2 + 0.5


def all_contrastive_metrics(sims, emb=None, threshold=None, rounding: int = 2, return_cols: bool = False):
    """Compute t2m and m2t retrieval metrics, with optional text-self-similarity tie breaking."""
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols = contrastive_metrics(sims, text_selfsim, threshold, return_cols=True, rounding=rounding)
    m2t_m, m2t_cols = contrastive_metrics(sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding)

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(scores, scores_t2t=None, threshold=None, rounding: int = 2, return_cols: bool = False):
    """Compute R@1/2/3/5/10 + median rank from a square score matrix; optional ``scores_t2t``
    resolves ties."""
    n, m = scores.shape
    assert n == m
    num_queries = n

    dists = -scores
    sorted_dists = np.sort(dists, axis=1)
    gt_dists = np.diag(dists)[:, None]

    if scores_t2t is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(scores_t2t > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        cols = break_ties_average(sorted_dists, gt_dists)

    assert cols.size == num_queries, f"expected ranks to match queries ({cols.size} vs {num_queries}) "

    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)
    metrics["len"] = num_queries

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    if return_cols:
        return metrics, cols
    return metrics


def break_ties_average(sorted_dists, gt_dists):
    """Resolve ranking ties by averaging the column indices of all matching positions."""
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    return summed_cols / counts


def calculate_activation_statistics(activations: np.ndarray):
    """Compute ``(mean, covariance)`` over a stack of latent activations."""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    """Numerically stable Frechet (FID-style) distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # scipy >= 1.18 dropped sqrtm's `disp` argument (and its error-estimate return); a non-finite
    # result is caught below either way
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        log.warning(
            "fid calculation produces singular product; adding %s to diagonal of cov estimates",
            eps,
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_tmr_retrieval_metrics(
    motion_emb: np.ndarray,
    text_emb: np.ndarray,
    gt_motion_emb: np.ndarray | None = None,
    rounding: int = 2,
) -> dict[str, float]:
    """Compute the TMR retrieval + FID summary (``TMR/*`` keys) from unit embeddings.

    FID entries are ``nan`` when fewer than 2 samples are available.
    """
    if motion_emb.shape != text_emb.shape:
        raise ValueError(f"Expected same shape for motion/text embeddings, got {motion_emb.shape} vs {text_emb.shape}")

    scores_t2m = get_score_matrix_unit(text_emb, motion_emb)
    scores_t2t = get_score_matrix_unit(text_emb, text_emb)

    output: dict[str, float] = {}
    t2m_metrics = contrastive_metrics(
        scores=scores_t2m,
        scores_t2t=scores_t2t,
        threshold=0.99,
        rounding=rounding,
    )
    for key, val in t2m_metrics.items():
        output[f"TMR/t2m_R/{key}"] = float(val)

    n_samples = len(motion_emb)
    if n_samples >= 2:
        mu_gen, cov_gen = calculate_activation_statistics(motion_emb)
        mu_text, cov_text = calculate_activation_statistics(text_emb)
        output["TMR/FID/gen_text"] = float(calculate_frechet_distance(mu_gen, cov_gen, mu_text, cov_text))
    else:
        output["TMR/FID/gen_text"] = float("nan")

    if gt_motion_emb is not None:
        if gt_motion_emb.shape != motion_emb.shape:
            raise ValueError(f"Expected gt motion embeddings shape {motion_emb.shape}, got {gt_motion_emb.shape}")

        scores_m2gm = get_score_matrix_unit(motion_emb, gt_motion_emb)
        scores_t2gm = get_score_matrix_unit(text_emb, gt_motion_emb)

        m2gm_metrics = contrastive_metrics(
            scores=scores_m2gm,
            scores_t2t=scores_t2t,
            threshold=0.99,
            rounding=rounding,
        )
        for key, val in m2gm_metrics.items():
            output[f"TMR/m2m_R/{key}"] = float(val)

        t2gm_metrics = contrastive_metrics(
            scores=scores_t2gm,
            scores_t2t=scores_t2t,
            threshold=0.99,
            rounding=rounding,
        )
        for key, val in t2gm_metrics.items():
            output[f"TMR/t2m_gt_R/{key}"] = float(val)

        if n_samples >= 2:
            mu_gt_motion, cov_gt_motion = calculate_activation_statistics(gt_motion_emb)
            output["TMR/FID/gen_gt"] = float(calculate_frechet_distance(mu_gen, cov_gen, mu_gt_motion, cov_gt_motion))
            output["TMR/FID/gt_text"] = float(
                calculate_frechet_distance(mu_gt_motion, cov_gt_motion, mu_text, cov_text)
            )
        else:
            output["TMR/FID/gen_gt"] = float("nan")
            output["TMR/FID/gt_text"] = float("nan")

    return output
