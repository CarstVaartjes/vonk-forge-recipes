# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMR evaluation metrics: text-motion retrieval, R-Precision, and related scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor

from kimodo.model.tmr import TMR

from .base import Metric
from .retrieval import (
    calculate_activation_statistics,
    calculate_frechet_distance,
    contrastive_metrics,
    get_score_matrix_unit,
    get_scores_unit,
)

# Scores are between 0 and 1


def compute_tmr_per_sample_retrieval(
    motion_emb: np.ndarray,
    text_emb: np.ndarray,
    sample_ids: List[str],
    texts: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """For each sample (text query i), compute t2m rank of motion i and top-k retrieved motions with
    ids and texts.

    Returns list of dicts: [{"rank": int, "top_k": [{"id": str, "text": str}, ...]}, ...].
    """
    motion_emb = np.asarray(motion_emb).squeeze()
    text_emb = np.asarray(text_emb).squeeze()
    if motion_emb.ndim == 1:
        motion_emb = motion_emb[np.newaxis, :]
    if text_emb.ndim == 1:
        text_emb = text_emb[np.newaxis, :]
    n = motion_emb.shape[0]
    assert text_emb.shape[0] == n and len(sample_ids) == n and len(texts) == n
    scores = get_score_matrix_unit(text_emb, motion_emb)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        row = np.asarray(scores[i])
        order = np.argsort(row)[::-1]
        rank = int(np.where(order == i)[0][0]) + 1
        top_indices = order[:top_k]
        top_k_list = [{"id": sample_ids[j], "text": texts[j]} for j in top_indices]
        out.append({"rank": rank, "top_k": top_k_list})
    return out


class TMR_Metric(Metric):
    def __init__(
        self,
        tmr_model: TMR,
        ranks: List = [1, 2, 3, 5, 10],
        ranks_rounding=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tmr_model = tmr_model
        self.ranks = ranks
        self.ranks_rounding = ranks_rounding

    def clear(self):
        self.saved_metrics = defaultdict(list)
        self.saved_text_latents = []
        self.saved_motion_gen_latents = []
        self.saved_motion_gt_latents = []

    def _compute(
        self,
        motion_rep,
        pred_joints_output: Dict,
        gt_joints_output: Dict,
        text_x_dict: Dict,
        lengths: Tensor,
        **kwargs,
    ) -> Dict:
        pred_posed_joints = pred_joints_output["posed_joints"]
        original_skeleton = motion_rep.skeleton if motion_rep is not None else None
        latents_motion = self.tmr_model.encode_motion(
            pred_posed_joints,
            lengths=lengths,
            original_skeleton=original_skeleton,
            unit_vector=True,
        )
        latents_motion = latents_motion.cpu().numpy()

        if isinstance(text_x_dict, dict) and "texts" in text_x_dict:
            latents_text = self.tmr_model.encode_raw_text(text_x_dict["texts"], unit_vector=True)
        else:
            latents_text = self.tmr_model.encode_text(text_x_dict, unit_vector=True)
        if latents_text.dim() == 1:
            latents_text = latents_text.unsqueeze(0)
        latents_text = latents_text.cpu().numpy()

        self.saved_text_latents.append(latents_text)
        self.saved_motion_gen_latents.append(latents_motion)

        scores_text = get_scores_unit(latents_motion, latents_text)
        output = {"TMR/t2m_sim": scores_text}

        if gt_joints_output is not None and "posed_joints" in gt_joints_output:
            gt_posed_joints = gt_joints_output["posed_joints"]
            gt_latents_motion = self.tmr_model.encode_motion(
                gt_posed_joints,
                lengths=lengths,
                original_skeleton=original_skeleton,
                unit_vector=True,
            )
            gt_latents_motion = gt_latents_motion.cpu().numpy()
            self.saved_motion_gt_latents.append(gt_latents_motion)

            gt_scores_text = get_scores_unit(gt_latents_motion, latents_text)
            scores_motion = get_scores_unit(latents_motion, gt_latents_motion)

            output["TMR/t2m_gt_sim"] = gt_scores_text
            output["TMR/m2m_sim"] = scores_motion

        # pytorch tensors
        for key, val in output.items():
            output[key] = torch.tensor(val)
        return output

    def aggregate(self):
        output = {}
        for key, lst in self.saved_metrics.items():
            output[key] = np.concatenate(lst)

        assert self.saved_text_latents, "Should call the metric at least once."

        text_latents = np.concatenate(self.saved_text_latents)
        motion_gen_latents = np.concatenate(self.saved_motion_gen_latents)

        batch_size = len(text_latents)
        assert text_latents.shape == motion_gen_latents.shape

        scores_t2m = get_score_matrix_unit(text_latents, motion_gen_latents)
        scores_t2t = get_score_matrix_unit(text_latents, text_latents)

        t2m_metrics = contrastive_metrics(
            scores=scores_t2m,
            scores_t2t=scores_t2t,
            threshold=0.99,
            rounding=2,
        )

        for key, val in t2m_metrics.items():
            output["TMR/t2m_R/" + key] = val

        mu_gen, cov_gen = calculate_activation_statistics(motion_gen_latents)
        mu_text, cov_text = calculate_activation_statistics(text_latents)

        fid_gen_text = calculate_frechet_distance(mu_gen, cov_gen, mu_text, cov_text)
        output["TMR/FID/gen_text"] = fid_gen_text

        if self.saved_motion_gt_latents:
            motion_gt_latents = np.concatenate(self.saved_motion_gt_latents)
            assert motion_gt_latents.shape == motion_gen_latents.shape

            scores_m2gm = get_score_matrix_unit(motion_gen_latents, motion_gt_latents)
            scores_t2gm = get_score_matrix_unit(text_latents, motion_gt_latents)

            m2gm_metrics = contrastive_metrics(
                scores=scores_m2gm,
                scores_t2t=scores_t2t,
                threshold=0.99,
                rounding=2,
            )
            for key, val in m2gm_metrics.items():
                output["TMR/m2m_R/" + key] = val

            t2gm_metrics = contrastive_metrics(
                scores=scores_t2gm,
                scores_t2t=scores_t2t,
                threshold=0.99,
                rounding=2,
            )
            for key, val in t2gm_metrics.items():
                output["TMR/t2m_gt_R/" + key] = val

            mu_gt_motion, cov_gt_motion = calculate_activation_statistics(motion_gt_latents)
            fid_gen_motion = calculate_frechet_distance(
                mu_gen,
                cov_gen,
                mu_gt_motion,
                cov_gt_motion,
            )
            output["TMR/FID/gen_gt"] = fid_gen_motion

            fid_gt_text = calculate_frechet_distance(
                mu_gt_motion,
                cov_gt_motion,
                mu_text,
                cov_text,
            )
            output["TMR/FID/gt_text"] = fid_gt_text

        for key, val in output.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                val = torch.tensor([val for _ in range(batch_size)])

            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            output[key] = val.cpu().float()
        return output


class TMR_EmbeddingMetric(Metric):
    """TMR metrics from precomputed motion and text embeddings (no model load).

    Use in the loop: pass motion_emb and text_emb per sample; aggregate() computes retrieval metrics.
    """

    def __init__(self, ranks_rounding: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.ranks_rounding = ranks_rounding

    def clear(self):
        self.saved_metrics = defaultdict(list)
        self.saved_text_latents = []
        self.saved_motion_gen_latents = []
        self.saved_motion_gt_latents = []

    def _compute(
        self,
        motion_emb=None,
        text_emb=None,
        gt_motion_emb=None,
        **kwargs,
    ) -> Dict:
        if motion_emb is None or text_emb is None:
            return {}
        motion_emb = np.asarray(motion_emb)
        text_emb = np.asarray(text_emb)
        if motion_emb.ndim == 1:
            motion_emb = motion_emb[np.newaxis, :]
        if text_emb.ndim == 1:
            text_emb = text_emb[np.newaxis, :]
        self.saved_text_latents.append(text_emb)
        self.saved_motion_gen_latents.append(motion_emb)
        if gt_motion_emb is not None:
            gt_motion_emb = np.asarray(gt_motion_emb)
            if gt_motion_emb.ndim == 1:
                gt_motion_emb = gt_motion_emb[np.newaxis, :]
            self.saved_motion_gt_latents.append(gt_motion_emb)
        scores = get_scores_unit(motion_emb, text_emb)
        return {"TMR/t2m_sim": torch.tensor(scores, dtype=torch.float32)}

    def aggregate(self):
        output = {}
        for key, lst in self.saved_metrics.items():
            output[key] = np.concatenate(lst)
        if not self.saved_text_latents:
            return output
        text_latents = np.concatenate(self.saved_text_latents)
        motion_gen_latents = np.concatenate(self.saved_motion_gen_latents)
        batch_size = len(text_latents)
        assert text_latents.shape == motion_gen_latents.shape
        scores_t2m = get_score_matrix_unit(text_latents, motion_gen_latents)
        scores_t2t = get_score_matrix_unit(text_latents, text_latents)
        t2m_metrics = contrastive_metrics(
            scores=scores_t2m,
            scores_t2t=scores_t2t,
            threshold=0.99,
            rounding=self.ranks_rounding,
        )
        for key, val in t2m_metrics.items():
            output["TMR/t2m_R/" + key] = val
        if batch_size >= 2:
            mu_gen, cov_gen = calculate_activation_statistics(motion_gen_latents)
            mu_text, cov_text = calculate_activation_statistics(text_latents)
            output["TMR/FID/gen_text"] = calculate_frechet_distance(mu_gen, cov_gen, mu_text, cov_text)
        else:
            output["TMR/FID/gen_text"] = float("nan")
        if self.saved_motion_gt_latents:
            motion_gt_latents = np.concatenate(self.saved_motion_gt_latents)
            assert motion_gt_latents.shape == motion_gen_latents.shape
            scores_m2gm = get_score_matrix_unit(motion_gen_latents, motion_gt_latents)
            scores_t2gm = get_score_matrix_unit(text_latents, motion_gt_latents)
            m2gm_metrics = contrastive_metrics(
                scores=scores_m2gm,
                scores_t2t=scores_t2t,
                threshold=0.99,
                rounding=self.ranks_rounding,
            )
            for key, val in m2gm_metrics.items():
                output["TMR/m2m_R/" + key] = val
            t2gm_metrics = contrastive_metrics(
                scores=scores_t2gm,
                scores_t2t=scores_t2t,
                threshold=0.99,
                rounding=self.ranks_rounding,
            )
            for key, val in t2gm_metrics.items():
                output["TMR/t2m_gt_R/" + key] = val
            if batch_size >= 2:
                mu_gt_motion, cov_gt_motion = calculate_activation_statistics(motion_gt_latents)
                output["TMR/FID/gen_gt"] = calculate_frechet_distance(mu_gen, cov_gen, mu_gt_motion, cov_gt_motion)
                output["TMR/FID/gt_text"] = calculate_frechet_distance(mu_gt_motion, cov_gt_motion, mu_text, cov_text)
            else:
                output["TMR/FID/gen_gt"] = float("nan")
                output["TMR/FID/gt_text"] = float("nan")
        for key, val in output.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                val = torch.tensor([val for _ in range(batch_size)])
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            output[key] = val.cpu().float()
        return output
