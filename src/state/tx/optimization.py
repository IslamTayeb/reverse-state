from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


def ensure_1d_float_tensor(
    value, *, pert_dim: Optional[int] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(dtype=torch.float32)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.asarray(value)).to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)

    tensor = tensor.reshape(-1)
    if pert_dim is not None and tensor.numel() != pert_dim:
        raise ValueError(f"Expected tensor with dim {pert_dim}, got {tensor.numel()}")

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def score_target_similarity(
    target_delta: torch.Tensor,
    candidate_delta: torch.Tensor,
    *,
    metric: str = "cosine",
    eps: float = 1e-8,
) -> torch.Tensor:
    target = ensure_1d_float_tensor(target_delta)
    candidate = ensure_1d_float_tensor(candidate_delta, pert_dim=target.numel(), device=target.device)

    if metric == "cosine":
        denom = torch.linalg.vector_norm(target, ord=2) * torch.linalg.vector_norm(candidate, ord=2)
        if float(denom.detach().cpu().item()) <= eps:
            return torch.zeros((), device=target.device, dtype=target.dtype)
        return torch.dot(target, candidate) / denom.clamp_min(eps)

    if metric == "pearson":
        target_centered = target - target.mean()
        candidate_centered = candidate - candidate.mean()
        denom = torch.linalg.vector_norm(target_centered, ord=2) * torch.linalg.vector_norm(candidate_centered, ord=2)
        if float(denom.detach().cpu().item()) <= eps:
            return torch.zeros((), device=target.device, dtype=target.dtype)
        return torch.dot(target_centered, candidate_centered) / denom.clamp_min(eps)

    if metric == "l2":
        return -torch.linalg.vector_norm(target - candidate, ord=2)

    raise ValueError(f"Unsupported match metric: {metric}")


def aggregate_context_scores(context_scores: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not context_scores:
        raise ValueError("context_scores must not be empty")

    scores = torch.stack([score.reshape(()) for score in context_scores.values()])
    if scores.numel() == 1:
        score_std = torch.zeros((), device=scores.device, dtype=scores.dtype)
    else:
        score_std = scores.std(unbiased=False)

    return {
        "score": scores.mean(),
        "score_min": scores.min(),
        "score_max": scores.max(),
        "score_std": score_std,
    }


def compute_retrieval_metrics(true_ranks: Sequence[int], topk: Iterable[int] = (1, 5)) -> dict[str, float]:
    if not true_ranks:
        raise ValueError("true_ranks must not be empty")

    ranks = np.asarray(true_ranks, dtype=np.int64)
    if np.any(ranks <= 0):
        raise ValueError("true_ranks must contain positive 1-indexed ranks")

    metrics = {
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
    }
    for k in topk:
        if int(k) <= 0:
            raise ValueError("topk values must be positive")
        metrics[f"top_{int(k)}"] = float((ranks <= int(k)).mean())
    return metrics
