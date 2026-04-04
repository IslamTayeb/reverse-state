from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch


def candidate_label(pert_a: str, pert_b: Optional[str] = None) -> str:
    if pert_b is None or pert_b == "":
        return str(pert_a)
    return f"{pert_a} + {pert_b}"


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
        raise ValueError(f"Expected perturbation vector with dim {pert_dim}, got {tensor.numel()}")

    if device is not None:
        tensor = tensor.to(device)
    return tensor


def combine_additive_deltas(*deltas: torch.Tensor) -> torch.Tensor:
    if not deltas:
        raise ValueError("At least one delta tensor is required")
    total = deltas[0]
    for delta in deltas[1:]:
        total = total + delta
    return total


def score_mean_deltas(
    target_deltas: Mapping[str, torch.Tensor],
    healthy_deltas: Optional[Mapping[str, torch.Tensor]] = None,
    healthy_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if not target_deltas:
        raise ValueError("target_deltas must not be empty")

    target_norms = torch.stack([torch.linalg.vector_norm(delta, ord=2) for delta in target_deltas.values()])
    target_efficacy = target_norms.mean()

    healthy_penalty: torch.Tensor
    if healthy_deltas:
        healthy_norms = torch.stack([torch.linalg.vector_norm(delta, ord=2) for delta in healthy_deltas.values()])
        healthy_penalty = healthy_norms.mean()
    else:
        healthy_penalty = torch.zeros((), device=target_efficacy.device, dtype=target_efficacy.dtype)

    score = target_efficacy - float(healthy_weight) * healthy_penalty
    return {
        "score": score,
        "target_efficacy": target_efficacy,
        "healthy_penalty": healthy_penalty,
    }


def realism_penalty(query_vectors: Sequence[torch.Tensor], library_vectors: torch.Tensor) -> torch.Tensor:
    if library_vectors.ndim != 2:
        raise ValueError(f"library_vectors must be rank-2, got shape {tuple(library_vectors.shape)}")
    if not query_vectors:
        raise ValueError("query_vectors must not be empty")

    penalties = []
    feature_dim = library_vectors.shape[1]
    for query in query_vectors:
        query_row = ensure_1d_float_tensor(query, pert_dim=feature_dim, device=library_vectors.device).unsqueeze(0)
        penalties.append(torch.cdist(query_row, library_vectors).min())
    return torch.stack(penalties).sum()


def topk_nearest_neighbors(
    query_vector: torch.Tensor,
    library_vectors: torch.Tensor,
    library_names: Sequence[str],
    k: int = 5,
) -> list[dict[str, float | str]]:
    if library_vectors.ndim != 2:
        raise ValueError(f"library_vectors must be rank-2, got shape {tuple(library_vectors.shape)}")
    if library_vectors.shape[0] != len(library_names):
        raise ValueError(
            f"library_vectors has {library_vectors.shape[0]} rows but library_names has {len(library_names)} entries"
        )
    if k <= 0:
        raise ValueError("k must be positive")

    query_row = ensure_1d_float_tensor(
        query_vector,
        pert_dim=library_vectors.shape[1],
        device=library_vectors.device,
    ).unsqueeze(0)
    distances = torch.cdist(query_row, library_vectors).reshape(-1)
    top_k = min(int(k), distances.numel())
    top_vals, top_idx = torch.topk(distances, k=top_k, largest=False)

    neighbors: list[dict[str, float | str]] = []
    for rank, (dist, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
        neighbors.append(
            {
                "name": str(library_names[idx]),
                "distance": float(dist),
                "neighbor_rank": rank,
            }
        )
    return neighbors
