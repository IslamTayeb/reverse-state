import pytest
import torch

from state.tx.optimization import (
    aggregate_context_scores,
    compute_retrieval_metrics,
    ensure_1d_float_tensor,
    score_target_similarity,
)


def test_ensure_1d_float_tensor_validates_dimension():
    tensor = ensure_1d_float_tensor(torch.tensor([[1, 2], [3, 4]]), pert_dim=4)
    assert tensor.shape == (4,)

    with pytest.raises(ValueError, match="Expected tensor with dim 3"):
        ensure_1d_float_tensor(torch.tensor([1.0, 2.0]), pert_dim=3)


def test_score_target_similarity_cosine_tracks_direction():
    target = torch.tensor([2.0, -1.0, 0.5])
    same_direction = torch.tensor([4.0, -2.0, 1.0])
    opposite_direction = torch.tensor([-4.0, 2.0, -1.0])

    assert score_target_similarity(target, same_direction, metric="cosine").item() > 0.999
    assert score_target_similarity(target, opposite_direction, metric="cosine").item() < -0.999


def test_score_target_similarity_pearson_ignores_global_shift():
    target = torch.tensor([1.0, 3.0, 5.0])
    shifted = torch.tensor([10.0, 12.0, 14.0])

    assert score_target_similarity(target, shifted, metric="pearson").item() > 0.999


def test_score_target_similarity_l2_prefers_closer_match():
    target = torch.tensor([1.0, -2.0])
    close = torch.tensor([1.1, -2.1])
    far = torch.tensor([4.0, 3.0])

    assert (
        score_target_similarity(target, close, metric="l2").item()
        > score_target_similarity(target, far, metric="l2").item()
    )


def test_aggregate_context_scores_returns_summary_stats():
    scores = {
        "ctx_a": torch.tensor(0.75),
        "ctx_b": torch.tensor(0.25),
    }

    summary = aggregate_context_scores(scores)

    assert summary["score"].item() == pytest.approx(0.5)
    assert summary["score_min"].item() == pytest.approx(0.25)
    assert summary["score_max"].item() == pytest.approx(0.75)
    assert summary["score_std"].item() == pytest.approx(0.25)


def test_compute_retrieval_metrics_reports_rank_stats():
    metrics = compute_retrieval_metrics([1, 2, 5])

    assert metrics["top_1"] == pytest.approx(1.0 / 3.0)
    assert metrics["top_5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.2) / 3.0)
    assert metrics["median_rank"] == pytest.approx(2.0)
