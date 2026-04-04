import torch

from state.tx.optimization import (
    candidate_label,
    combine_additive_deltas,
    score_mean_deltas,
    topk_nearest_neighbors,
)


def test_candidate_label_formats_pairs():
    assert candidate_label("drug_a") == "drug_a"
    assert candidate_label("drug_a", "drug_b") == "drug_a + drug_b"


def test_combine_additive_deltas_sums_components():
    delta_a = torch.tensor([1.0, -2.0, 0.5])
    delta_b = torch.tensor([0.5, 1.0, -1.5])

    combined = combine_additive_deltas(delta_a, delta_b)

    assert torch.equal(combined, torch.tensor([1.5, -1.0, -1.0]))


def test_score_mean_deltas_penalizes_healthy_shift():
    target = {"tumor": torch.tensor([3.0, 4.0])}
    healthy = {"healthy": torch.tensor([0.0, 5.0])}

    score = score_mean_deltas(target, healthy, healthy_weight=0.5)

    assert score["target_efficacy"].item() == 5.0
    assert score["healthy_penalty"].item() == 5.0
    assert score["score"].item() == 2.5


def test_topk_nearest_neighbors_returns_sorted_names():
    library = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ]
    )
    neighbors = topk_nearest_neighbors(
        query_vector=torch.tensor([0.9, 0.9]),
        library_vectors=library,
        library_names=["dmso", "drug_a", "drug_b"],
        k=2,
    )

    assert [neighbor["name"] for neighbor in neighbors] == ["drug_a", "dmso"]
    assert neighbors[0]["distance"] <= neighbors[1]["distance"]
