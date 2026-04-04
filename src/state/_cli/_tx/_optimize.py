import argparse
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def add_arguments_optimize(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, defaults to model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key in adata.obsm for input features (if None, uses adata.X).",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drugname_drugconc",
        help="Column in adata.obs for perturbation labels",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Path to the training run directory. Must contain config.yaml, var_dims.pkl, pert_onehot_map.pt, and "
            "batch_onehot_map.torch (legacy batch_onehot_map.pkl is also supported)."
        ),
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help="Column in adata.obs to group by. If omitted, tries config then common fallbacks.",
    )
    parser.add_argument(
        "--target-celltypes",
        type=str,
        required=True,
        help="Comma-separated list of target cell types. Use '__ALL__' when no cell type column exists.",
    )
    parser.add_argument(
        "--healthy-celltypes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of healthy/off-target cell types. Defaults to all remaining control cell types not "
            "listed in --target-celltypes."
        ),
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch column name in adata.obs. If omitted, tries config['data']['kwargs']['batch_col'] then fallbacks.",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=None,
        help="Override the control perturbation label. If omitted, read from config.",
    )
    parser.add_argument(
        "--candidate-perts",
        type=str,
        default=None,
        help="Optional comma-separated list of candidate perturbations to score.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional limit on the number of candidate perturbations considered after filtering.",
    )
    parser.add_argument(
        "--pair-mode",
        type=str,
        choices=["none", "topk", "all"],
        default="topk",
        help="Whether to evaluate additive pairs in addition to single drugs.",
    )
    parser.add_argument(
        "--pair-topk",
        type=int,
        default=16,
        help="When --pair-mode=topk, form pairs from the top K singles by score.",
    )
    parser.add_argument(
        "--max-set-len",
        type=int,
        default=None,
        help="Maximum number of control cells sampled per cell-type cohort. Defaults to the trained cell_set_len.",
    )
    parser.add_argument(
        "--healthy-weight",
        type=float,
        default=1.0,
        help="Penalty multiplier for healthy/off-target expression changes.",
    )
    parser.add_argument(
        "--continuous-steps",
        type=int,
        default=0,
        help=(
            "If >0, optimize continuous perturbation vectors seeded from the best discrete candidates, then retrieve "
            "nearest real compounds."
        ),
    )
    parser.add_argument(
        "--continuous-lr",
        type=float,
        default=5e-2,
        help="Learning rate for continuous perturbation-vector optimization.",
    )
    parser.add_argument(
        "--realism-weight",
        type=float,
        default=0.05,
        help="Penalty multiplier encouraging continuous vectors to stay near real compounds.",
    )
    parser.add_argument(
        "--optimize-pair",
        action="store_true",
        help="If set with --continuous-steps > 0, also optimize two additive perturbation vectors.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=5,
        help="Number of nearest real compounds to retrieve per optimized vector.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=50,
        help="Number of top rows to print in the CLI summary.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TSV path. Defaults to <input>_optimized.tsv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cohort sampling and optimization initialization.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )


def run_tx_optimize(args: argparse.Namespace):
    import os
    import pickle
    import warnings

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel
    from ...tx.optimization import (
        candidate_label,
        combine_additive_deltas,
        ensure_1d_float_tensor,
        realism_penalty,
        score_mean_deltas,
        topk_nearest_neighbors,
    )

    def split_csv_argument(value: Optional[str]) -> List[str]:
        if value is None:
            return []
        return [piece.strip() for piece in value.split(",") if piece.strip()]

    def load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def to_dense(mat):
        try:
            import scipy.sparse as sp

            if sp.issparse(mat):
                return mat.toarray()
        except Exception:
            pass
        return np.asarray(mat)

    def pick_first_present(d: "sc.AnnData", candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate and candidate in d.obs:
                return candidate
        return None

    def argmax_index_from_any(v, expected_dim: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        try:
            if torch.is_tensor(v):
                if v.ndim == 1:
                    if expected_dim is not None and v.numel() != expected_dim:
                        return None
                    return int(torch.argmax(v).item())
                return None
        except Exception:
            pass
        try:
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    if expected_dim is not None and v.size != expected_dim:
                        return None
                    return int(v.argmax())
                return None
        except Exception:
            pass
        if isinstance(v, (int, np.integer)):
            return int(v)
        return None

    def load_onehot_map(model_dir: str, basename: str):
        candidates = [
            f"{basename}.torch",
            f"{basename}.pt",
            f"{basename}.pkl",
        ]
        resolved_paths = [os.path.join(model_dir, name) for name in candidates]
        for path in resolved_paths:
            if not os.path.exists(path):
                continue
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    mapping = pickle.load(f)
            else:
                mapping = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(mapping, dict):
                raise TypeError(f"Expected dict in {path}, got {type(mapping).__name__}")
            return mapping, path, resolved_paths
        return None, None, resolved_paths

    def resolve_control_pert(cfg: dict) -> str:
        control_pert = args.control_pert
        if control_pert is None:
            try:
                control_pert = cfg["data"]["kwargs"]["control_pert"]
            except Exception:
                control_pert = None
        if control_pert is None and args.pert_col == "drugname_drugconc":
            control_pert = "[('DMSO_TF', 0.0, 'uM')]"
        if control_pert is None:
            control_pert = "non-targeting"
        return str(control_pert)

    def predict_mean_delta(cohort: dict, pert_vector: torch.Tensor) -> torch.Tensor:
        ctrl = cohort["ctrl_cell_emb"]
        pert_batch = pert_vector.view(1, -1).repeat(ctrl.shape[0], 1)
        batch = {
            "ctrl_cell_emb": ctrl,
            "pert_emb": pert_batch,
            "pert_name": [cohort["name"]] * ctrl.shape[0],
        }
        if cohort["batch_indices"] is not None:
            batch["batch"] = cohort["batch_indices"]

        outputs = model.forward(batch, padded=False)
        preds = outputs[0] if isinstance(outputs, tuple) else outputs
        return (preds - ctrl).mean(dim=0)

    def score_row_from_deltas(
        candidate_type: str, pert_a: str, pert_b: Optional[str], deltas: Dict[str, torch.Tensor]
    ) -> dict:
        target_deltas = {name: deltas[name] for name in target_cohorts.keys()}
        healthy_deltas = {name: deltas[name] for name in healthy_cohorts.keys()}
        score_parts = score_mean_deltas(target_deltas, healthy_deltas, healthy_weight=args.healthy_weight)
        return {
            "candidate_type": candidate_type,
            "candidate": candidate_label(pert_a, pert_b),
            "pert_a": pert_a,
            "pert_b": pert_b or "",
            "score": float(score_parts["score"].detach().cpu().item()),
            "target_efficacy": float(score_parts["target_efficacy"].detach().cpu().item()),
            "healthy_penalty": float(score_parts["healthy_penalty"].detach().cpu().item()),
        }

    def optimize_vectors(init_vectors: List[torch.Tensor]) -> dict:
        params = [torch.nn.Parameter(vec.detach().clone().to(device)) for vec in init_vectors]
        optimizer = torch.optim.Adam(params, lr=args.continuous_lr)
        history: List[dict] = []

        for step in range(int(args.continuous_steps)):
            optimizer.zero_grad()
            deltas: Dict[str, torch.Tensor] = {}
            for cohort_name, cohort in all_cohorts.items():
                component_deltas = [predict_mean_delta(cohort, param) for param in params]
                deltas[cohort_name] = combine_additive_deltas(*component_deltas)

            target_deltas = {name: deltas[name] for name in target_cohorts.keys()}
            healthy_deltas = {name: deltas[name] for name in healthy_cohorts.keys()}
            score_parts = score_mean_deltas(target_deltas, healthy_deltas, healthy_weight=args.healthy_weight)
            realism = realism_penalty(params, library_vectors)
            objective = score_parts["score"] - float(args.realism_weight) * realism
            loss = -objective
            loss.backward()
            optimizer.step()

            history.append(
                {
                    "step": step + 1,
                    "objective": float(objective.detach().cpu().item()),
                    "score": float(score_parts["score"].detach().cpu().item()),
                    "target_efficacy": float(score_parts["target_efficacy"].detach().cpu().item()),
                    "healthy_penalty": float(score_parts["healthy_penalty"].detach().cpu().item()),
                    "realism_penalty": float(realism.detach().cpu().item()),
                }
            )

        final_vectors = [param.detach().cpu() for param in params]
        final_deltas = {name: deltas[name].detach().cpu() for name in deltas}
        final_score_parts = score_mean_deltas(
            {name: final_deltas[name] for name in target_cohorts.keys()},
            {name: final_deltas[name] for name in healthy_cohorts.keys()},
            healthy_weight=args.healthy_weight,
        )
        final_realism = realism_penalty(final_vectors, library_vectors_cpu)
        return {
            "vectors": final_vectors,
            "deltas": final_deltas,
            "history": history,
            "score": float(final_score_parts["score"].detach().cpu().item()),
            "target_efficacy": float(final_score_parts["target_efficacy"].detach().cpu().item()),
            "healthy_penalty": float(final_score_parts["healthy_penalty"].detach().cpu().item()),
            "realism_penalty": float(final_realism.detach().cpu().item()),
        }

    if not args.quiet:
        print("==> STATE: tx optimize (phase-one selective combination search)")

    if args.max_set_len is not None and int(args.max_set_len) <= 0:
        raise ValueError("--max-set-len must be positive when provided")
    if int(args.continuous_steps) < 0:
        raise ValueError("--continuous-steps must be non-negative")
    if int(args.neighbor_k) <= 0:
        raise ValueError("--neighbor-k must be positive")

    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    control_pert = resolve_control_pert(cfg)

    if not args.quiet:
        print(f"Loaded config: {config_path}")
        print(f"Control perturbation: {control_pert}")

    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"Missing var_dims.pkl at {var_dims_path}")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    pert_dim = var_dims.get("pert_dim")
    batch_dim = var_dims.get("batch_dim", None)

    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_onehot_map_path):
        raise FileNotFoundError(f"Missing pert_onehot_map.pt at {pert_onehot_map_path}")
    pert_onehot_map: Dict[str, torch.Tensor] = torch.load(pert_onehot_map_path, map_location="cpu", weights_only=False)
    pert_name_lookup: Dict[str, object] = {str(k): k for k in pert_onehot_map.keys()}

    batch_onehot_map, loaded_batch_onehot_map_path, batch_onehot_map_candidates = load_onehot_map(
        args.model_dir, "batch_onehot_map"
    )
    if loaded_batch_onehot_map_path is not None and not args.quiet:
        print(f"Loaded batch one-hot map from: {loaded_batch_onehot_map_path}")

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        if not args.quiet:
            print(f"No --checkpoint given, using {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    cell_set_len = args.max_set_len if args.max_set_len is not None else getattr(model, "cell_sentence_len", 256)
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None

    if not args.quiet:
        print(f"Model device: {device}")
        print(f"Cohort cell cap: {cell_set_len}")
        print(f"Model uses batch encoder: {bool(uses_batch_encoder)}")

    adata = sc.read_h5ad(args.adata)
    if args.celltype_col is None:
        ct_from_cfg = None
        try:
            ct_from_cfg = cfg["data"]["kwargs"].get("cell_type_key", None)
        except Exception:
            pass
        args.celltype_col = pick_first_present(
            adata,
            candidates=[ct_from_cfg, "cell_type", "celltype", "cellType", "ctype", "celltype_col"]
            if ct_from_cfg
            else ["cell_type", "celltype", "cellType", "ctype", "celltype_col"],
        )

    if args.embed_key is None:
        X_in = to_dense(adata.X)
    else:
        if args.embed_key not in adata.obsm:
            raise KeyError(f"Embedding key '{args.embed_key}' not found in adata.obsm")
        X_in = np.asarray(adata.obsm[args.embed_key])

    if args.pert_col not in adata.obs:
        raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")

    pert_names_all = adata.obs[args.pert_col].astype(str).values
    ctl_mask = pert_names_all == control_pert
    if not bool(np.any(ctl_mask)):
        raise ValueError(
            f"No control cells found for perturbation '{control_pert}' in column '{args.pert_col}'. "
            "Optimization expects unperturbed/control cells as the input cohort."
        )

    if args.celltype_col and args.celltype_col in adata.obs:
        group_labels = adata.obs[args.celltype_col].astype(str).values
    else:
        group_labels = np.array(["__ALL__"] * adata.n_obs)

    target_celltypes = split_csv_argument(args.target_celltypes)
    if not target_celltypes:
        raise ValueError("--target-celltypes must include at least one value")

    control_group_labels = np.unique(group_labels[ctl_mask]).tolist()
    if args.healthy_celltypes:
        healthy_celltypes = split_csv_argument(args.healthy_celltypes)
    else:
        healthy_celltypes = [ct for ct in control_group_labels if ct not in set(target_celltypes)]

    overlap = sorted(set(target_celltypes) & set(healthy_celltypes))
    if overlap:
        healthy_celltypes = [ct for ct in healthy_celltypes if ct not in set(overlap)]
        if not args.quiet:
            preview = ", ".join(overlap)
            print(f"Warning: removing overlapping target/healthy groups from healthy set: {preview}")

    if args.batch_col is None:
        try:
            args.batch_col = cfg["data"]["kwargs"].get("batch_col", None)
        except Exception:
            args.batch_col = None

    batch_indices_all: Optional[np.ndarray] = None
    if uses_batch_encoder:
        batch_col = args.batch_col
        if batch_col is None:
            candidates = ["gem_group", "gemgroup", "batch", "donor", "plate", "experiment", "lane", "batch_id"]
            batch_col = next((candidate for candidate in candidates if candidate in adata.obs), None)
        if batch_col is not None and batch_col in adata.obs:
            raw_labels = adata.obs[batch_col].astype(str).values
            if batch_onehot_map is None:
                warnings.warn(
                    "Model has a batch encoder, but no batch one-hot map was found at any of: "
                    + ", ".join(batch_onehot_map_candidates)
                    + ". Batch info will be ignored; optimization scores may degrade."
                )
                uses_batch_encoder = False
            else:
                label_to_idx: Dict[str, int] = {}
                for key, value in batch_onehot_map.items():
                    idx = argmax_index_from_any(value, expected_dim=batch_dim)
                    if idx is not None:
                        label_to_idx[str(key)] = idx
                idxs = np.zeros(len(raw_labels), dtype=np.int64)
                misses = 0
                for i, label in enumerate(raw_labels):
                    if label in label_to_idx:
                        idxs[i] = label_to_idx[label]
                    else:
                        misses += 1
                        idxs[i] = 0
                if misses and not args.quiet:
                    print(
                        f"Warning: {misses} / {len(raw_labels)} batch labels were not found in the saved mapping; using index 0."
                    )
                batch_indices_all = idxs
        else:
            uses_batch_encoder = False
            if not args.quiet:
                print("Batch encoder present, but no batch column found; proceeding without batch indices.")

    rng = np.random.RandomState(args.seed)

    def build_cohorts(selected_groups: List[str]) -> Dict[str, dict]:
        cohorts: Dict[str, dict] = {}
        for group_name in selected_groups:
            group_mask = ctl_mask & (group_labels == group_name)
            indices = np.where(group_mask)[0]
            if len(indices) == 0:
                raise ValueError(
                    f"No control cells found for cell type '{group_name}'. Provide an AnnData with baseline cells for every cohort."
                )
            if len(indices) > cell_set_len:
                indices = np.sort(rng.choice(indices, size=cell_set_len, replace=False))
            ctrl_tensor = torch.tensor(X_in[indices, :], dtype=torch.float32, device=device)
            batch_tensor = None
            if uses_batch_encoder and batch_indices_all is not None:
                batch_tensor = torch.tensor(batch_indices_all[indices], dtype=torch.long, device=device)
            cohorts[group_name] = {
                "name": group_name,
                "ctrl_cell_emb": ctrl_tensor,
                "batch_indices": batch_tensor,
                "n_cells": int(ctrl_tensor.shape[0]),
            }
        return cohorts

    target_cohorts = build_cohorts(target_celltypes)
    healthy_cohorts = build_cohorts(healthy_celltypes) if healthy_celltypes else {}
    all_cohorts = {**target_cohorts, **healthy_cohorts}

    if not args.quiet:
        target_summary = ", ".join([f"{name}:{cohort['n_cells']}" for name, cohort in target_cohorts.items()])
        healthy_summary = ", ".join([f"{name}:{cohort['n_cells']}" for name, cohort in healthy_cohorts.items()])
        print(f"Target cohorts:  {target_summary}")
        print(f"Healthy cohorts: {healthy_summary if healthy_summary else '(none)'}")

    candidate_names = [name for name in pert_name_lookup.keys() if name != control_pert]
    if args.candidate_perts:
        requested = split_csv_argument(args.candidate_perts)
        missing = [name for name in requested if name not in pert_name_lookup]
        if missing and not args.quiet:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            print(
                f"Warning: {len(missing)} requested perturbations are not in the saved mapping and will be skipped ({preview})."
            )
        candidate_names = [name for name in requested if name in pert_name_lookup and name != control_pert]

    if args.max_candidates is not None:
        candidate_names = candidate_names[: int(args.max_candidates)]

    if not candidate_names:
        raise ValueError("No candidate perturbations remain after filtering")

    candidate_vectors: Dict[str, torch.Tensor] = {}
    for candidate_name in candidate_names:
        map_key = pert_name_lookup[candidate_name]
        candidate_vectors[candidate_name] = ensure_1d_float_tensor(
            pert_onehot_map[map_key],
            pert_dim=pert_dim,
            device=device,
        )

    library_names = list(candidate_vectors.keys())
    library_vectors = torch.stack([candidate_vectors[name] for name in library_names], dim=0)
    library_vectors_cpu = library_vectors.detach().cpu()

    single_rows: List[dict] = []
    single_delta_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    single_iter = tqdm(library_names, desc="Scoring singles", disable=args.quiet)
    with torch.no_grad():
        for candidate_name in single_iter:
            candidate_vector = candidate_vectors[candidate_name]
            deltas: Dict[str, torch.Tensor] = {}
            for cohort_name, cohort in all_cohorts.items():
                deltas[cohort_name] = predict_mean_delta(cohort, candidate_vector).detach().cpu()
            single_delta_cache[candidate_name] = deltas
            single_rows.append(score_row_from_deltas("single", candidate_name, None, deltas))

    single_rows.sort(key=lambda row: row["score"], reverse=True)

    pair_rows: List[dict] = []
    pair_seed_names: List[str] = []
    if args.pair_mode != "none" and len(single_rows) >= 2:
        if args.pair_mode == "topk":
            pair_seed_names = [row["pert_a"] for row in single_rows[: max(int(args.pair_topk), 0)]]
        else:
            pair_seed_names = [row["pert_a"] for row in single_rows]

        if len(pair_seed_names) >= 2:
            pair_total = len(pair_seed_names) * (len(pair_seed_names) - 1) // 2
            pair_iter = tqdm(
                combinations(pair_seed_names, 2),
                total=pair_total,
                desc="Scoring additive pairs",
                disable=args.quiet,
            )
            for pert_a, pert_b in pair_iter:
                combined_deltas = {
                    cohort_name: combine_additive_deltas(
                        single_delta_cache[pert_a][cohort_name],
                        single_delta_cache[pert_b][cohort_name],
                    )
                    for cohort_name in all_cohorts.keys()
                }
                pair_rows.append(score_row_from_deltas("pair", pert_a, pert_b, combined_deltas))
            pair_rows.sort(key=lambda row: row["score"], reverse=True)

    continuous_rows: List[dict] = []
    optimization_artifacts: dict = {
        "single": None,
        "pair": None,
    }

    if int(args.continuous_steps) > 0:
        if not args.quiet:
            print(f"Running continuous optimization for {args.continuous_steps} steps...")

        best_single_name = single_rows[0]["pert_a"]
        single_opt = optimize_vectors([candidate_vectors[best_single_name]])
        optimization_artifacts["single"] = single_opt
        continuous_rows.append(
            {
                "candidate_type": "optimized_single_raw",
                "candidate": "__continuous_single__",
                "pert_a": "",
                "pert_b": "",
                "score": single_opt["score"],
                "target_efficacy": single_opt["target_efficacy"],
                "healthy_penalty": single_opt["healthy_penalty"],
                "realism_penalty": single_opt["realism_penalty"],
                "seed_candidate": best_single_name,
                "neighbor_rank": 0,
                "retrieval_distance": 0.0,
            }
        )

        single_neighbors = topk_nearest_neighbors(
            single_opt["vectors"][0],
            library_vectors_cpu,
            library_names,
            k=args.neighbor_k,
        )
        for neighbor in single_neighbors:
            base_row = next(row for row in single_rows if row["pert_a"] == neighbor["name"])
            continuous_rows.append(
                {
                    **base_row,
                    "candidate_type": "optimized_single_neighbor",
                    "realism_penalty": single_opt["realism_penalty"],
                    "seed_candidate": best_single_name,
                    "neighbor_rank": neighbor["neighbor_rank"],
                    "retrieval_distance": neighbor["distance"],
                }
            )

        if args.optimize_pair and len(single_rows) >= 2:
            if pair_rows:
                init_pair = [pair_rows[0]["pert_a"], pair_rows[0]["pert_b"]]
            else:
                init_pair = [single_rows[0]["pert_a"], single_rows[1]["pert_a"]]

            pair_opt = optimize_vectors([candidate_vectors[init_pair[0]], candidate_vectors[init_pair[1]]])
            optimization_artifacts["pair"] = pair_opt
            continuous_rows.append(
                {
                    "candidate_type": "optimized_pair_raw",
                    "candidate": "__continuous_pair__",
                    "pert_a": "",
                    "pert_b": "",
                    "score": pair_opt["score"],
                    "target_efficacy": pair_opt["target_efficacy"],
                    "healthy_penalty": pair_opt["healthy_penalty"],
                    "realism_penalty": pair_opt["realism_penalty"],
                    "seed_candidate": candidate_label(init_pair[0], init_pair[1]),
                    "neighbor_rank": 0,
                    "retrieval_distance": 0.0,
                }
            )

            neighbors_a = topk_nearest_neighbors(
                pair_opt["vectors"][0], library_vectors_cpu, library_names, k=args.neighbor_k
            )
            neighbors_b = topk_nearest_neighbors(
                pair_opt["vectors"][1], library_vectors_cpu, library_names, k=args.neighbor_k
            )
            seen_pairs = set()
            for left, right in product(neighbors_a, neighbors_b):
                pert_a = str(left["name"])
                pert_b = str(right["name"])
                if pert_a == pert_b:
                    continue
                pair_key = tuple(sorted((pert_a, pert_b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                combined_deltas = {
                    cohort_name: combine_additive_deltas(
                        single_delta_cache[pair_key[0]][cohort_name],
                        single_delta_cache[pair_key[1]][cohort_name],
                    )
                    for cohort_name in all_cohorts.keys()
                }
                row = score_row_from_deltas("optimized_pair_neighbor", pair_key[0], pair_key[1], combined_deltas)
                row["realism_penalty"] = pair_opt["realism_penalty"]
                row["seed_candidate"] = candidate_label(init_pair[0], init_pair[1])
                row["neighbor_rank"] = min(int(left["neighbor_rank"]), int(right["neighbor_rank"]))
                row["retrieval_distance"] = float(left["distance"]) + float(right["distance"])
                continuous_rows.append(row)

    discrete_results = single_rows + pair_rows
    final_rows = discrete_results + continuous_rows
    results_df = pd.DataFrame(final_rows)
    if results_df.empty:
        raise RuntimeError("No optimization results were produced")

    if "realism_penalty" not in results_df.columns:
        results_df["realism_penalty"] = np.nan
    if "seed_candidate" not in results_df.columns:
        results_df["seed_candidate"] = ""
    if "neighbor_rank" not in results_df.columns:
        results_df["neighbor_rank"] = np.nan
    if "retrieval_distance" not in results_df.columns:
        results_df["retrieval_distance"] = np.nan

    results_df = results_df.sort_values(
        by=["score", "candidate_type", "candidate"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.adata).with_suffix("").with_name(Path(args.adata).stem + "_optimized.tsv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, sep="\t", index=False)

    if int(args.continuous_steps) > 0:
        sidecar_path = output_path.with_suffix(".continuous.pt")
        torch.save(optimization_artifacts, sidecar_path)
    else:
        sidecar_path = None

    print("\n=== Optimization complete ===")
    print(f"Candidates scored:   {len(single_rows)} singles")
    if pair_rows:
        print(f"Additive pairs:      {len(pair_rows)}")
    print(f"Saved ranking TSV:   {output_path}")
    if sidecar_path is not None:
        print(f"Saved raw vectors:   {sidecar_path}")

    preview_cols = [
        "rank",
        "candidate_type",
        "candidate",
        "score",
        "target_efficacy",
        "healthy_penalty",
        "retrieval_distance",
    ]
    preview_cols = [col for col in preview_cols if col in results_df.columns]
    print()
    print(results_df.loc[: max(int(args.top_results), 1) - 1, preview_cols].to_string(index=False))
