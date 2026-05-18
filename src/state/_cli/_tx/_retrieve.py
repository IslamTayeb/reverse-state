from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _add_shared_retrieve_arguments(parser: argparse.ArgumentParser, *, require_target_pert: bool):
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
        help="Input embedding key. If omitted, uses the saved training config's embed_key, else falls back to adata.X.",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=None,
        help="Perturbation column in adata.obs. If omitted, uses the saved training config then common fallbacks.",
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
    if require_target_pert:
        parser.add_argument(
            "--target-pert",
            type=str,
            required=True,
            help="Perturbation label to reverse-engineer from the AnnData query set.",
        )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help="Column in adata.obs to group by. If omitted, uses the saved training config then common fallbacks.",
    )
    parser.add_argument(
        "--celltypes",
        type=str,
        default=None,
        help="Optional comma-separated list of cell types/contexts to include. Defaults to all contexts containing the query perturbation.",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch column name in adata.obs. If omitted, uses the saved training config then common fallbacks.",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=None,
        help="Override the control perturbation label. If omitted, reads from config.",
    )
    parser.add_argument(
        "--candidate-perts",
        type=str,
        default=None,
        help="Optional comma-separated list of candidate perturbations to rank.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional limit on the number of candidate perturbations considered after filtering.",
    )
    parser.add_argument(
        "--match-metric",
        type=str,
        choices=["cosine", "pearson", "l2"],
        default="cosine",
        help="Similarity metric between the observed target delta and each predicted candidate delta.",
    )
    parser.add_argument(
        "--max-set-len",
        type=int,
        default=None,
        help="Maximum number of cells per forward pass. Defaults to the trained cell_set_len.",
    )
    parser.add_argument(
        "--max-cells-per-context",
        type=int,
        default=None,
        help="Optional cap on the number of query perturbation cells used per context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query/control sampling.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )


def add_arguments_retrieve(parser: argparse.ArgumentParser):
    _add_shared_retrieve_arguments(parser, require_target_pert=True)
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
        help="Output TSV path. Defaults to <input>_retrieved.tsv.",
    )


def add_arguments_retrieve_benchmark(parser: argparse.ArgumentParser):
    _add_shared_retrieve_arguments(parser, require_target_pert=False)
    parser.add_argument(
        "--query-perts",
        type=str,
        default=None,
        help="Optional comma-separated list of perturbations to benchmark. Defaults to shared observed/library perturbations.",
    )
    parser.add_argument(
        "--max-query-perts",
        type=int,
        default=None,
        help="Optional limit on the number of benchmark query perturbations.",
    )
    parser.add_argument(
        "--rankings-output",
        type=str,
        default=None,
        help="Optional TSV path for the full candidate rankings of each query perturbation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TSV path. Defaults to <input>_retrieve_benchmark.tsv.",
    )


def _split_csv_argument(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _load_runtime(args: argparse.Namespace) -> dict:
    import os
    import pickle
    import warnings

    import scanpy as sc
    import torch
    import yaml

    from ...tx.models.state_transition import StateTransitionPerturbationModel
    from ...tx.optimization import ensure_1d_float_tensor

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

    def resolve_control_pert(cfg: dict, pert_col: str) -> str:
        control_pert = args.control_pert
        if control_pert is None:
            control_pert = cfg.get("data", {}).get("kwargs", {}).get("control_pert")
        if control_pert is None and pert_col == "drugname_drugconc":
            control_pert = "[('DMSO_TF', 0.0, 'uM')]"
        if control_pert is None:
            control_pert = "non-targeting"
        return str(control_pert)

    if args.max_set_len is not None and int(args.max_set_len) <= 0:
        raise ValueError("--max-set-len must be positive when provided")
    if args.max_cells_per_context is not None and int(args.max_cells_per_context) <= 0:
        raise ValueError("--max-cells-per-context must be positive when provided")
    if args.max_candidates is not None and int(args.max_candidates) <= 0:
        raise ValueError("--max-candidates must be positive when provided")

    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    data_kwargs = cfg.get("data", {}).get("kwargs", {})

    if args.embed_key is None:
        args.embed_key = data_kwargs.get("embed_key")

    adata = sc.read_h5ad(args.adata)

    if args.pert_col is None:
        args.pert_col = data_kwargs.get("pert_col")
        if args.pert_col is None:
            args.pert_col = pick_first_present(adata, ["drugname_drugconc", "target_gene", "gene", "perturbation"])
    if args.pert_col is None:
        raise KeyError("Could not resolve a perturbation column. Pass --pert-col explicitly.")
    if args.pert_col not in adata.obs:
        raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")

    control_pert = resolve_control_pert(cfg, args.pert_col)

    if args.celltype_col is None:
        ct_from_cfg = data_kwargs.get("cell_type_key")
        celltype_candidates = [ct_from_cfg] if ct_from_cfg else []
        celltype_candidates += ["cell_type", "celltype", "cellType", "ctype", "celltype_col"]
        args.celltype_col = pick_first_present(adata, celltype_candidates)

    if args.batch_col is None:
        args.batch_col = data_kwargs.get("batch_col")

    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"Missing var_dims.pkl at {var_dims_path}")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    pert_dim = var_dims.get("pert_dim")
    batch_dim = var_dims.get("batch_dim")

    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_onehot_map_path):
        raise FileNotFoundError(f"Missing pert_onehot_map.pt at {pert_onehot_map_path}")
    pert_onehot_map: Dict[str, torch.Tensor] = torch.load(pert_onehot_map_path, map_location="cpu", weights_only=False)
    pert_name_lookup: Dict[str, object] = {str(k): k for k in pert_onehot_map.keys()}

    batch_onehot_map, loaded_batch_onehot_map_path, batch_onehot_map_candidates = load_onehot_map(
        args.model_dir, "batch_onehot_map"
    )

    checkpoint_path = args.checkpoint or os.path.join(args.model_dir, "checkpoints", "final.ckpt")
    model_name = str(cfg.get("model", {}).get("name", "state")).lower()
    if model_name not in {"state", "pertsets", "neuralot"}:
        raise NotImplementedError(
            "tx retrieve currently supports StateTransitionPerturbationModel-backed runs only "
            f"(got model={cfg.get('model', {}).get('name')!r})."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()

    cell_set_len = (
        int(args.max_set_len) if args.max_set_len is not None else int(getattr(model, "cell_sentence_len", 256))
    )
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None
    output_space = getattr(model, "output_space", data_kwargs.get("output_space", "gene"))
    counts_expected = (args.embed_key is not None and args.embed_key != "X_hvg" and output_space == "gene") or (
        args.embed_key is not None and output_space == "all"
    )

    if args.embed_key is None:
        X_in = to_dense(adata.X)
    else:
        if args.embed_key not in adata.obsm:
            raise KeyError(f"Embedding key '{args.embed_key}' not found in adata.obsm")
        X_in = np.asarray(adata.obsm[args.embed_key])

    if counts_expected:
        if output_space == "gene":
            if args.embed_key in {None, "X_hvg"}:
                score_reference_matrix = X_in
            elif "X_hvg" in adata.obsm:
                score_reference_matrix = np.asarray(adata.obsm["X_hvg"])
            else:
                raise KeyError(
                    "Retrieval requires adata.obsm['X_hvg'] when scoring a decoder-backed gene-space model "
                    f"trained with embed_key={args.embed_key!r}."
                )
        else:
            score_reference_matrix = to_dense(adata.X)
    else:
        score_reference_matrix = X_in

    pert_names_all = adata.obs[args.pert_col].astype(str).values
    ctl_mask = pert_names_all == control_pert
    if not bool(np.any(ctl_mask)):
        raise ValueError(
            f"No control cells found for perturbation '{control_pert}' in column '{args.pert_col}'. "
            "Retrieval expects control cells to define the query delta."
        )

    if args.celltype_col and args.celltype_col in adata.obs:
        group_labels = adata.obs[args.celltype_col].astype(str).values
    else:
        group_labels = np.array(["__ALL__"] * adata.n_obs)

    batch_indices_all: Optional[np.ndarray] = None
    if uses_batch_encoder:
        batch_col = args.batch_col
        if batch_col is None:
            batch_col = pick_first_present(
                adata, ["gem_group", "gemgroup", "batch", "donor", "plate", "experiment", "lane"]
            )
        if batch_col is not None and batch_col in adata.obs:
            raw_labels = adata.obs[batch_col].astype(str).values
            if batch_onehot_map is None:
                warnings.warn(
                    "Model has a batch encoder, but no batch one-hot map was found at any of: "
                    + ", ".join(batch_onehot_map_candidates)
                    + ". Batch info will be ignored; retrieval quality may degrade."
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
                args.batch_col = batch_col
        else:
            uses_batch_encoder = False
            if not args.quiet:
                print("Batch encoder present, but no batch column was found; proceeding without batch indices.")

    candidate_vectors: Dict[str, torch.Tensor] = {}
    for candidate_name, map_key in pert_name_lookup.items():
        if candidate_name == control_pert:
            continue
        candidate_vectors[candidate_name] = ensure_1d_float_tensor(
            pert_onehot_map[map_key],
            pert_dim=pert_dim,
            device=device,
        )

    observed_non_control_perts = sorted({name for name in pert_names_all.tolist() if name != control_pert})

    if not args.quiet:
        print("==> STATE: tx retrieve")
        print(f"Loaded config: {config_path}")
        print(f"Control perturbation: {control_pert}")
        if loaded_batch_onehot_map_path is not None:
            print(f"Loaded batch one-hot map from: {loaded_batch_onehot_map_path}")
        if args.checkpoint is None:
            print(f"No --checkpoint given, using {checkpoint_path}")
        print(f"Model device: {device}")
        print(f"Window size: {cell_set_len}")
        print(f"Model uses batch encoder: {bool(uses_batch_encoder)}")
        print(f"Retrieval score space: {'gene-counts' if counts_expected else 'model-output'}")

    return {
        "cfg": cfg,
        "adata": adata,
        "args": args,
        "model": model,
        "device": device,
        "cell_set_len": cell_set_len,
        "uses_batch_encoder": uses_batch_encoder,
        "pert_dim": pert_dim,
        "pert_names_all": pert_names_all,
        "control_pert": control_pert,
        "group_labels": group_labels,
        "all_control_indices": np.where(ctl_mask)[0],
        "batch_indices_all": batch_indices_all,
        "X_in": X_in,
        "score_reference_matrix": score_reference_matrix,
        "candidate_vectors": candidate_vectors,
        "observed_non_control_perts": observed_non_control_perts,
        "counts_expected": counts_expected,
        "rng": np.random.RandomState(args.seed),
        "quiet": bool(args.quiet),
    }


def _resolve_candidate_names(
    runtime: dict,
    requested: Optional[str],
    max_candidates: Optional[int],
    *,
    observed_only: bool = False,
) -> List[str]:
    candidate_names = sorted(runtime["candidate_vectors"].keys())
    if observed_only:
        observed = set(runtime["observed_non_control_perts"])
        candidate_names = [name for name in candidate_names if name in observed]

    if requested:
        requested_names = _split_csv_argument(requested)
        allowed = set(candidate_names)
        missing = [name for name in requested_names if name not in allowed]
        if missing and not runtime["quiet"]:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            print(
                f"Warning: {len(missing)} requested perturbations are unavailable after filtering and will be skipped ({preview})."
            )
        candidate_names = [name for name in requested_names if name in allowed]

    if max_candidates is not None:
        candidate_names = candidate_names[: int(max_candidates)]

    if not candidate_names:
        raise ValueError("No candidate perturbations remain after filtering")
    return candidate_names


def _resolve_query_names(
    runtime: dict, requested: Optional[str], candidate_names: List[str], max_query_perts: Optional[int]
) -> List[str]:
    candidate_set = set(candidate_names)
    if requested:
        requested_names = _split_csv_argument(requested)
        missing = [name for name in requested_names if name not in candidate_set]
        if missing and not runtime["quiet"]:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += ", ..."
            print(
                f"Warning: {len(missing)} requested benchmark perturbations are not in the candidate set and will be skipped ({preview})."
            )
        query_names = [name for name in requested_names if name in candidate_set]
    else:
        query_names = [name for name in runtime["observed_non_control_perts"] if name in candidate_set]

    if max_query_perts is not None:
        if int(max_query_perts) <= 0:
            raise ValueError("--max-query-perts must be positive when provided")
        query_names = query_names[: int(max_query_perts)]

    if not query_names:
        raise ValueError("No query perturbations remain after filtering")
    return query_names


def _group_control_indices(runtime: dict, group_name: str) -> tuple[np.ndarray, bool]:
    if group_name == "__ALL__":
        return runtime["all_control_indices"], False

    group_mask = runtime["group_labels"] == group_name
    group_controls = np.where(group_mask & (runtime["pert_names_all"] == runtime["control_pert"]))[0]
    if len(group_controls) > 0:
        return group_controls, False
    return runtime["all_control_indices"], True


def _build_query_contexts(
    runtime: dict,
    target_pert: str,
    selected_groups: List[str],
    *,
    max_cells_per_context: Optional[int],
    strict_groups: bool,
) -> Dict[str, dict]:
    import torch

    target_mask = runtime["pert_names_all"] == target_pert
    if not bool(np.any(target_mask)):
        raise ValueError(f"No cells found for target perturbation '{target_pert}'")

    if not selected_groups:
        selected_groups = np.unique(runtime["group_labels"][target_mask]).tolist()
    if not selected_groups:
        raise ValueError(f"No contexts found for target perturbation '{target_pert}'")

    contexts: Dict[str, dict] = {}
    for group_name in selected_groups:
        group_target_indices = np.where(target_mask & (runtime["group_labels"] == group_name))[0]
        if len(group_target_indices) == 0:
            if strict_groups:
                raise ValueError(f"No cells found for target perturbation '{target_pert}' in context '{group_name}'")
            continue

        if max_cells_per_context is not None and len(group_target_indices) > int(max_cells_per_context):
            group_target_indices = np.sort(
                runtime["rng"].choice(group_target_indices, size=int(max_cells_per_context), replace=False)
            )

        control_pool, used_global_controls = _group_control_indices(runtime, group_name)
        if len(control_pool) == 0:
            raise ValueError(
                f"No control cells available for context '{group_name}'. Retrieval requires control cells for the query delta."
            )
        if used_global_controls and not runtime["quiet"]:
            print(f"Warning: context '{group_name}' has no matched controls; using global controls instead.")

        windows = []
        target_delta_sum = None
        total_cells = 0
        for start in range(0, len(group_target_indices), runtime["cell_set_len"]):
            end = min(start + runtime["cell_set_len"], len(group_target_indices))
            target_window_idx = group_target_indices[start:end]
            window_size = len(target_window_idx)
            sampled_ctrl_idx = runtime["rng"].choice(control_pool, size=window_size, replace=True)

            ctrl_input = torch.tensor(
                runtime["X_in"][sampled_ctrl_idx, :], dtype=torch.float32, device=runtime["device"]
            )
            baseline_scores = torch.tensor(
                runtime["score_reference_matrix"][sampled_ctrl_idx, :],
                dtype=torch.float32,
                device=runtime["device"],
            )
            target_scores = torch.tensor(
                runtime["score_reference_matrix"][target_window_idx, :],
                dtype=torch.float32,
                device=runtime["device"],
            )
            batch_tensor = None
            if runtime["uses_batch_encoder"] and runtime["batch_indices_all"] is not None:
                batch_tensor = torch.tensor(
                    runtime["batch_indices_all"][target_window_idx],
                    dtype=torch.long,
                    device=runtime["device"],
                )

            current_delta_sum = (target_scores - baseline_scores).sum(dim=0)
            target_delta_sum = current_delta_sum if target_delta_sum is None else target_delta_sum + current_delta_sum
            total_cells += window_size
            windows.append(
                {
                    "ctrl_cell_emb": ctrl_input,
                    "score_reference": baseline_scores,
                    "batch_indices": batch_tensor,
                    "n_cells": window_size,
                }
            )

        contexts[group_name] = {
            "name": group_name,
            "target_pert": target_pert,
            "n_target_cells": total_cells,
            "target_delta": (target_delta_sum / total_cells).detach().cpu(),
            "windows": windows,
        }

    if not contexts:
        raise ValueError(f"No valid query contexts remain for target perturbation '{target_pert}'")
    return contexts


def _predict_context_delta(runtime: dict, context: dict, candidate_name: str, candidate_vector):
    import torch

    delta_sum = None
    total_cells = 0
    for window in context["windows"]:
        pert_batch = candidate_vector.view(1, -1).repeat(window["n_cells"], 1)
        batch = {
            "ctrl_cell_emb": window["ctrl_cell_emb"],
            "pert_emb": pert_batch,
            "pert_name": [candidate_name] * window["n_cells"],
        }
        if window["batch_indices"] is not None:
            batch["batch"] = window["batch_indices"]

        outputs = runtime["model"].predict_step(batch, batch_idx=0, padded=False)
        if runtime["counts_expected"] and outputs.get("pert_cell_counts_preds") is not None:
            preds = outputs["pert_cell_counts_preds"]
        else:
            preds = outputs["preds"]

        baseline = window["score_reference"]
        if preds.shape != baseline.shape:
            raise ValueError(
                "Prediction/reference shape mismatch during retrieval: "
                f"preds={tuple(preds.shape)} vs baseline={tuple(baseline.shape)}"
            )

        current_delta_sum = (preds - baseline).sum(dim=0)
        delta_sum = current_delta_sum if delta_sum is None else delta_sum + current_delta_sum
        total_cells += window["n_cells"]

    return (delta_sum / total_cells).detach().cpu()


def _score_candidates_for_query(
    runtime: dict,
    contexts: Dict[str, dict],
    candidate_names: List[str],
    *,
    match_metric: str,
    progress_desc: str,
) -> pd.DataFrame:
    import torch
    from tqdm import tqdm

    from ...tx.optimization import aggregate_context_scores, score_target_similarity

    total_target_cells = sum(context["n_target_cells"] for context in contexts.values())
    rows: List[dict] = []

    score_iter = tqdm(candidate_names, desc=progress_desc, disable=runtime["quiet"])
    with torch.no_grad():
        for candidate_name in score_iter:
            candidate_vector = runtime["candidate_vectors"][candidate_name]
            context_scores = {}
            for context_name, context in contexts.items():
                predicted_delta = _predict_context_delta(runtime, context, candidate_name, candidate_vector)
                context_scores[context_name] = score_target_similarity(
                    context["target_delta"],
                    predicted_delta,
                    metric=match_metric,
                )

            aggregate = aggregate_context_scores(context_scores)
            rows.append(
                {
                    "candidate": candidate_name,
                    "score": float(aggregate["score"].detach().cpu().item()),
                    "score_min": float(aggregate["score_min"].detach().cpu().item()),
                    "score_max": float(aggregate["score_max"].detach().cpu().item()),
                    "score_std": float(aggregate["score_std"].detach().cpu().item()),
                    "metric": match_metric,
                    "n_contexts": len(contexts),
                    "n_target_cells": total_target_cells,
                }
            )

    if not rows:
        raise RuntimeError("No retrieval results were produced")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by=["score", "candidate"], ascending=[False, True]).reset_index(drop=True)
    results_df.insert(0, "rank", np.arange(1, len(results_df) + 1))
    return results_df


def _default_output_path(input_path: str, suffix: str) -> Path:
    base = Path(input_path)
    return base.with_suffix("").with_name(base.stem + suffix)


def run_tx_retrieve(args: argparse.Namespace):
    runtime = _load_runtime(args)

    if args.target_pert == runtime["control_pert"]:
        raise ValueError("--target-pert must not be the control perturbation")

    selected_groups = _split_csv_argument(args.celltypes)
    contexts = _build_query_contexts(
        runtime,
        args.target_pert,
        selected_groups,
        max_cells_per_context=args.max_cells_per_context,
        strict_groups=bool(selected_groups),
    )
    candidate_names = _resolve_candidate_names(runtime, args.candidate_perts, args.max_candidates)

    if not runtime["quiet"]:
        context_summary = ", ".join([f"{name}:{context['n_target_cells']}" for name, context in contexts.items()])
        print(f"Target perturbation: {args.target_pert}")
        print(f"Query contexts: {context_summary}")
        print(f"Candidates scored: {len(candidate_names)}")

    results_df = _score_candidates_for_query(
        runtime,
        contexts,
        candidate_names,
        match_metric=args.match_metric,
        progress_desc=f"Ranking singles for {args.target_pert}",
    )
    results_df["is_target_label"] = results_df["candidate"] == str(args.target_pert)

    output_path = Path(args.output) if args.output else _default_output_path(args.adata, "_retrieved.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, sep="\t", index=False)

    print("\n=== Retrieval complete ===")
    print(f"Saved ranking TSV: {output_path}")
    if bool(results_df["is_target_label"].any()):
        true_row = results_df.loc[results_df["is_target_label"]].iloc[0]
        print(f"Target label rank: {int(true_row['rank'])}")

    preview_cols = ["rank", "candidate", "score", "score_min", "score_max", "metric", "is_target_label"]
    print()
    print(results_df.loc[: max(int(args.top_results), 1) - 1, preview_cols].to_string(index=False))


def run_tx_retrieve_benchmark(args: argparse.Namespace):
    from ...tx.optimization import compute_retrieval_metrics

    runtime = _load_runtime(args)
    candidate_names = _resolve_candidate_names(
        runtime,
        args.candidate_perts,
        args.max_candidates,
        observed_only=True,
    )
    query_names = _resolve_query_names(runtime, args.query_perts, candidate_names, args.max_query_perts)
    selected_groups = _split_csv_argument(args.celltypes)

    if not runtime["quiet"] and args.candidate_perts is None:
        print(
            "Benchmark candidate set defaults to perturbations observed in both the AnnData and saved mapping. "
            "Pass --candidate-perts for an explicit train-seen benchmark set."
        )

    summary_rows: List[dict] = []
    rankings_frames: List[pd.DataFrame] = []
    for query_name in query_names:
        try:
            contexts = _build_query_contexts(
                runtime,
                query_name,
                selected_groups,
                max_cells_per_context=args.max_cells_per_context,
                strict_groups=bool(selected_groups),
            )
        except ValueError as exc:
            if not runtime["quiet"]:
                print(f"Skipping '{query_name}': {exc}")
            continue

        results_df = _score_candidates_for_query(
            runtime,
            contexts,
            candidate_names,
            match_metric=args.match_metric,
            progress_desc=f"Ranking singles for {query_name}",
        )
        target_rows = results_df.loc[results_df["candidate"] == query_name]
        if target_rows.empty:
            if not runtime["quiet"]:
                print(f"Skipping '{query_name}': query label is not in the benchmark candidate set.")
            continue

        true_row = target_rows.iloc[0]
        top_row = results_df.iloc[0]
        summary_rows.append(
            {
                "query_pert": query_name,
                "true_rank": int(true_row["rank"]),
                "true_rank_percentile": float(true_row["rank"]) / float(len(results_df)),
                "true_score": float(true_row["score"]),
                "top_candidate": str(top_row["candidate"]),
                "top_score": float(top_row["score"]),
                "top_1_match": int(true_row["rank"] == 1),
                "top_5_match": int(true_row["rank"] <= 5),
                "num_candidates": int(len(results_df)),
                "n_contexts": int(true_row["n_contexts"]),
                "n_target_cells": int(true_row["n_target_cells"]),
                "metric": args.match_metric,
            }
        )

        if args.rankings_output:
            query_rankings = results_df.copy()
            query_rankings.insert(0, "query_pert", query_name)
            rankings_frames.append(query_rankings)

    if not summary_rows:
        raise RuntimeError("No benchmark results were produced")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["true_rank", "query_pert"], ascending=[True, True]).reset_index(drop=True)
    output_path = Path(args.output) if args.output else _default_output_path(args.adata, "_retrieve_benchmark.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, sep="\t", index=False)

    rankings_path = None
    if args.rankings_output:
        rankings_path = Path(args.rankings_output)
        rankings_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rankings_frames, ignore_index=True).to_csv(rankings_path, sep="\t", index=False)

    metrics = compute_retrieval_metrics(summary_df["true_rank"].tolist())

    print("\n=== Retrieval benchmark complete ===")
    print(f"Queries evaluated: {len(summary_df)}")
    print(f"Saved summary TSV: {output_path}")
    if rankings_path is not None:
        print(f"Saved rankings TSV: {rankings_path}")
    print(f"Top-1: {metrics['top_1']:.4f}")
    print(f"Top-5: {metrics['top_5']:.4f}")
    print(f"MRR:   {metrics['mrr']:.4f}")
    print(f"Median rank: {metrics['median_rank']:.2f}")
