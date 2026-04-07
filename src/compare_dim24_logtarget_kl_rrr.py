#!/usr/bin/env python3
"""Dim-24 PCA comparison using reduced-rank regression (RRR) in log-target space.

Targets:
- per-process 3-simplex KL (rank=3)
- joint 8-simplex KL (rank=8)
- block 3-simplex KL (rank=3)

For both final residual stream and after-first-layer residual stream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import (
    _build_joint_targets,
    _split_indices,
    load_checkpoint,
    load_rows,
    reduce_features_with_pca,
)


DIM = 24
TRAIN_FRAC = 0.8
SEED = 0
MAX_SEQS = 6000
EPS = 1e-8

DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}

OUT_JSON = Path("artifacts/residual_simplex/comparison_trained_vs_control_dim24_logtarget_kl_rrr.json")
OUT_JSON_LAYER1 = Path("artifacts/residual_simplex/comparison_trained_vs_control_dim24_logtarget_kl_rrr_layer1.json")


def build_residual_dataset_from_cache_key(model, cfg, rows: list[dict], cache_key: str) -> dict:
    xs = []
    ys = []
    process_ids = []
    sequence_ids = []
    positions = []

    model.eval()
    for seq_idx, row in enumerate(rows):
        tokens = [int(t) for t in row["tokens"]]
        pid = int(row["process_id"])
        if len(tokens) < cfg.seq_len:
            continue

        x_tokens = tokens[: cfg.seq_len]
        inp = torch.tensor(x_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, cache = model(inp, capture_residuals=True)
        residuals = cache[cache_key][0].detach().cpu()

        process = Mess3Process(alpha=float(row["alpha"]), x=float(row["x"]), dtype=torch.float64, device="cpu")
        traj = process.belief_trajectory(x_tokens)

        for pos in range(cfg.seq_len):
            xs.append(residuals[pos].to(dtype=torch.float32))
            ys.append(traj[pos + 1].to(dtype=torch.float32))
            process_ids.append(pid)
            sequence_ids.append(seq_idx)
            positions.append(pos)

    return {
        "X": torch.stack(xs, dim=0),
        "Y": torch.stack(ys, dim=0),
        "process_id": torch.tensor(process_ids, dtype=torch.long),
        "sequence_id": torch.tensor(sequence_ids, dtype=torch.long),
        "position": torch.tensor(positions, dtype=torch.long),
    }


def _mean_kl(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = EPS) -> float:
    y = y_true.clamp_min(eps)
    p = y_pred.clamp_min(eps)
    return float((y * (torch.log(y) - torch.log(p))).sum(dim=1).mean().item())


def _fit_rrr_with_bias(X: torch.Tensor, Y: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit reduced-rank multivariate regression Y ~ X with bias using OLS->RRR projection.

    Returns W, b such that prediction is X @ W.T + b.
    """
    n, d = X.shape
    k = Y.shape[1]
    r = max(1, min(rank, k))

    ones = torch.ones((n, 1), dtype=X.dtype, device=X.device)
    X1 = torch.cat([X, ones], dim=1)  # [n, d+1]

    # OLS solution in one shot for all outputs.
    B_ols = torch.linalg.lstsq(X1, Y).solution  # [d+1, k]
    Y_hat = X1 @ B_ols  # [n, k]

    # RRR projection in output space.
    _, _, vh = torch.linalg.svd(Y_hat, full_matrices=False)
    V = vh.T  # [k, k]
    Vr = V[:, :r]  # [k, r]
    B_rrr = B_ols @ Vr @ Vr.T  # [d+1, k]

    W = B_rrr[:-1, :].T.contiguous()  # [k, d]
    b = B_rrr[-1, :].contiguous()  # [k]
    return W, b


def _fit_and_eval_rrr_logtarget_kl(X: torch.Tensor, Y: torch.Tensor, split_seed: int, rank: int) -> float:
    tr, va = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=split_seed)
    Y_log = torch.log(Y + EPS)
    W, b = _fit_rrr_with_bias(X[tr], Y_log[tr], rank=rank)
    pred_va = torch.softmax(X[va] @ W.T + b, dim=1)
    return _mean_kl(Y[va], pred_va, eps=EPS)


def _per_process_kl_rrr(ds: dict) -> dict[str, float]:
    X = ds["X"]
    Y = ds["Y"]
    pid = ds["process_id"]
    out: dict[str, float] = {}
    for p in sorted(int(v) for v in pid.unique().tolist()):
        idx = torch.where(pid == p)[0]
        out[str(p)] = _fit_and_eval_rrr_logtarget_kl(X[idx], Y[idx], split_seed=SEED + p, rank=3)
    return out


def _joint_8_kl_rrr(ds: dict) -> float:
    X = ds["X"]
    Y_joint = _build_joint_targets(ds["Y"], ds["process_id"], num_processes=3)
    return _fit_and_eval_rrr_logtarget_kl(X, Y_joint, split_seed=SEED + 123, rank=8)


def _block_3_kl_rrr(ds: dict, rows: list[dict], seq_len: int) -> float:
    block_targets = build_block_targets(rows, seq_len=seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    return _fit_and_eval_rrr_logtarget_kl(Xb, Yb, split_seed=SEED + 777, rank=3)


def evaluate_case(ckpt_path: Path, rows: list[dict], cache_key: str) -> dict:
    model, cfg, _ = load_checkpoint(ckpt_path, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=cache_key)
    x_raw = ds["X"].to(torch.float32)
    x_red, _, _ = reduce_features_with_pca(x_raw, reduced_dim=DIM)

    ds_k = dict(ds)
    ds_k["X"] = x_red

    per_kl = _per_process_kl_rrr(ds_k)
    joint_kl = _joint_8_kl_rrr(ds_k)
    block_kl = _block_3_kl_rrr(ds_k, rows=rows, seq_len=cfg.seq_len)

    return {
        "dim": DIM,
        "rrr_rank_per_process": 3,
        "rrr_rank_joint_8": 8,
        "rrr_rank_block": 3,
        "per_process_kl_val": per_kl,
        "joint_8_simplex_kl_val": float(joint_kl),
        "block_3_simplex_kl_val": float(block_kl),
    }


def main() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)

    results_final = {name: evaluate_case(path, rows, cache_key="final_ln") for name, path in CASES.items()}
    results_layer1 = {name: evaluate_case(path, rows, cache_key="layer_0_after_mlp") for name, path in CASES.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results_final, indent=2), encoding="utf-8")
    OUT_JSON_LAYER1.write_text(json.dumps(results_layer1, indent=2), encoding="utf-8")

    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_JSON_LAYER1}")
    print("\nfinal_ln")
    for name in ("trained", "control_random_init"):
        print(name, results_final[name])
    print("\nlayer_0_after_mlp (after first layer)")
    for name in ("trained", "control_random_init"):
        print(name, results_layer1[name])


if __name__ == "__main__":
    main()

