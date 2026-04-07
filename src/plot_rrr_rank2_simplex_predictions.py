#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.mess3 import Mess3Process
from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices
from src.residual_simplex_regression import _split_indices, load_checkpoint, load_rows, reduce_features_with_pca


# Hardcoded config
DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
MAX_SEQS = 6000
PCA_DIM = 24
RANK_3WAY = 2
TRAIN_FRAC = 0.8
SEED = 0
EPS = 1e-8
MAX_POINTS_PER_PLOT = 120000

CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}
CACHE_KEYS = {
    "final_ln": "final_ln",
    "layer1": "layer_0_after_mlp",
}
OUT_DIR = Path("artifacts/residual_simplex/rrr_rank2_simplex_plots")


def build_residual_dataset_from_cache_key(model, cfg, rows: List[dict], cache_key: str) -> dict:
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
        residuals = cache[cache_key][0].detach().cpu()  # [T, d_model]

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


def fit_rrr_with_bias(X: torch.Tensor, Y: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n, d = X.shape
    k = Y.shape[1]
    r = max(1, min(rank, k))

    ones = torch.ones((n, 1), dtype=X.dtype, device=X.device)
    X1 = torch.cat([X, ones], dim=1)  # [n, d+1]
    B_ols = torch.linalg.lstsq(X1, Y).solution  # [d+1, k]
    Y_hat = X1 @ B_ols
    _, _, vh = torch.linalg.svd(Y_hat, full_matrices=False)
    Vr = vh.T[:, :r]
    B_rrr = B_ols @ Vr @ Vr.T
    W = B_rrr[:-1, :].T.contiguous()
    b = B_rrr[-1, :].contiguous()
    return W, b


def fit_predict_rrr_prob(X: torch.Tensor, Y_prob: torch.Tensor, rank: int, split_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tr, _ = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=split_seed)
    Y_log = torch.log(Y_prob + EPS)
    W, b = fit_rrr_with_bias(X[tr], Y_log[tr], rank=rank)
    pred = torch.softmax(X @ W.T + b, dim=1)
    return Y_prob, pred


def subsample_pair(Y_true: torch.Tensor, Y_pred: torch.Tensor, max_points: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = Y_true.shape[0]
    if n <= max_points:
        return Y_true, Y_pred
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_points]
    return Y_true[idx], Y_pred[idx]


def plot_true_pred(Y_true: torch.Tensor, Y_pred: torch.Tensor, title_true: str, title_pred: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[:1]])
    xy_t = barycentric_to_cartesian(Y_true.to(torch.float64))
    xy_p = barycentric_to_cartesian(Y_pred.to(torch.float64))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax in axes:
        ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black", linewidth=1.2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("simplex x")
        ax.set_ylabel("simplex y")

    axes[0].scatter(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), s=2.0, alpha=0.08, color="#1f77b4")
    axes[0].set_title(title_true)

    axes[1].scatter(xy_p[:, 0].numpy(), xy_p[:, 1].numpy(), s=2.0, alpha=0.08, color="#d62728")
    axes[1].set_title(title_pred)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_case(case_name: str, ckpt: Path, rows: List[dict], cache_tag: str, cache_key: str) -> List[Path]:
    model, cfg, _ = load_checkpoint(ckpt, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=cache_key)
    X_red, _, _ = reduce_features_with_pca(ds["X"].float(), reduced_dim=PCA_DIM)
    ds["X"] = X_red

    out_paths: List[Path] = []

    # Per-process 2-simplex plots (rank=2 RRR in log-target space)
    pid = ds["process_id"]
    for p in (0, 1, 2):
        idx = torch.where(pid == p)[0]
        Y_true, Y_pred = fit_predict_rrr_prob(ds["X"][idx], ds["Y"][idx], rank=RANK_3WAY, split_seed=SEED + p)
        Yt, Yp = subsample_pair(Y_true, Y_pred, MAX_POINTS_PER_PLOT, seed=SEED + p * 11)
        out = OUT_DIR / f"{case_name}_{cache_tag}_process_{p}_true_vs_pred.png"
        plot_true_pred(
            Yt,
            Yp,
            title_true=f"Process {p} ground truth",
            title_pred=f"Process {p} RRR rank={RANK_3WAY} prediction",
            out_path=out,
        )
        out_paths.append(out)

    # Block-simplex plot (also 3-way, rank=2)
    block_targets = build_block_targets(rows, seq_len=cfg.seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    Y_true_b, Y_pred_b = fit_predict_rrr_prob(Xb, Yb, rank=RANK_3WAY, split_seed=SEED + 777)
    Ytb, Ypb = subsample_pair(Y_true_b, Y_pred_b, MAX_POINTS_PER_PLOT, seed=SEED + 999)
    out_b = OUT_DIR / f"{case_name}_{cache_tag}_block_true_vs_pred.png"
    plot_true_pred(
        Ytb,
        Ypb,
        title_true="Block-simplex ground truth",
        title_pred=f"Block-simplex RRR rank={RANK_3WAY} prediction",
        out_path=out_b,
    )
    out_paths.append(out_b)

    return out_paths


def main() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths: List[Path] = []
    for case_name, ckpt in CASES.items():
        for cache_tag, cache_key in CACHE_KEYS.items():
            all_paths.extend(evaluate_case(case_name, ckpt, rows, cache_tag, cache_key))

    print(f"Saved {len(all_paths)} plots to {OUT_DIR}")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

