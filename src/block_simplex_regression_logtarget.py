#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices
from src.residual_simplex_regression import (
    _fit_linear_map,
    _split_indices,
    build_residual_dataset,
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
RUNS = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}


def fit_block_map_log_targets(X: torch.Tensor, Y: torch.Tensor) -> dict:
    tr, va = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=SEED)
    Y_log = torch.log(Y + EPS)
    W, b = _fit_linear_map(X[tr], Y_log[tr])  # map into log-prob space

    pred_tr = torch.softmax(X[tr] @ W.T + b, dim=1)
    pred_va = torch.softmax(X[va] @ W.T + b, dim=1)

    mse_tr = ((pred_tr - Y[tr]) ** 2).mean().item()
    mse_va = ((pred_va - Y[va]) ** 2).mean().item()
    mae_va = (pred_va - Y[va]).abs().mean().item()
    ce_va = (-(Y[va] * torch.log(pred_va + EPS)).sum(dim=1)).mean().item()
    acc_va = (pred_va.argmax(dim=1) == Y[va].argmax(dim=1)).float().mean().item()

    return {
        "W": W,
        "b": b,
        "metrics": {
            "num_samples": int(X.shape[0]),
            "num_train": int(tr.shape[0]),
            "num_val": int(va.shape[0]),
            "mse_train": mse_tr,
            "mse_val": mse_va,
            "mae_val": mae_va,
            "soft_ce_val": ce_va,
            "argmax_acc_val": acc_va,
        },
    }


def plot_block_simplex_true_pred(Y_true: torch.Tensor, Y_pred: torch.Tensor, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[0:1]])

    xy_t = barycentric_to_cartesian(Y_true.to(dtype=torch.float64))
    xy_p = barycentric_to_cartesian(Y_pred.to(dtype=torch.float64))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), s=3, alpha=0.08, color="#1f77b4")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex ground truth")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_true.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_p[:, 0].numpy(), xy_p[:, 1].numpy(), s=3, alpha=0.08, color="#d62728")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex prediction (log-target)")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_pred.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), s=3, alpha=0.05, color="#1f77b4", label="true")
    ax.scatter(xy_p[:, 0].numpy(), xy_p[:, 1].numpy(), s=3, alpha=0.05, color="#d62728", label="pred")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex overlay (log-target)")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one(case_name: str, ckpt: Path) -> dict:
    out_dir = Path("artifacts") / f"block_simplex_logtarget_{case_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, mcfg, _ = load_checkpoint(ckpt, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    residual_ds = build_residual_dataset(model, mcfg, rows, torch.device("cpu"))

    X_raw = residual_ds["X"].float()
    X_red, pca_mean, pca_components = reduce_features_with_pca(X_raw, reduced_dim=DIM)
    residual_ds["X"] = X_red

    block_targets = build_block_targets(rows, seq_len=mcfg.seq_len)
    X, Y = align_xy(residual_ds, block_targets)

    fit = fit_block_map_log_targets(X, Y)
    pred_all = torch.softmax(X @ fit["W"].T + fit["b"], dim=1)

    plot_block_simplex_true_pred(Y, pred_all, out_dir)

    torch.save({"X": X, "Y_block": Y}, out_dir / "block_simplex_dataset.pt")
    torch.save(
        {
            "W": fit["W"],
            "b": fit["b"],
            "metrics": fit["metrics"],
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "reduced_dim": DIM,
        },
        out_dir / "block_simplex_linear_map.pt",
    )
    (out_dir / "block_simplex_metrics.json").write_text(
        json.dumps(
            {"metrics": fit["metrics"], "W_shape": list(fit["W"].shape), "b_shape": list(fit["b"].shape)},
            indent=2,
        ),
        encoding="utf-8",
    )

    m = fit["metrics"]
    print(f"\n{case_name} (log-target):")
    print(f"  mse_val={m['mse_val']:.6f}")
    print(f"  mae_val={m['mae_val']:.6f}")
    print(f"  soft_ce_val={m['soft_ce_val']:.6f}")
    print(f"  argmax_acc_val={m['argmax_acc_val']:.6f}")
    print(f"  saved {out_dir / 'block_simplex_metrics.json'}")
    return m


def main() -> None:
    summary = {}
    for name, ckpt in RUNS.items():
        summary[name] = run_one(name, ckpt)

    out = Path("artifacts/block_simplex_logtarget_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()

