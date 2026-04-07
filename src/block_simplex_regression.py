#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import (
    _fit_linear_map,
    _project_to_simplex,
    _split_indices,
    build_residual_dataset,
    load_checkpoint,
    load_rows,
    reduce_features_with_pca,
)


@dataclass
class Config:
    checkpoint_path: str = "artifacts/tiny_transformer.pt"
    dataset_path: str = "src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"
    out_dir: str = "artifacts/block_simplex"
    device: str = "cpu"

    max_sequences: int = 6000
    seq_len: int = 12
    reduced_dim: int = 24
    train_frac: float = 0.8
    seed: int = 0

    plot_max_points: int = 120000


def extract_specs(rows: List[dict]) -> List[Tuple[float, float]]:
    # Preserve semantic alignment: index i in returned list corresponds to process_id=i.
    pid_to_spec: Dict[int, Tuple[float, float]] = {}
    for r in rows:
        pid = int(r["process_id"])
        spec = (float(r["alpha"]), float(r["x"]))
        if pid in pid_to_spec and pid_to_spec[pid] != spec:
            raise ValueError(f"process_id {pid} has multiple specs: {pid_to_spec[pid]} vs {spec}")
        pid_to_spec[pid] = spec
    if set(pid_to_spec.keys()) != {0, 1, 2}:
        raise ValueError(f"Expected process_ids {{0,1,2}}, found {sorted(pid_to_spec.keys())}")
    return [pid_to_spec[0], pid_to_spec[1], pid_to_spec[2]]


def block_posteriors_for_tokens(tokens: List[int], specs: List[Tuple[float, float]]) -> torch.Tensor:
    """Return [T,3] block simplex posteriors p(process | x_1:t)."""
    procs = [Mess3Process(alpha=a, x=x, dtype=torch.float64, device="cpu") for a, x in specs]
    b_list = [p.stationary_distribution().clone() for p in procs]
    w = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)  # p(process)

    out: List[torch.Tensor] = []
    for tok in tokens:
        evid = torch.zeros(3, dtype=torch.float64)
        new_b: List[torch.Tensor] = []
        for i, p in enumerate(procs):
            b_new, e = p.update_belief(b_list[i], int(tok))
            new_b.append(b_new)
            evid[i] = e
        w = w * evid
        w = w / w.sum()
        b_list = new_b
        out.append(w.clone())
    return torch.stack(out, dim=0).to(dtype=torch.float32)


def build_block_targets(rows: List[dict], seq_len: int) -> Dict[Tuple[int, int], torch.Tensor]:
    specs = extract_specs(rows)
    targets: Dict[Tuple[int, int], torch.Tensor] = {}
    for seq_idx, row in enumerate(rows):
        tokens = [int(t) for t in row["tokens"]]
        if len(tokens) < seq_len:
            continue
        toks = tokens[:seq_len]
        block_traj = block_posteriors_for_tokens(toks, specs)  # [T,3]
        for pos in range(seq_len):
            targets[(seq_idx, pos)] = block_traj[pos]
    return targets


def align_xy(residual_ds: dict, block_targets: Dict[Tuple[int, int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = residual_ds["sequence_id"].tolist()
    pos = residual_ds["position"].tolist()
    y = []
    keep = []
    for i, (s, p) in enumerate(zip(seq, pos)):
        key = (int(s), int(p))
        if key in block_targets:
            y.append(block_targets[key])
            keep.append(i)
    idx = torch.tensor(keep, dtype=torch.long)
    X = residual_ds["X"][idx]
    Y = torch.stack(y, dim=0)
    return X, Y


def fit_block_map(X: torch.Tensor, Y: torch.Tensor, train_frac: float, seed: int) -> dict:
    tr, va = _split_indices(X.shape[0], train_frac=train_frac, seed=seed)
    W, b = _fit_linear_map(X[tr], Y[tr])  # [3,d], [3]

    pred_tr = _project_to_simplex(X[tr] @ W.T + b)
    pred_va = _project_to_simplex(X[va] @ W.T + b)

    mse_tr = ((pred_tr - Y[tr]) ** 2).mean().item()
    mse_va = ((pred_va - Y[va]) ** 2).mean().item()
    mae_va = (pred_va - Y[va]).abs().mean().item()
    ce_va = (-(Y[va] * torch.log(pred_va + 1e-8)).sum(dim=1)).mean().item()
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
        "val_indices": va,
    }


def plot_block_simplex(X: torch.Tensor, Y: torch.Tensor, fit: dict, out_dir: Path, max_points: int) -> None:
    import matplotlib.pyplot as plt
    from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices

    W, b = fit["W"], fit["b"]
    pred = _project_to_simplex(X @ W.T + b)

    if X.shape[0] > max_points:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(X.shape[0], generator=g)[:max_points]
        Y = Y[idx]
        pred = pred[idx]

    xy_t = barycentric_to_cartesian(Y.to(dtype=torch.float64))
    xy_p = barycentric_to_cartesian(pred.to(dtype=torch.float64))

    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[0:1]])

    # True
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), s=3, alpha=0.08, color="#1f77b4")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex ground truth")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_true.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_p[:, 0].numpy(), xy_p[:, 1].numpy(), s=3, alpha=0.08, color="#d62728")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex linear-map prediction")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_pred.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Overlay
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black")
    ax.scatter(xy_t[:, 0].numpy(), xy_t[:, 1].numpy(), s=3, alpha=0.05, color="#1f77b4", label="true")
    ax.scatter(xy_p[:, 0].numpy(), xy_p[:, 1].numpy(), s=3, alpha=0.05, color="#d62728", label="pred")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.set_title("Block-simplex overlay")
    ax.grid(alpha=0.2)
    fig.savefig(out_dir / "block_simplex_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, mcfg, _ = load_checkpoint(Path(cfg.checkpoint_path), torch.device(cfg.device))
    rows = load_rows(Path(cfg.dataset_path), cfg.max_sequences)
    residual_ds = build_residual_dataset(model, mcfg, rows, torch.device(cfg.device))

    X_raw = residual_ds["X"].to(torch.float32)
    X_red, pca_mean, pca_components = reduce_features_with_pca(X_raw, reduced_dim=cfg.reduced_dim)
    residual_ds["X_raw"] = X_raw
    residual_ds["X"] = X_red
    residual_ds["pca_mean"] = pca_mean
    residual_ds["pca_components"] = pca_components
    residual_ds["reduced_dim"] = torch.tensor([cfg.reduced_dim], dtype=torch.long)

    block_targets = build_block_targets(rows, seq_len=mcfg.seq_len)
    X, Y = align_xy(residual_ds, block_targets)

    fit = fit_block_map(X, Y, train_frac=cfg.train_frac, seed=cfg.seed)

    torch.save(
        {
            "W": fit["W"],
            "b": fit["b"],
            "metrics": fit["metrics"],
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "reduced_dim": cfg.reduced_dim,
        },
        out_dir / "block_simplex_linear_map.pt",
    )
    (out_dir / "block_simplex_metrics.json").write_text(
        json.dumps({"metrics": fit["metrics"], "W_shape": list(fit["W"].shape), "b_shape": list(fit["b"].shape)}, indent=2),
        encoding="utf-8",
    )
    torch.save({"X": X, "Y_block": Y}, out_dir / "block_simplex_dataset.pt")

    plot_block_simplex(X, Y, fit, out_dir=out_dir, max_points=cfg.plot_max_points)

    m = fit["metrics"]
    print("Saved:")
    print(f"  dataset: {out_dir / 'block_simplex_dataset.pt'}")
    print(f"  map:     {out_dir / 'block_simplex_linear_map.pt'}")
    print(f"  metrics: {out_dir / 'block_simplex_metrics.json'}")
    print(
        f"  mse_val={m['mse_val']:.6f} mae_val={m['mae_val']:.6f} "
        f"soft_ce_val={m['soft_ce_val']:.6f} argmax_acc_val={m['argmax_acc_val']:.6f}"
    )


if __name__ == "__main__":
    main()
