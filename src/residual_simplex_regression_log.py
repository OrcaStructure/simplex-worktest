#!/usr/bin/env python3
"""Residual-to-simplex regression variant with log-probability targets.

This variant keeps residual features X unchanged and applies log to targets:
  Y_log = log(Y + eps)
where Y are simplex probability targets.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.residual_simplex_regression import (
    _fit_linear_map,
    _split_indices,
    _build_joint_targets,
    build_residual_dataset,
    load_checkpoint,
    load_rows,
)


@dataclass
class PipelineConfigLog:
    checkpoint_path: str = "artifacts/tiny_transformer.pt"
    dataset_path: str = "src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"
    out_dir: str = "artifacts/residual_simplex_log"
    device: str = "cpu"

    max_sequences: int = 6000
    train_frac: float = 0.8
    seed: int = 0

    plot_sequences_per_process_linear: int = 200
    plot_sequences_per_process_ground_truth: int = 3000


def _softmax_probs(z: torch.Tensor) -> torch.Tensor:
    return torch.softmax(z, dim=-1)


def fit_per_process_maps_log_targets(
    residual_ds: dict,
    train_frac: float,
    seed: int,
    eps: float = 1e-8,
) -> dict:
    X = residual_ds["X"]
    Y = residual_ds["Y"]
    pid = residual_ds["process_id"]

    out = {}
    for p in sorted(int(v) for v in pid.unique().tolist()):
        idx = torch.where(pid == p)[0]
        Xp = X[idx]
        Yp = Y[idx]
        Yp_log = torch.log(Yp + eps)

        tr, va = _split_indices(Xp.shape[0], train_frac=train_frac, seed=seed + p)
        W, b = _fit_linear_map(Xp[tr], Yp_log[tr])  # fit in log-prob space

        pred_train_log = Xp[tr] @ W.T + b
        pred_val_log = Xp[va] @ W.T + b
        pred_train = _softmax_probs(pred_train_log)
        pred_val = _softmax_probs(pred_val_log)

        mse_train = torch.mean((pred_train - Yp[tr]) ** 2).item()
        mse_val = torch.mean((pred_val - Yp[va]) ** 2).item()
        mae_val = torch.mean(torch.abs(pred_val - Yp[va])).item()
        ce_val = (-(Yp[va] * torch.log(pred_val + eps)).sum(dim=1)).mean().item()

        out[p] = {
            "W": W,
            "b": b,
            "metrics": {
                "num_samples": int(Xp.shape[0]),
                "num_train": int(tr.shape[0]),
                "num_val": int(va.shape[0]),
                "mse_train": mse_train,
                "mse_val": mse_val,
                "mae_val": mae_val,
                "soft_ce_val": ce_val,
            },
        }
    return out


def fit_joint_8_simplex_map_log_targets(
    residual_ds: dict,
    train_frac: float,
    seed: int,
    eps: float = 1e-8,
) -> dict:
    X = residual_ds["X"]
    Y_local = residual_ds["Y"]
    pid = residual_ds["process_id"]
    Y_joint = _build_joint_targets(Y_local, pid, num_processes=3)
    Y_joint_log = torch.log(Y_joint + eps)

    tr, va = _split_indices(X.shape[0], train_frac=train_frac, seed=seed + 123)
    W, b = _fit_linear_map(X[tr], Y_joint_log[tr])

    pred_train = _softmax_probs(X[tr] @ W.T + b)
    pred_val = _softmax_probs(X[va] @ W.T + b)

    y_tr = Y_joint[tr]
    y_va = Y_joint[va]

    mse_train = torch.mean((pred_train - y_tr) ** 2).item()
    mse_val = torch.mean((pred_val - y_va) ** 2).item()
    mae_val = torch.mean(torch.abs(pred_val - y_va)).item()
    ce_val = (-(y_va * torch.log(pred_val + eps)).sum(dim=1)).mean().item()

    block_val = pred_val.view(pred_val.shape[0], 3, 3).sum(dim=2)
    pred_pid = torch.argmax(block_val, dim=1)
    true_pid = pid[va]
    block_acc = (pred_pid == true_pid).float().mean().item()
    true_block_mass = torch.gather(block_val, dim=1, index=true_pid.unsqueeze(1)).squeeze(1)
    true_block_mass_mean = true_block_mass.mean().item()

    return {
        "W": W,
        "b": b,
        "metrics": {
            "num_samples": int(X.shape[0]),
            "num_train": int(tr.shape[0]),
            "num_val": int(va.shape[0]),
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mae_val": mae_val,
            "soft_ce_val": ce_val,
            "process_block_acc_val": block_acc,
            "true_process_block_mass_mean_val": true_block_mass_mean,
        },
    }


def save_per_process_metrics_json(per_process: dict, out_path: Path) -> None:
    payload = {
        str(pid): {
            "metrics": data["metrics"],
            "W_shape": list(data["W"].shape),
            "b_shape": list(data["b"].shape),
        }
        for pid, data in per_process.items()
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_true_and_pred_points_log_target(
    residual_ds: dict,
    per_process: dict,
    out_dir: Path,
    max_sequences_per_process_linear: int,
    max_sequences_per_process_ground_truth: int,
) -> None:
    import matplotlib.pyplot as plt
    from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices

    X = residual_ds["X"]
    Y = residual_ds["Y"]
    pid = residual_ds["process_id"]
    sid = residual_ds["sequence_id"]
    pos = residual_ds["position"]

    verts = simplex_vertices()
    triangle = torch.vstack([verts, verts[0:1]])

    for p, pdata in per_process.items():
        W = pdata["W"]
        b = pdata["b"]

        seq_ids_all = torch.unique(sid[pid == p]).tolist()
        seq_ids_true = seq_ids_all[:max_sequences_per_process_ground_truth]
        seq_ids_pred = seq_ids_all[:max_sequences_per_process_linear]

        fig_true, ax_true = plt.subplots(figsize=(7, 6))
        fig_pred, ax_pred = plt.subplots(figsize=(7, 6))
        for ax in (ax_true, ax_pred):
            ax.plot(triangle[:, 0].numpy(), triangle[:, 1].numpy(), color="black", linewidth=1.5)

        for seq_id in seq_ids_true:
            idx = torch.where((pid == p) & (sid == seq_id))[0]
            order = torch.argsort(pos[idx])
            idx = idx[order]
            y_true = Y[idx]
            xy_true = barycentric_to_cartesian(y_true.to(dtype=torch.float64))
            ax_true.scatter(xy_true[:, 0].numpy(), xy_true[:, 1].numpy(), color="#1f77b4", alpha=0.07, s=6)

        for seq_id in seq_ids_pred:
            idx = torch.where((pid == p) & (sid == seq_id))[0]
            order = torch.argsort(pos[idx])
            idx = idx[order]
            y_pred = _softmax_probs(X[idx] @ W.T + b)
            xy_pred = barycentric_to_cartesian(y_pred.to(dtype=torch.float64))
            ax_pred.scatter(xy_pred[:, 0].numpy(), xy_pred[:, 1].numpy(), color="#d62728", alpha=0.25, s=8)

        for ax, title in (
            (ax_true, f"Process {p}: ground-truth points"),
            (ax_pred, f"Process {p}: log-target linear-map points"),
        ):
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xlabel("simplex x")
            ax.set_ylabel("simplex y")
            ax.grid(alpha=0.2)

        fig_true.savefig(out_dir / f"process_{p}_ground_truth.png", dpi=180, bbox_inches="tight")
        fig_pred.savefig(out_dir / f"process_{p}_linear_map.png", dpi=180, bbox_inches="tight")
        plt.close(fig_true)
        plt.close(fig_pred)


def main() -> None:
    cfg = PipelineConfigLog()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model, model_cfg, _ = load_checkpoint(Path(cfg.checkpoint_path), device)
    rows = load_rows(Path(cfg.dataset_path), cfg.max_sequences)

    residual_ds = build_residual_dataset(model, model_cfg, rows, device)
    residual_ds["Y_log"] = torch.log(residual_ds["Y"] + 1e-8)
    torch.save(residual_ds, out_dir / "residual_dataset_log.pt")

    per_process = fit_per_process_maps_log_targets(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)
    joint_map = fit_joint_8_simplex_map_log_targets(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)

    params_to_save = {
        int(pid): {
            "W": pdata["W"],
            "b": pdata["b"],
            "metrics": pdata["metrics"],
        }
        for pid, pdata in per_process.items()
    }
    torch.save(params_to_save, out_dir / "linear_maps_per_process_log.pt")
    save_per_process_metrics_json(per_process, out_dir / "linear_maps_metrics_log.json")

    torch.save(joint_map, out_dir / "linear_map_joint_8_simplex_log.pt")
    (out_dir / "joint_8_simplex_metrics_log.json").write_text(
        json.dumps(
            {
                "metrics": joint_map["metrics"],
                "W_shape": list(joint_map["W"].shape),
                "b_shape": list(joint_map["b"].shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_true_and_pred_points_log_target(
        residual_ds,
        per_process,
        out_dir=out_dir,
        max_sequences_per_process_linear=cfg.plot_sequences_per_process_linear,
        max_sequences_per_process_ground_truth=cfg.plot_sequences_per_process_ground_truth,
    )

    ylog = residual_ds["Y_log"]
    print("Saved (log target variant):")
    print(f"  residual dataset: {out_dir / 'residual_dataset_log.pt'}")
    print(f"  linear maps:      {out_dir / 'linear_maps_per_process_log.pt'}")
    print(f"  metrics json:     {out_dir / 'linear_maps_metrics_log.json'}")
    print(f"  joint 8-simplex:  {out_dir / 'linear_map_joint_8_simplex_log.pt'}")
    print(f"  joint metrics:    {out_dir / 'joint_8_simplex_metrics_log.json'}")
    print(f"  Y(log) stats: min={float(ylog.min()):.6f} max={float(ylog.max()):.6f} mean={float(ylog.mean()):.6f}")

    for pid, pdata in sorted(per_process.items()):
        m = pdata["metrics"]
        print(
            f"  process {pid}: n={m['num_samples']} mse_val={m['mse_val']:.6f} "
            f"mae_val={m['mae_val']:.6f} soft_ce_val={m['soft_ce_val']:.6f}"
        )
    jm = joint_map["metrics"]
    print(
        "  joint(8-simplex): "
        f"mse_val={jm['mse_val']:.6f} mae_val={jm['mae_val']:.6f} "
        f"soft_ce_val={jm['soft_ce_val']:.6f} "
        f"process_block_acc_val={jm['process_block_acc_val']:.6f} "
        f"true_process_block_mass_mean_val={jm['true_process_block_mass_mean_val']:.6f}"
    )


if __name__ == "__main__":
    main()
