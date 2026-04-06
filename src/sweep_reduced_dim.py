#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.residual_simplex_regression import fit_joint_8_simplex_map, fit_per_process_maps


def weighted_mean(per_process: dict, key: str) -> float:
    num = 0.0
    den = 0.0
    for pdata in per_process.values():
        n = float(pdata["metrics"]["num_val"])
        num += n * float(pdata["metrics"][key])
        den += n
    return num / den


def main() -> None:
    dims = [4, 8, 12, 16, 24, 32, 48]
    seed = 0
    train_frac = 0.8

    out_dir = Path("artifacts/residual_simplex")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = torch.load(out_dir / "residual_dataset.pt", map_location="cpu")
    X_raw = ds.get("X_raw", ds["X"]).to(dtype=torch.float32)
    Y = ds["Y"]
    pid = ds["process_id"]
    sid = ds["sequence_id"]
    pos = ds["position"]

    # One PCA with max dimension, slice for each k.
    max_dim = min(max(dims), X_raw.shape[1])
    mean = X_raw.mean(dim=0, keepdim=True)
    xc = X_raw - mean
    _, _, v = torch.pca_lowrank(xc, q=max_dim, center=False)

    rows = []
    for k in dims:
        kk = min(k, v.shape[1])
        Xk = xc @ v[:, :kk]
        ds_k = {
            "X": Xk,
            "Y": Y,
            "process_id": pid,
            "sequence_id": sid,
            "position": pos,
        }

        per_process = fit_per_process_maps(ds_k, train_frac=train_frac, seed=seed)
        joint = fit_joint_8_simplex_map(ds_k, train_frac=train_frac, seed=seed)

        row = {
            "dim": kk,
            "perproc_mse_val_weighted": weighted_mean(per_process, "mse_val"),
            "perproc_mae_val_weighted": weighted_mean(per_process, "mae_val"),
            "perproc_soft_ce_val_weighted": weighted_mean(per_process, "soft_ce_val"),
            "joint_mse_val": float(joint["metrics"]["mse_val"]),
            "joint_mae_val": float(joint["metrics"]["mae_val"]),
            "joint_soft_ce_val": float(joint["metrics"]["soft_ce_val"]),
            "joint_block_acc_val": float(joint["metrics"]["process_block_acc_val"]),
            "joint_true_block_mass_mean_val": float(joint["metrics"]["true_process_block_mass_mean_val"]),
        }
        rows.append(row)
        print(
            f"dim={kk:2d} perproc_mse={row['perproc_mse_val_weighted']:.6f} "
            f"perproc_mae={row['perproc_mae_val_weighted']:.6f} "
            f"joint_mse={row['joint_mse_val']:.6f} joint_block_acc={row['joint_block_acc_val']:.6f}"
        )

    (out_dir / "reduced_dim_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "reduced_dim_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_dir / 'reduced_dim_sweep.json'}")
    print(f"Saved: {out_dir / 'reduced_dim_sweep.csv'}")


if __name__ == "__main__":
    main()
