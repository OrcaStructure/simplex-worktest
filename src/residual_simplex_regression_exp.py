#!/usr/bin/env python3
"""Residual-to-simplex regression variant with exp-transformed features.

This is an experiment variant of residual_simplex_regression.py where we apply:
  X_exp = exp(clamp(X, -20, 20))
before fitting linear maps.
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
    build_residual_dataset,
    fit_joint_8_simplex_map,
    fit_per_process_maps,
    load_checkpoint,
    load_rows,
    plot_true_and_pred_trajectories_separately,
    save_metrics_json,
)


@dataclass
class PipelineConfigExp:
    checkpoint_path: str = "artifacts/tiny_transformer.pt"
    dataset_path: str = "src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"
    out_dir: str = "artifacts/residual_simplex_exp"
    device: str = "cpu"

    max_sequences: int = 6000
    train_frac: float = 0.8
    seed: int = 0

    plot_sequences_per_process_linear: int = 200
    plot_sequences_per_process_ground_truth: int = 3000


def exp_transform_features(X: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(X, min=-20.0, max=20.0))


def main() -> None:
    cfg = PipelineConfigExp()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model, model_cfg, _ = load_checkpoint(Path(cfg.checkpoint_path), device)
    rows = load_rows(Path(cfg.dataset_path), cfg.max_sequences)

    residual_ds = build_residual_dataset(model, model_cfg, rows, device)
    residual_ds["X_raw"] = residual_ds["X"].clone()
    residual_ds["X"] = exp_transform_features(residual_ds["X"])
    torch.save(residual_ds, out_dir / "residual_dataset_exp.pt")

    per_process = fit_per_process_maps(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)
    joint_map = fit_joint_8_simplex_map(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)

    params_to_save = {
        int(pid): {
            "W": pdata["W"],
            "b": pdata["b"],
            "metrics": pdata["metrics"],
        }
        for pid, pdata in per_process.items()
    }
    torch.save(params_to_save, out_dir / "linear_maps_per_process_exp.pt")
    save_metrics_json(per_process, out_dir / "linear_maps_metrics_exp.json")

    torch.save(joint_map, out_dir / "linear_map_joint_8_simplex_exp.pt")
    (out_dir / "joint_8_simplex_metrics_exp.json").write_text(
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

    plot_true_and_pred_trajectories_separately(
        residual_ds,
        per_process,
        out_dir=out_dir,
        max_sequences_per_process_linear=cfg.plot_sequences_per_process_linear,
        max_sequences_per_process_ground_truth=cfg.plot_sequences_per_process_ground_truth,
    )

    x = residual_ds["X"]
    print("Saved (exp feature variant):")
    print(f"  residual dataset: {out_dir / 'residual_dataset_exp.pt'}")
    print(f"  linear maps:      {out_dir / 'linear_maps_per_process_exp.pt'}")
    print(f"  metrics json:     {out_dir / 'linear_maps_metrics_exp.json'}")
    print(f"  joint 8-simplex:  {out_dir / 'linear_map_joint_8_simplex_exp.pt'}")
    print(f"  joint metrics:    {out_dir / 'joint_8_simplex_metrics_exp.json'}")
    print(f"  X(exp) stats: min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f}")

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

