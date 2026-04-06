#!/usr/bin/env python3
"""Streamlined comparison: trained vs random-init, dim=24, log-target regression.

Outputs only:
- per-process MSE (0,1,2)
- joint 8-simplex MSE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.residual_simplex_regression import (
    load_checkpoint,
    load_rows,
    reduce_features_with_pca,
)
from src.residual_simplex_regression_log import (
    fit_joint_8_simplex_map_log_targets,
    fit_per_process_maps_log_targets,
)
from src.hmm_process.mess3 import Mess3Process


# Fixed settings (streamlined for your exact experiment)
DIM = 24
TRAIN_FRAC = 0.8
SEED = 0
MAX_SEQS = 6000
DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")

CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}

OUT_JSON = Path("artifacts/residual_simplex/comparison_trained_vs_control_dim24_logtarget_mse.json")
OUT_JSON_LAYER1 = Path("artifacts/residual_simplex/comparison_trained_vs_control_dim24_logtarget_mse_layer1.json")


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


def evaluate_case(ckpt_path: Path, rows: list[dict], cache_key: str) -> dict:
    model, cfg, _ = load_checkpoint(ckpt_path, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=cache_key)
    x_raw = ds["X"].to(torch.float32)
    x_red, _, _ = reduce_features_with_pca(x_raw, reduced_dim=DIM)

    ds_k = dict(ds)
    ds_k["X"] = x_red

    per_log = fit_per_process_maps_log_targets(ds_k, train_frac=TRAIN_FRAC, seed=SEED)
    joint_log = fit_joint_8_simplex_map_log_targets(ds_k, train_frac=TRAIN_FRAC, seed=SEED)

    return {
        "dim": DIM,
        "per_process_mse_val": {
            "0": float(per_log[0]["metrics"]["mse_val"]),
            "1": float(per_log[1]["metrics"]["mse_val"]),
            "2": float(per_log[2]["metrics"]["mse_val"]),
        },
        "joint_8_simplex_mse_val": float(joint_log["metrics"]["mse_val"]),
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
