#!/usr/bin/env python3
"""Comparison with direct probability targets (no log transform at all).

Fits linear maps directly in probability space:
  y_hat_raw = W x + b
  y_hat = project_to_simplex(y_hat_raw)
and reports KL on validation splits.

Outputs for trained vs random-init at:
- final_ln
- layer_0_after_mlp
for:
- per-process 3-simplex
- joint 8-simplex / 9-way probs
- block 3-simplex
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
    _fit_linear_map,
    _project_to_simplex,
    _split_indices,
    load_checkpoint,
    load_rows,
)


TRAIN_FRAC = 0.8
SEED = 0
MAX_SEQS = 6000
EPS = 1e-8

DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}

OUT_JSON = Path("artifacts/residual_simplex/comparison_trained_vs_control_nopca_prob_kl.json")
OUT_JSON_LAYER1 = Path("artifacts/residual_simplex/comparison_trained_vs_control_nopca_prob_kl_layer1.json")


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


def mean_kl(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = EPS) -> float:
    y = y_true.clamp_min(eps)
    p = y_pred.clamp_min(eps)
    return float((y * (torch.log(y) - torch.log(p))).sum(dim=1).mean().item())


def fit_eval_prob_kl(X: torch.Tensor, Y: torch.Tensor, split_seed: int) -> float:
    tr, va = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=split_seed)
    W, b = _fit_linear_map(X[tr], Y[tr])
    Y_pred = _project_to_simplex(X[va] @ W.T + b)
    return mean_kl(Y[va], Y_pred, eps=EPS)


def per_process_kl(ds: dict) -> dict[str, float]:
    X = ds["X"]
    Y = ds["Y"]
    pid = ds["process_id"]
    out: dict[str, float] = {}
    for p in sorted(int(v) for v in pid.unique().tolist()):
        idx = torch.where(pid == p)[0]
        out[str(p)] = fit_eval_prob_kl(X[idx], Y[idx], split_seed=SEED + p)
    return out


def joint_8_kl(ds: dict) -> float:
    X = ds["X"]
    Y_joint = _build_joint_targets(ds["Y"], ds["process_id"], num_processes=3)
    return fit_eval_prob_kl(X, Y_joint, split_seed=SEED + 123)


def block_3_kl(ds: dict, rows: list[dict], seq_len: int) -> float:
    block_targets = build_block_targets(rows, seq_len=seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    return fit_eval_prob_kl(Xb, Yb, split_seed=SEED + 777)


def evaluate_case(ckpt_path: Path, rows: list[dict], cache_key: str) -> dict:
    model, cfg, _ = load_checkpoint(ckpt_path, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=cache_key)
    ds["X"] = ds["X"].float()

    return {
        "input_dim": int(ds["X"].shape[1]),
        "target_space": "probability",
        "per_process_kl_val": per_process_kl(ds),
        "joint_8_simplex_kl_val": float(joint_8_kl(ds)),
        "block_3_simplex_kl_val": float(block_3_kl(ds, rows=rows, seq_len=cfg.seq_len)),
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
