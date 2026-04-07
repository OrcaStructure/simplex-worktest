#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import _fit_linear_map, load_checkpoint, load_rows


DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
CHECKPOINT = Path("artifacts/tiny_transformer.pt")
OUT_JSON = Path("artifacts/residual_simplex/block_alr_common_shift_experiment.json")

MAX_SEQS = 6000
EPS = 1e-8
ALPHA_TARGET = 0.1  # small positive target shift in both ALR coordinates
TOL = 1e-8


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def build_residual_dataset_from_cache_key(model, cfg, rows: List[dict], cache_key: str) -> dict:
    xs = []
    ys = []
    pids = []
    sids = []
    poss = []
    model.eval()
    for seq_idx, row in enumerate(rows):
        toks = [int(t) for t in row["tokens"]]
        pid = int(row["process_id"])
        if len(toks) < cfg.seq_len:
            continue
        x_tokens = toks[: cfg.seq_len]
        inp = torch.tensor(x_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, cache = model(inp, capture_residuals=True)
        residuals = cache["final_ln"][0].detach().cpu()

        proc = Mess3Process(alpha=float(row["alpha"]), x=float(row["x"]), dtype=torch.float64, device="cpu")
        traj = proc.belief_trajectory(x_tokens)
        for t in range(cfg.seq_len):
            xs.append(residuals[t].to(torch.float32))
            ys.append(traj[t + 1].to(torch.float32))
            pids.append(pid)
            sids.append(seq_idx)
            poss.append(t)

    return {
        "X": torch.stack(xs, dim=0),
        "Y": torch.stack(ys, dim=0),
        "process_id": torch.tensor(pids, dtype=torch.long),
        "sequence_id": torch.tensor(sids, dtype=torch.long),
        "position": torch.tensor(poss, dtype=torch.long),
    }


def rowspace_basis(W: torch.Tensor, tol: float = TOL) -> torch.Tensor:
    # W: [k,d], row-space basis in R^d
    _, s, vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    r = int((s > tol).sum().item())
    return vh[:r].T.contiguous() if r > 0 else torch.zeros((W.shape[1], 0), dtype=torch.float64)


def nullspace_basis(A: torch.Tensor, tol: float = TOL) -> torch.Tensor:
    # A: [m,d], nullspace basis in R^d
    _, s, vh = torch.linalg.svd(A.to(torch.float64), full_matrices=True)
    r = int((s > tol).sum().item())
    return vh[r:].T.contiguous()


def solve_delta_with_subspace(W_target: torch.Tensor, S: torch.Tensor, target_shift: torch.Tensor) -> torch.Tensor:
    # delta = S z, solve min ||W_target S z - target_shift||_2
    A = W_target.to(torch.float64) @ S  # [2,r]
    z = torch.linalg.lstsq(A, target_shift.to(torch.float64)).solution  # [r]
    return (S @ z).to(torch.float32)


def fit_maps(ds: dict, rows: List[dict], seq_len: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    X = ds["X"].float()
    Y = ds["Y"].float()
    pid = ds["process_id"]
    maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for p in (0, 1, 2):
        idx = torch.where(pid == p)[0]
        C = to_alr(Y[idx])
        maps[f"p{p}"] = _fit_linear_map(X[idx], C)

    block_targets = build_block_targets(rows, seq_len=seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    Cb = to_alr(Yb.float())
    maps["block"] = _fit_linear_map(Xb.float(), Cb)
    return maps


def evaluate_delta(delta: torch.Tensor, maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    out = {}
    for name, (W, _) in maps.items():
        shift = (W.float() @ delta.float()).tolist()
        out[name] = {
            "delta_c1": float(shift[0]),
            "delta_c2": float(shift[1]),
            "mean_shift": float(0.5 * (shift[0] + shift[1])),
            "coord_gap_abs": float(abs(shift[0] - shift[1])),
        }
    return out


def main() -> None:
    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    maps = fit_maps(ds, rows, seq_len=cfg.seq_len)

    W_block, _ = maps["block"]  # [2,d]
    W_p0, _ = maps["p0"]
    W_p1, _ = maps["p1"]
    W_p2, _ = maps["p2"]

    target = torch.tensor([ALPHA_TARGET, ALPHA_TARGET], dtype=torch.float64)

    # Method A: constrain delta to row-space of block map.
    S_a = rowspace_basis(W_block)
    delta_a = solve_delta_with_subspace(W_block, S_a, target)

    # Method B: constrain delta to nullspace of other maps.
    A_other = torch.cat([W_p0.float(), W_p1.float(), W_p2.float()], dim=0)  # [6,d]
    S_b = nullspace_basis(A_other)
    delta_b = solve_delta_with_subspace(W_block, S_b, target)

    eval_a = evaluate_delta(delta_a, maps)
    eval_b = evaluate_delta(delta_b, maps)

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "max_seqs": MAX_SEQS,
            "layer": "final_ln",
            "target_shift": [ALPHA_TARGET, ALPHA_TARGET],
            "target_map": "block",
            "coords": "ALR(2D)",
        },
        "method_A_rowspace_block": {
            "delta_norm_l2": float(torch.norm(delta_a).item()),
            "subspace_dim": int(S_a.shape[1]),
            "shifts": eval_a,
        },
        "method_B_nullspace_others": {
            "delta_norm_l2": float(torch.norm(delta_b).item()),
            "subspace_dim": int(S_b.shape[1]),
            "shifts": eval_b,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"saved {OUT_JSON}")
    for method_name, payload in (
        ("A(rowspace block)", out["method_A_rowspace_block"]),
        ("B(nullspace others)", out["method_B_nullspace_others"]),
    ):
        b = payload["shifts"]["block"]
        p0 = payload["shifts"]["p0"]
        p1 = payload["shifts"]["p1"]
        p2 = payload["shifts"]["p2"]
        print(f"\n{method_name}")
        print(
            f"  block: d1={b['delta_c1']:.4f} d2={b['delta_c2']:.4f} "
            f"mean={b['mean_shift']:.4f} gap={b['coord_gap_abs']:.4e}"
        )
        print(
            f"  leakage p0/p1/p2 mean shifts: "
            f"{p0['mean_shift']:.4f}, {p1['mean_shift']:.4f}, {p2['mean_shift']:.4f}"
        )
        print(
            f"  leakage p0/p1/p2 coord gaps: "
            f"{p0['coord_gap_abs']:.4e}, {p1['coord_gap_abs']:.4e}, {p2['coord_gap_abs']:.4e}"
        )


if __name__ == "__main__":
    main()
