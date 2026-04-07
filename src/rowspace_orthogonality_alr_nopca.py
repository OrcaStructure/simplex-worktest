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
MAX_SEQS = 6000
EPS = 1e-8

CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}
CACHE_KEYS = {
    "final_ln": "final_ln",
    "layer1": "layer_0_after_mlp",
}

OUT_JSON = Path("artifacts/residual_simplex/rowspace_orthogonality_alr_nopca.json")


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


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def fit_alr_map(X: torch.Tensor, Y_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Map into 2D ALR coordinates for 3-way simplex.
    C = to_alr(Y_prob)
    W, b = _fit_linear_map(X, C)  # W: [2,d], b: [2]
    return W, b


def rowspace_basis(W: torch.Tensor, tol: float = 1e-8) -> Tuple[torch.Tensor, int]:
    _, s, vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    rank = int((s > tol).sum().item())
    q = vh[:rank].T.contiguous() if rank > 0 else torch.zeros((W.shape[1], 0), dtype=torch.float64)
    return q, rank


def subspace_stats(q1: torch.Tensor, q2: torch.Tensor) -> dict:
    if q1.shape[1] == 0 or q2.shape[1] == 0:
        return {
            "k": 0,
            "principal_angles_deg": [],
            "mean_angle_deg": None,
            "min_angle_deg": None,
            "max_angle_deg": None,
            "avg_cos2": None,
        }
    s = torch.linalg.svdvals(q1.T @ q2).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(s))
    k = min(q1.shape[1], q2.shape[1])
    return {
        "k": int(k),
        "principal_angles_deg": [float(v) for v in angles.tolist()],
        "mean_angle_deg": float(angles.mean().item()),
        "min_angle_deg": float(angles.min().item()),
        "max_angle_deg": float(angles.max().item()),
        "avg_cos2": float((s.pow(2).sum() / k).item()),
    }


def fit_four_maps(rows: List[dict], ckpt_path: Path, cache_key: str) -> Tuple[Dict[str, torch.Tensor], int]:
    model, cfg, _ = load_checkpoint(ckpt_path, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key)
    ds["X"] = ds["X"].float()

    maps: Dict[str, torch.Tensor] = {}
    pid = ds["process_id"]
    for p in (0, 1, 2):
        idx = torch.where(pid == p)[0]
        w, _ = fit_alr_map(ds["X"][idx], ds["Y"][idx])
        maps[f"p{p}"] = w

    block_targets = build_block_targets(rows, seq_len=cfg.seq_len)
    xb, yb = align_xy(ds, block_targets)
    wb, _ = fit_alr_map(xb, yb)
    maps["block"] = wb

    return maps, int(ds["X"].shape[1])


def within_case_pairwise(bases: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    pairs = [("p0", "p1"), ("p0", "p2"), ("p1", "p2"), ("p0", "block"), ("p1", "block"), ("p2", "block")]
    out = {}
    for a, b in pairs:
        out[f"{a}_vs_{b}"] = subspace_stats(bases[a], bases[b])
    return out


def main() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    result = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "max_seqs": MAX_SEQS,
            "pca_dim": None,
            "target": "ALR(2D) for p0,p1,p2,block maps",
        },
        "results": {},
    }

    for tag, cache_key in CACHE_KEYS.items():
        maps_by_case: Dict[str, Dict[str, torch.Tensor]] = {}
        bases_by_case: Dict[str, Dict[str, torch.Tensor]] = {}
        ranks_by_case: Dict[str, Dict[str, int]] = {}
        input_dim = None

        for case_name, ckpt in CASES.items():
            maps, d_in = fit_four_maps(rows, ckpt, cache_key=cache_key)
            input_dim = d_in
            maps_by_case[case_name] = maps
            bases = {}
            ranks = {}
            for name, w in maps.items():
                q, r = rowspace_basis(w)
                bases[name] = q
                ranks[name] = r
            bases_by_case[case_name] = bases
            ranks_by_case[case_name] = ranks

        cross_case = {}
        for name in ("p0", "p1", "p2", "block"):
            cross_case[f"trained_vs_control_{name}"] = subspace_stats(
                bases_by_case["trained"][name], bases_by_case["control_random_init"][name]
            )

        result["results"][tag] = {
            "input_dim": input_dim,
            "ranks": ranks_by_case,
            "within_case_pairwise": {
                case_name: within_case_pairwise(bases_by_case[case_name]) for case_name in CASES
            },
            "cross_case_same_map": cross_case,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"saved {OUT_JSON}")
    for tag in CACHE_KEYS:
        r = result["results"][tag]
        print(f"\n[{tag}] input_dim={r['input_dim']}")
        print(f"  ranks: {r['ranks']}")
        for k, v in r["cross_case_same_map"].items():
            print(
                f"  {k}: mean_angle_deg={v['mean_angle_deg']:.3f} "
                f"min_angle_deg={v['min_angle_deg']:.3f} avg_cos2={v['avg_cos2']:.4f}"
            )


if __name__ == "__main__":
    main()
