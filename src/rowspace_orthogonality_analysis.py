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
from src.residual_simplex_regression import _fit_linear_map, load_checkpoint, load_rows, reduce_features_with_pca


# Hardcoded config at top (easy to tweak)
DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
CASES = {
    "trained": Path("artifacts/tiny_transformer.pt"),
    "control_random_init": Path("artifacts/tiny_transformer_random_init.pt"),
}
MAX_SEQS = 6000
USE_PCA = True
PCA_DIM = 24
EPS = 1e-8
RANDOM_CONTROL_SAMPLES = 2000
RANDOM_CONTROL_SEED = 0

CACHE_KEYS = {
    "final_ln": "final_ln",
    "layer1": "layer_0_after_mlp",
}

OUT_JSON = Path("artifacts/residual_simplex/rowspace_orthogonality_logtarget.json")


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


def fit_logtarget_map(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Y_log = torch.log(Y + EPS)
    W, b = _fit_linear_map(X, Y_log)  # W: [k, d], b: [k]
    return W, b


def rowspace_basis(W: torch.Tensor, tol: float = 1e-7) -> Tuple[torch.Tensor, int, List[float]]:
    # Row-space(W) in input space R^d is spanned by right singular vectors with nonzero singular values.
    _, s, vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    rank = int((s > tol).sum().item())
    Q = vh[:rank].T.contiguous() if rank > 0 else torch.zeros((W.shape[1], 0), dtype=torch.float64)
    return Q, rank, [float(v) for v in s.tolist()]


def subspace_stats(Q1: torch.Tensor, Q2: torch.Tensor) -> dict:
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return {
            "k": 0,
            "principal_angles_deg": [],
            "mean_angle_deg": None,
            "min_angle_deg": None,
            "max_angle_deg": None,
            "avg_cos2": None,
        }
    m = Q1.T @ Q2
    sv = torch.linalg.svdvals(m).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(sv))
    k = min(Q1.shape[1], Q2.shape[1])
    avg_cos2 = float((sv.pow(2).sum() / k).item())
    return {
        "k": int(k),
        "principal_angles_deg": [float(v) for v in angles.tolist()],
        "mean_angle_deg": float(angles.mean().item()),
        "min_angle_deg": float(angles.min().item()),
        "max_angle_deg": float(angles.max().item()),
        "avg_cos2": avg_cos2,  # 0=orthogonal, 1=identical (for equal dims)
    }


def _random_orthonormal_basis(d: int, r: int, generator: torch.Generator) -> torch.Tensor:
    a = torch.randn((d, r), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q


def random_subspace_control(
    d: int,
    r1: int,
    r2: int,
    samples: int,
    seed: int,
    observed: dict,
) -> dict:
    if r1 == 0 or r2 == 0:
        return {
            "samples": int(samples),
            "null_mean_angle_deg_mean": None,
            "null_mean_angle_deg_std": None,
            "null_avg_cos2_mean": None,
            "null_avg_cos2_std": None,
            "observed_mean_angle_percentile": None,
            "observed_avg_cos2_percentile": None,
        }

    g = torch.Generator().manual_seed(seed)
    mean_angles = torch.empty(samples, dtype=torch.float64)
    avg_cos2 = torch.empty(samples, dtype=torch.float64)
    for i in range(samples):
        q1 = _random_orthonormal_basis(d, r1, g)
        q2 = _random_orthonormal_basis(d, r2, g)
        st = subspace_stats(q1, q2)
        mean_angles[i] = st["mean_angle_deg"]
        avg_cos2[i] = st["avg_cos2"]

    obs_mean = float(observed["mean_angle_deg"])
    obs_cos2 = float(observed["avg_cos2"])
    pct_mean = float((mean_angles <= obs_mean).to(torch.float64).mean().item())
    pct_cos2 = float((avg_cos2 <= obs_cos2).to(torch.float64).mean().item())

    return {
        "samples": int(samples),
        "null_mean_angle_deg_mean": float(mean_angles.mean().item()),
        "null_mean_angle_deg_std": float(mean_angles.std(unbiased=True).item()),
        "null_avg_cos2_mean": float(avg_cos2.mean().item()),
        "null_avg_cos2_std": float(avg_cos2.std(unbiased=True).item()),
        # Percentile under null CDF.
        # For mean angle: high percentile => more separated/orthogonal than random.
        # For avg_cos2: low percentile => more separated/orthogonal than random.
        "observed_mean_angle_percentile": pct_mean,
        "observed_avg_cos2_percentile": pct_cos2,
    }


def evaluate_case(case_name: str, ckpt: Path, rows: List[dict], cache_key: str) -> dict:
    model, cfg, _ = load_checkpoint(ckpt, torch.device("cpu"))
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=cache_key)
    X = ds["X"].to(torch.float32)
    if USE_PCA:
        X, _, _ = reduce_features_with_pca(X, reduced_dim=PCA_DIM)
    ds["X"] = X

    maps: Dict[str, torch.Tensor] = {}
    ranks: Dict[str, int] = {}
    singvals: Dict[str, List[float]] = {}

    pid = ds["process_id"]
    for p in (0, 1, 2):
        idx = torch.where(pid == p)[0]
        Wp, _ = fit_logtarget_map(ds["X"][idx], ds["Y"][idx])
        maps[f"p{p}"] = Wp
        _, r, s = rowspace_basis(Wp)
        ranks[f"p{p}"] = r
        singvals[f"p{p}"] = s

    block_targets = build_block_targets(rows, seq_len=cfg.seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    Wb, _ = fit_logtarget_map(Xb, Yb)
    maps["block"] = Wb
    _, r, s = rowspace_basis(Wb)
    ranks["block"] = r
    singvals["block"] = s

    bases = {}
    for name, W in maps.items():
        Q, _, _ = rowspace_basis(W)
        bases[name] = Q

    pairs = [("p0", "p1"), ("p0", "p2"), ("p1", "p2"), ("p0", "block"), ("p1", "block"), ("p2", "block")]
    pair_stats = {}
    for i, (a, b) in enumerate(pairs):
        key = f"{a}_vs_{b}"
        obs = subspace_stats(bases[a], bases[b])
        ctrl = random_subspace_control(
            d=int(ds["X"].shape[1]),
            r1=int(bases[a].shape[1]),
            r2=int(bases[b].shape[1]),
            samples=RANDOM_CONTROL_SAMPLES,
            seed=RANDOM_CONTROL_SEED + 9973 * i,
            observed=obs,
        )
        pair_stats[key] = {
            **obs,
            "random_control": ctrl,
        }

    return {
        "case": case_name,
        "cache_key": cache_key,
        "input_dim": int(ds["X"].shape[1]),
        "ranks": ranks,
        "singular_values": singvals,
        "pairwise": pair_stats,
    }


def main() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "max_seqs": MAX_SEQS,
            "use_pca": USE_PCA,
            "pca_dim": PCA_DIM if USE_PCA else None,
            "eps": EPS,
            "random_control_samples": RANDOM_CONTROL_SAMPLES,
            "random_control_seed": RANDOM_CONTROL_SEED,
        },
        "results": {},
    }

    for rep_name, ck in CASES.items():
        out["results"][rep_name] = {}
        for tag, cache_key in CACHE_KEYS.items():
            out["results"][rep_name][tag] = evaluate_case(rep_name, ck, rows, cache_key)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"saved {OUT_JSON}")
    for rep_name in CASES:
        for tag in CACHE_KEYS:
            r = out["results"][rep_name][tag]
            print(f"\n[{rep_name} | {tag}] input_dim={r['input_dim']} ranks={r['ranks']}")
            for k, v in r["pairwise"].items():
                print(
                    f"  {k}: mean_angle_deg={v['mean_angle_deg']:.3f} "
                    f"min_angle_deg={v['min_angle_deg']:.3f} avg_cos2={v['avg_cos2']:.4f} "
                    f"| rand(mean_angle_pct={v['random_control']['observed_mean_angle_percentile']:.3f}, "
                    f"avg_cos2_pct={v['random_control']['observed_avg_cos2_percentile']:.3f})"
                )


if __name__ == "__main__":
    main()
