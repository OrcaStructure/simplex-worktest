#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import build_block_targets, extract_specs
from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import load_rows


DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
OUT_DIR = Path("artifacts/residual_simplex/alr_ground_truth")
MAX_SEQS = 6000
SEQ_LEN = 12
MAX_POINTS_PER_SET = 120000
EPS = 1e-8
SEED = 0


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def build_targets(rows: List[dict], seq_len: int) -> Dict[str, torch.Tensor]:
    specs = extract_specs(rows)
    procs = [Mess3Process(alpha=a, x=x, dtype=torch.float64, device="cpu") for a, x in specs]
    block_t = build_block_targets(rows, seq_len=seq_len)

    p0, p1, p2, pb = [], [], [], []
    for seq_idx, r in enumerate(rows):
        toks = [int(t) for t in r["tokens"][:seq_len]]
        if len(toks) < seq_len:
            continue
        trajs = [p.belief_trajectory(toks).to(torch.float32) for p in procs]
        for pos in range(seq_len):
            p0.append(trajs[0][pos + 1])
            p1.append(trajs[1][pos + 1])
            p2.append(trajs[2][pos + 1])
            pb.append(block_t[(seq_idx, pos)])
    return {
        "p0": torch.stack(p0, dim=0),
        "p1": torch.stack(p1, dim=0),
        "p2": torch.stack(p2, dim=0),
        "block": torch.stack(pb, dim=0),
    }


def subsample(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    return x[idx]


def main() -> None:
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    targets = build_targets(rows, seq_len=SEQ_LEN)

    alr = {k: to_alr(v) for k, v in targets.items()}
    alr = {k: subsample(v, MAX_POINTS_PER_SET, seed=SEED + i) for i, (k, v) in enumerate(alr.items())}

    colors = {"p0": "#1f77b4", "p1": "#ff7f0e", "p2": "#2ca02c", "block": "#d62728"}

    # 2x2 separate panels
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    for ax, name in zip(axes.flatten(), ("p0", "p1", "p2", "block")):
        pts = alr[name]
        ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), s=2.0, alpha=0.08, color=colors[name])
        ax.set_title(f"{name} in ALR coords")
        ax.set_xlabel("log(p1/p3)")
        ax.set_ylabel("log(p2/p3)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ground_truth_alr_2x2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Overlay
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for name in ("p0", "p1", "p2", "block"):
        pts = alr[name]
        ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), s=2.0, alpha=0.05, color=colors[name], label=name)
    ax.set_title("Ground-truth distributions in ALR coordinates")
    ax.set_xlabel("log(p1/p3)")
    ax.set_ylabel("log(p2/p3)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ground_truth_alr_overlay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'ground_truth_alr_2x2.png'}")
    print(f"Saved: {OUT_DIR / 'ground_truth_alr_overlay.png'}")


if __name__ == "__main__":
    main()

