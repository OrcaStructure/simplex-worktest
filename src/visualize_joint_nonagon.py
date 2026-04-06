#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.residual_simplex_regression import _build_joint_targets, _project_to_simplex


RESIDUAL_DS_PATH = Path("artifacts/residual_simplex/residual_dataset.pt")
JOINT_MAP_PATH = Path("artifacts/residual_simplex/linear_map_joint_8_simplex.pt")
OUT_DIR = Path("artifacts/residual_simplex")

MAX_POINTS = 120000


def nonagon_vertices() -> torch.Tensor:
    # 9 vertices on unit circle.
    verts = []
    for i in range(9):
        theta = 2.0 * math.pi * i / 9.0
        verts.append([math.cos(theta), math.sin(theta)])
    return torch.tensor(verts, dtype=torch.float64)


def project_to_nonagon(points_9simplex: torch.Tensor) -> torch.Tensor:
    verts = nonagon_vertices().to(dtype=points_9simplex.dtype, device=points_9simplex.device)  # [9,2]
    return points_9simplex @ verts  # [N,2]


def _subsample_indices(n: int, max_points: int) -> torch.Tensor:
    if n <= max_points:
        return torch.arange(n)
    g = torch.Generator().manual_seed(0)
    return torch.randperm(n, generator=g)[:max_points]


def main() -> None:
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = torch.load(RESIDUAL_DS_PATH, map_location="cpu")
    jm = torch.load(JOINT_MAP_PATH, map_location="cpu")

    X = ds["X"].to(dtype=torch.float32)  # expected already PCA-reduced to match W
    Y_local = ds["Y"].to(dtype=torch.float32)
    pid = ds["process_id"].to(dtype=torch.long)

    Y_joint_true = _build_joint_targets(Y_local, pid, num_processes=3).to(dtype=torch.float32)
    Y_joint_pred = _project_to_simplex(X @ jm["W"].to(dtype=torch.float32).T + jm["b"].to(dtype=torch.float32))

    idx = _subsample_indices(X.shape[0], MAX_POINTS)
    Yt = Y_joint_true[idx]
    Yp = Y_joint_pred[idx]
    pid_s = pid[idx]

    XY_true = project_to_nonagon(Yt.to(dtype=torch.float64))
    XY_pred = project_to_nonagon(Yp.to(dtype=torch.float64))

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    verts = nonagon_vertices()
    ring = torch.vstack([verts, verts[0:1]])

    for name, xy in [("true", XY_true), ("pred", XY_pred)]:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(ring[:, 0].numpy(), ring[:, 1].numpy(), color="black", linewidth=1.2, alpha=0.8)
        ax.scatter(verts[:, 0].numpy(), verts[:, 1].numpy(), color="black", s=10)
        for i in range(9):
            ax.text(float(verts[i, 0]) * 1.08, float(verts[i, 1]) * 1.08, str(i), ha="center", va="center", fontsize=8)

        for p in (0, 1, 2):
            pidx = torch.where(pid_s == p)[0]
            ax.scatter(
                xy[pidx, 0].numpy(),
                xy[pidx, 1].numpy(),
                s=3,
                alpha=0.08 if name == "true" else 0.15,
                color=colors[p],
                label=f"process {p}",
            )

        ax.set_aspect("equal")
        ax.set_title(f"Joint 8-simplex projected to nonagon ({name})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        out = OUT_DIR / f"joint_nonagon_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Optional overlay plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ring[:, 0].numpy(), ring[:, 1].numpy(), color="black", linewidth=1.2, alpha=0.8)
    ax.scatter(XY_true[:, 0].numpy(), XY_true[:, 1].numpy(), s=2, alpha=0.05, color="#1f77b4", label="true")
    ax.scatter(XY_pred[:, 0].numpy(), XY_pred[:, 1].numpy(), s=2, alpha=0.05, color="#d62728", label="pred")
    ax.set_aspect("equal")
    ax.set_title("Joint 8-simplex projected to nonagon (overlay)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.savefig(OUT_DIR / "joint_nonagon_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'joint_nonagon_true.png'}")
    print(f"Saved: {OUT_DIR / 'joint_nonagon_pred.png'}")
    print(f"Saved: {OUT_DIR / 'joint_nonagon_overlay.png'}")


if __name__ == "__main__":
    main()
