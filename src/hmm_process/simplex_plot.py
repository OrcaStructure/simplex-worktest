from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import torch


def _get_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def simplex_vertices() -> torch.Tensor:
    # Equilateral triangle in 2D: vertices map to pure states e1,e2,e3.
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8660254037844386],  # sqrt(3)/2
        ],
        dtype=torch.float64,
    )


def barycentric_to_cartesian(beliefs: torch.Tensor) -> torch.Tensor:
    if beliefs.ndim != 2 or beliefs.shape[1] != 3:
        raise ValueError("beliefs must have shape [N,3]")
    verts = simplex_vertices().to(dtype=beliefs.dtype, device=beliefs.device)
    return beliefs @ verts


def plot_belief_trajectories_on_simplex(
    trajectories: Sequence[torch.Tensor],
    observations: Sequence[Sequence[int]] | None = None,
    title: str = "Mess3 Belief Trajectories on 2-Simplex",
    save_path: str | None = None,
) -> None:
    plt = _get_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 6))
    verts = simplex_vertices()
    triangle = torch.vstack([verts, verts[0:1]])
    ax.plot(triangle[:, 0].numpy(), triangle[:, 1].numpy(), color="black", linewidth=1.5)

    labels = ["state 0", "state 1", "state 2"]
    for i in range(3):
        ax.text(float(verts[i, 0]), float(verts[i, 1]) + 0.03, labels[i], ha="center")

    cmap = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for idx, traj in enumerate(trajectories):
        xy = barycentric_to_cartesian(traj.detach().cpu())
        ax.plot(xy[:, 0].numpy(), xy[:, 1].numpy(), color="0.6", linewidth=1.0, alpha=0.7)
        ax.scatter(xy[0, 0].item(), xy[0, 1].item(), color="black", s=18, zorder=4)

        if observations is not None and idx < len(observations):
            obs = observations[idx]
            for t, tok in enumerate(obs):
                c = cmap.get(int(tok), "0.3")
                ax.scatter(xy[t + 1, 0].item(), xy[t + 1, 1].item(), color=c, s=22, zorder=4)
        else:
            ax.scatter(xy[1:, 0].numpy(), xy[1:, 1].numpy(), color="#1f77b4", s=20, zorder=4)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=f"token {tok}") for tok, c in cmap.items()]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("simplex x")
    ax.set_ylabel("simplex y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

