#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process.mess3 import Mess3Process
from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices


# Hardcoded config (edit here)
DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
OUT_DIR = Path("src/hmm_process/artifacts/mess3_mixture_viz")
MAX_ROWS = 6000
SEQ_LEN = 12
MAX_SIMPLEX_POINTS_PER_PROCESS = 100000
SEED = 0


def load_rows(path: Path, max_rows: int | None = None) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def extract_specs(rows: List[dict]) -> Dict[int, Tuple[float, float]]:
    pid_to_spec: Dict[int, Tuple[float, float]] = {}
    for r in rows:
        pid = int(r["process_id"])
        spec = (float(r["alpha"]), float(r["x"]))
        if pid in pid_to_spec and pid_to_spec[pid] != spec:
            raise ValueError(f"inconsistent spec for process_id={pid}: {pid_to_spec[pid]} vs {spec}")
        pid_to_spec[pid] = spec
    return pid_to_spec


def build_simplex_point_cloud(rows: List[dict], seq_len: int, seed: int) -> Dict[int, torch.Tensor]:
    pid_to_spec = extract_specs(rows)
    procs = {pid: Mess3Process(alpha=a, x=x, dtype=torch.float64, device="cpu") for pid, (a, x) in pid_to_spec.items()}

    clouds: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for r in rows:
        pid = int(r["process_id"])
        toks = [int(t) for t in r["tokens"][:seq_len]]
        if len(toks) < seq_len:
            continue
        traj = procs[pid].belief_trajectory(toks)  # [T+1, 3]
        # Skip initial prior point so we only show updates from observed tokens.
        clouds[pid].append(traj[1:].to(torch.float32))

    out: Dict[int, torch.Tensor] = {}
    g = torch.Generator().manual_seed(seed)
    for pid, chunks in clouds.items():
        all_pts = torch.cat(chunks, dim=0)
        if all_pts.shape[0] > MAX_SIMPLEX_POINTS_PER_PROCESS:
            idx = torch.randperm(all_pts.shape[0], generator=g)[:MAX_SIMPLEX_POINTS_PER_PROCESS]
            all_pts = all_pts[idx]
        out[pid] = all_pts
    return out


def plot_process_simplex(clouds: Dict[int, torch.Tensor], specs: Dict[int, Tuple[float, float]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[:1]])
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for pid in sorted(clouds):
        pts = clouds[pid]
        xy = barycentric_to_cartesian(pts.to(torch.float64))
        a, x = specs[pid]

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black", linewidth=1.2)
        ax.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(), s=2.0, alpha=0.08, color=colors.get(pid, "0.3"))
        ax.set_aspect("equal")
        ax.set_xlabel("simplex x")
        ax.set_ylabel("simplex y")
        ax.grid(alpha=0.2)
        ax.set_title(f"Process {pid} simplex points (alpha={a:.2f}, x={x:.2f})")
        fig.savefig(out_dir / f"process_{pid}_simplex_points.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_mixture_summary(rows: List[dict], specs: Dict[int, Tuple[float, float]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    process_counts = Counter(int(r["process_id"]) for r in rows)
    token_counts_global = Counter()
    token_counts_by_process = {pid: Counter() for pid in specs}
    for r in rows:
        pid = int(r["process_id"])
        toks = [int(t) for t in r["tokens"][:SEQ_LEN]]
        token_counts_global.update(toks)
        token_counts_by_process[pid].update(toks)

    pids = sorted(specs)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # Left: sequence counts by process
    seq_vals = [process_counts[p] for p in pids]
    axes[0].bar([str(p) for p in pids], seq_vals, color=colors)
    axes[0].set_title("Sequence count per process")
    axes[0].set_xlabel("process_id")
    axes[0].set_ylabel("num sequences")
    for i, p in enumerate(pids):
        a, x = specs[p]
        axes[0].text(i, seq_vals[i], f"a={a:.2f}\nx={x:.2f}", ha="center", va="bottom", fontsize=8)

    # Middle: global token histogram
    toks = [0, 1, 2]
    gvals = [token_counts_global[t] for t in toks]
    axes[1].bar([str(t) for t in toks], gvals, color="#4c78a8")
    axes[1].set_title("Global token counts")
    axes[1].set_xlabel("token")
    axes[1].set_ylabel("count")

    # Right: per-process token proportions (stacked)
    bottom = torch.zeros(len(pids), dtype=torch.float64)
    for tok, c in zip(toks, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        vals = []
        for p in pids:
            cnts = token_counts_by_process[p]
            total = sum(cnts.values())
            vals.append(0.0 if total == 0 else cnts[tok] / total)
        vals_t = torch.tensor(vals, dtype=torch.float64)
        axes[2].bar([str(p) for p in pids], vals_t.numpy(), bottom=bottom.numpy(), color=c, label=f"token {tok}")
        bottom += vals_t
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Per-process token proportions")
    axes[2].set_xlabel("process_id")
    axes[2].set_ylabel("proportion")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_simplex(clouds: Dict[int, torch.Tensor], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[:1]])
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black", linewidth=1.2)
    for pid in sorted(clouds):
        xy = barycentric_to_cartesian(clouds[pid].to(torch.float64))
        ax.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(), s=2.0, alpha=0.06, color=colors.get(pid, "0.3"), label=f"process {pid}")
    ax.set_aspect("equal")
    ax.set_xlabel("simplex x")
    ax.set_ylabel("simplex y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_title("All processes overlaid in simplex")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET_PATH, MAX_ROWS)
    specs = extract_specs(rows)
    clouds = build_simplex_point_cloud(rows, seq_len=SEQ_LEN, seed=SEED)

    plot_process_simplex(clouds, specs, OUT_DIR)
    plot_overlay_simplex(clouds, OUT_DIR / "all_processes_overlay_simplex_points.png")
    plot_mixture_summary(rows, specs, OUT_DIR / "mixture_summary.png")

    print(f"Saved visualizations to: {OUT_DIR}")
    for p in sorted(specs):
        print(f"  {OUT_DIR / f'process_{p}_simplex_points.png'}")
    print(f"  {OUT_DIR / 'all_processes_overlay_simplex_points.png'}")
    print(f"  {OUT_DIR / 'mixture_summary.png'}")


if __name__ == "__main__":
    main()

