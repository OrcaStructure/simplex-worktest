#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
OUT_DIR = ROOT / "artifacts" / "canonical_figures"

E1_JSON = ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_alr_kl.json"
E1_JSON_L1 = ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_alr_kl_layer1.json"
E2_JSON = ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_prob_kl.json"
E2_JSON_L1 = ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_prob_kl_layer1.json"
E3_JSON = ROOT / "artifacts" / "residual_simplex" / "rowspace_orthogonality_alr_nopca.json"
E4_JSON = ROOT / "artifacts" / "residual_simplex" / "alr_to_alr_kl.json"
E5_JSON = ROOT / "artifacts" / "residual_simplex" / "prob_to_prob_kl.json"
E6_IMG_A = ROOT / "src" / "hmm_process" / "artifacts" / "mess3_mixture_viz" / "all_processes_overlay_simplex_points.png"
E6_IMG_B = ROOT / "artifacts" / "residual_simplex" / "alr_ground_truth" / "ground_truth_alr_overlay.png"


PY = sys.executable


def run_if_missing(path: Path, cmd: List[str]) -> None:
    if path.exists():
        return
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_heatmap(ax, arr: np.ndarray, row_labels: List[str], col_labels: List[str], title: str) -> None:
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def figure_e1_or_e2(path_final: Path, path_layer1: Path, out_path: Path, title_left: str, title_right: str) -> None:
    d0 = load_json(path_final)
    d1 = load_json(path_layer1)
    cases = ["trained", "control_random_init"]
    cols = ["p0", "p1", "p2", "block", "joint8"]

    def mat(d: Dict[str, dict]) -> np.ndarray:
        m = np.zeros((2, 5), dtype=float)
        for i, c in enumerate(cases):
            m[i, 0] = float(d[c]["per_process_kl_val"]["0"])
            m[i, 1] = float(d[c]["per_process_kl_val"]["1"])
            m[i, 2] = float(d[c]["per_process_kl_val"]["2"])
            m[i, 3] = float(d[c]["block_3_simplex_kl_val"])
            m[i, 4] = float(d[c]["joint_8_simplex_kl_val"])
        return m

    m0 = mat(d0)
    m1 = mat(d1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    save_heatmap(axes[0], m0, ["trained", "control"], cols, title_left)
    save_heatmap(axes[1], m1, ["trained", "control"], cols, title_right)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_e3(path: Path, out_path: Path) -> None:
    d = load_json(path)["results"]
    names = ["p0", "p1", "p2", "block"]

    def matrix_for(tag: str, case: str) -> np.ndarray:
        M = np.zeros((4, 4), dtype=float)
        np.fill_diagonal(M, 0.0)
        pairs = d[tag]["within_case_pairwise"][case]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                k = f"{a}_vs_{b}" if f"{a}_vs_{b}" in pairs else f"{b}_vs_{a}"
                M[i, j] = float(pairs[k]["mean_angle_deg"])
        return M

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.0))
    tags = [("final_ln", "trained"), ("final_ln", "control_random_init"), ("layer1", "trained"), ("layer1", "control_random_init")]
    for ax, (tag, case) in zip(axes.flatten(), tags):
        M = matrix_for(tag, case)
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=90)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"{tag} | {'control' if case!='trained' else 'trained'}", fontsize=9)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", color="white", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pair_matrix(path: Path) -> np.ndarray:
    d = load_json(path)["pairs"]
    names = ["p0", "p1", "p2", "block"]
    M = np.zeros((4, 4), dtype=float)
    for i, s in enumerate(names):
        for j, t in enumerate(names):
            M[i, j] = float(d[f"{s}_to_{t}"]["kl_val"])
    return M


def figure_e4_or_e5(path: Path, out_path: Path, title: str) -> None:
    names = ["p0", "p1", "p2", "block"]
    M = pair_matrix(path)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_e6(img_a: Path, img_b: Path, out_path: Path) -> None:
    a = plt.imread(img_a)
    b = plt.imread(img_b)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].imshow(a)
    axes[0].axis("off")
    axes[0].set_title("simplex overlay", fontsize=10)
    axes[1].imshow(b)
    axes[1].axis("off")
    axes[1].set_title("ALR overlay", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure required artifacts exist.
    run_if_missing(E1_JSON, [PY, "src/compare_alr_kl_nopca.py"])
    run_if_missing(E1_JSON_L1, [PY, "src/compare_alr_kl_nopca.py"])
    run_if_missing(E2_JSON, [PY, "src/compare_prob_kl_nopca.py"])
    run_if_missing(E2_JSON_L1, [PY, "src/compare_prob_kl_nopca.py"])
    run_if_missing(E3_JSON, [PY, "src/rowspace_orthogonality_alr_nopca.py"])
    run_if_missing(E4_JSON, [PY, "src/alr_to_alr_regression.py"])
    run_if_missing(E5_JSON, [PY, "src/prob_to_prob_regression.py"])
    run_if_missing(E6_IMG_A, [PY, "src/hmm_process/visualize_mess3_mixture.py"])
    run_if_missing(E6_IMG_B, [PY, "src/visualize_ground_truth_alr.py"])

    figure_e1_or_e2(
        E1_JSON,
        E1_JSON_L1,
        OUT_DIR / "E1_alr_residual_decoding.png",
        "final_ln",
        "layer1",
    )
    figure_e1_or_e2(
        E2_JSON,
        E2_JSON_L1,
        OUT_DIR / "E2_prob_residual_decoding.png",
        "final_ln",
        "layer1",
    )
    figure_e3(E3_JSON, OUT_DIR / "E3_rowspace_orthogonality.png")
    figure_e4_or_e5(E4_JSON, OUT_DIR / "E4_alr_to_alr_kl.png", "ALR->ALR KL")
    figure_e4_or_e5(E5_JSON, OUT_DIR / "E5_prob_to_prob_kl.png", "prob->prob KL")
    figure_e6(E6_IMG_A, E6_IMG_B, OUT_DIR / "E6_visual_diagnostics.png")

    print(f"Saved canonical figures to: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
