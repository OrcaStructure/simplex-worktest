#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.mess3 import Mess3Process
from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices
from src.residual_simplex_regression import _fit_linear_map, _project_to_simplex, _split_indices, load_checkpoint, load_rows


PY = sys.executable
ROOT = Path(".")
OUT_DIR = ROOT / "artifacts" / "canonical_outputs"

DATASET_PATH = ROOT / "src" / "hmm_process" / "artifacts" / "mess3_mixed_dataset.jsonl"
MAX_SEQS = 6000
TRAIN_FRAC = 0.8
SEED = 0
EPS = 1e-8
MAX_POINTS = 120000

CASES = {
    "trained": ROOT / "artifacts" / "tiny_transformer.pt",
    "control": ROOT / "artifacts" / "tiny_transformer_random_init.pt",
}
CACHE_KEY = "final_ln"


def run_if_missing(path: Path, cmd: List[str]) -> None:
    if path.exists():
        return
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def from_alr(c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([c, torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)], dim=1)
    return torch.softmax(z, dim=1)


def build_residual_dataset_from_cache_key(model, cfg, rows: List[dict], cache_key: str) -> dict:
    xs = []
    ys = []
    pid = []
    sid = []
    pos = []
    model.eval()
    for seq_idx, row in enumerate(rows):
        toks = [int(t) for t in row["tokens"]]
        p = int(row["process_id"])
        if len(toks) < cfg.seq_len:
            continue
        x_tokens = toks[: cfg.seq_len]
        inp = torch.tensor(x_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, cache = model(inp, capture_residuals=True)
        residuals = cache[cache_key][0].detach().cpu()
        proc = Mess3Process(alpha=float(row["alpha"]), x=float(row["x"]), dtype=torch.float64, device="cpu")
        traj = proc.belief_trajectory(x_tokens)
        for t in range(cfg.seq_len):
            xs.append(residuals[t].to(torch.float32))
            ys.append(traj[t + 1].to(torch.float32))
            pid.append(p)
            sid.append(seq_idx)
            pos.append(t)
    return {
        "X": torch.stack(xs, dim=0),
        "Y": torch.stack(ys, dim=0),
        "process_id": torch.tensor(pid, dtype=torch.long),
        "sequence_id": torch.tensor(sid, dtype=torch.long),
        "position": torch.tensor(pos, dtype=torch.long),
    }


def fit_predict_prob(X: torch.Tensor, Y: torch.Tensor, seed: int) -> torch.Tensor:
    tr, _ = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=seed)
    W, b = _fit_linear_map(X[tr], Y[tr])
    return _project_to_simplex(X @ W.T + b)


def fit_predict_alr(X: torch.Tensor, Y: torch.Tensor, seed: int) -> torch.Tensor:
    tr, _ = _split_indices(X.shape[0], train_frac=TRAIN_FRAC, seed=seed)
    C = to_alr(Y)
    W, b = _fit_linear_map(X[tr], C[tr])
    return from_alr(X @ W.T + b)


def sample_pairs(y_true: torch.Tensor, y_pred: torch.Tensor, n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if y_true.shape[0] <= n:
        return y_true, y_pred
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(y_true.shape[0], generator=g)[:n]
    return y_true[idx], y_pred[idx]


def plot_side_by_side_simplex(
    panels: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    out_path: Path,
    title: str,
) -> None:
    # panels keys expected: p0,p1,p2,block ; value=(y_true,y_pred)
    verts = simplex_vertices()
    tri = torch.vstack([verts, verts[:1]])
    order = ["p0", "p1", "p2", "block"]
    colors = {"p0": "#1f77b4", "p1": "#ff7f0e", "p2": "#2ca02c", "block": "#d62728"}

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 13.6))
    fig.suptitle(title, fontsize=11)
    for r, name in enumerate(order):
        y_true, y_pred = panels[name]
        xy_t = barycentric_to_cartesian(y_true.to(torch.float64))
        xy_p = barycentric_to_cartesian(y_pred.to(torch.float64))
        for c, xy in enumerate((xy_t, xy_p)):
            ax = axes[r, c]
            ax.plot(tri[:, 0].numpy(), tri[:, 1].numpy(), color="black", linewidth=1.0)
            ax.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(), s=1.8, alpha=0.07, color=colors[name])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if c == 0:
                ax.set_title(f"{name} true", fontsize=9)
            else:
                ax.set_title(f"{name} pred", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_e1_e2_csvs() -> None:
    e1 = load_json(ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_alr_kl.json")
    e1_l1 = load_json(ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_alr_kl_layer1.json")
    e2 = load_json(ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_prob_kl.json")
    e2_l1 = load_json(ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_prob_kl_layer1.json")

    def rows_from(payload: dict, layer: str, exp: str) -> List[List[object]]:
        rows = []
        for case in ("trained", "control_random_init"):
            per = payload[case]["per_process_kl_val"]
            rows.append([exp, layer, case, "p0", float(per["0"])])
            rows.append([exp, layer, case, "p1", float(per["1"])])
            rows.append([exp, layer, case, "p2", float(per["2"])])
            rows.append([exp, layer, case, "block", float(payload[case]["block_3_simplex_kl_val"])])
            rows.append([exp, layer, case, "joint8", float(payload[case]["joint_8_simplex_kl_val"])])
        return rows

    all_rows = []
    all_rows += rows_from(e1, "final_ln", "E1_ALR")
    all_rows += rows_from(e1_l1, "layer1", "E1_ALR")
    all_rows += rows_from(e2, "final_ln", "E2_PROB")
    all_rows += rows_from(e2_l1, "layer1", "E2_PROB")
    write_csv(OUT_DIR / "E1_E2_metrics.csv", ["experiment", "layer", "case", "target", "kl_val"], all_rows)


def export_e3_csvs() -> None:
    d = load_json(ROOT / "artifacts" / "residual_simplex" / "rowspace_orthogonality_alr_nopca.json")["results"]
    within_rows: List[List[object]] = []
    cross_rows: List[List[object]] = []
    for layer in ("final_ln", "layer1"):
        for case in ("trained", "control_random_init"):
            pairs = d[layer]["within_case_pairwise"][case]
            for pair, vals in pairs.items():
                within_rows.append([layer, case, pair, vals["mean_angle_deg"], vals["min_angle_deg"], vals["avg_cos2"]])
        for pair, vals in d[layer]["cross_case_same_map"].items():
            cross_rows.append([layer, pair, vals["mean_angle_deg"], vals["min_angle_deg"], vals["avg_cos2"]])

    write_csv(
        OUT_DIR / "E3_within_case_pairwise.csv",
        ["layer", "case", "pair", "mean_angle_deg", "min_angle_deg", "avg_cos2"],
        within_rows,
    )
    write_csv(
        OUT_DIR / "E3_cross_case_same_map.csv",
        ["layer", "pair", "mean_angle_deg", "min_angle_deg", "avg_cos2"],
        cross_rows,
    )


def export_e4_e5_csvs() -> None:
    for exp_name, path in (
        ("E4_ALR_TO_ALR", ROOT / "artifacts" / "residual_simplex" / "alr_to_alr_kl.json"),
        ("E5_PROB_TO_PROB", ROOT / "artifacts" / "residual_simplex" / "prob_to_prob_kl.json"),
    ):
        d = load_json(path)["pairs"]
        rows = []
        for pair, vals in d.items():
            src, tgt = pair.split("_to_")
            rows.append([exp_name, src, tgt, vals["kl_val"]])
        write_csv(OUT_DIR / f"{exp_name}_kl_matrix.csv", ["experiment", "source", "target", "kl_val"], rows)


def make_e1_e2_side_by_side() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    for case_name, ckpt in CASES.items():
        model, cfg, _ = load_checkpoint(ckpt, torch.device("cpu"))
        ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key=CACHE_KEY)
        X = ds["X"].float()
        Y = ds["Y"]
        pid = ds["process_id"]

        # Build block target aligned to ds rows.
        block_targets = build_block_targets(rows, seq_len=cfg.seq_len)
        Xb, Yb = align_xy(ds, block_targets)

        panels_alr: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        panels_prob: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for p in (0, 1, 2):
            idx = torch.where(pid == p)[0]
            y_true = Y[idx]
            y_pred_alr = fit_predict_alr(X[idx], y_true, seed=SEED + p)
            y_pred_prob = fit_predict_prob(X[idx], y_true, seed=SEED + p)
            y_ta, y_pa = sample_pairs(y_true, y_pred_alr, MAX_POINTS, seed=SEED + 10 * p)
            y_tp, y_pp = sample_pairs(y_true, y_pred_prob, MAX_POINTS, seed=SEED + 10 * p)
            panels_alr[f"p{p}"] = (y_ta, y_pa)
            panels_prob[f"p{p}"] = (y_tp, y_pp)

        y_pred_block_alr = fit_predict_alr(Xb, Yb, seed=SEED + 777)
        y_pred_block_prob = fit_predict_prob(Xb, Yb, seed=SEED + 777)
        y_tb_a, y_pb_a = sample_pairs(Yb, y_pred_block_alr, MAX_POINTS, seed=SEED + 999)
        y_tb_p, y_pb_p = sample_pairs(Yb, y_pred_block_prob, MAX_POINTS, seed=SEED + 999)
        panels_alr["block"] = (y_tb_a, y_pb_a)
        panels_prob["block"] = (y_tb_p, y_pb_p)

        plot_side_by_side_simplex(
            panels_alr,
            OUT_DIR / f"E1_side_by_side_simplex_{case_name}_final_ln.png",
            f"E1 ALR | {case_name} | final_ln",
        )
        plot_side_by_side_simplex(
            panels_prob,
            OUT_DIR / f"E2_side_by_side_simplex_{case_name}_final_ln.png",
            f"E2 prob | {case_name} | final_ln",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure canonical no-PCA artifacts exist.
    run_if_missing(
        ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_alr_kl.json",
        [PY, "src/compare_alr_kl_nopca.py"],
    )
    run_if_missing(
        ROOT / "artifacts" / "residual_simplex" / "comparison_trained_vs_control_nopca_prob_kl.json",
        [PY, "src/compare_prob_kl_nopca.py"],
    )
    run_if_missing(
        ROOT / "artifacts" / "residual_simplex" / "rowspace_orthogonality_alr_nopca.json",
        [PY, "src/rowspace_orthogonality_alr_nopca.py"],
    )
    run_if_missing(ROOT / "artifacts" / "residual_simplex" / "alr_to_alr_kl.json", [PY, "src/alr_to_alr_regression.py"])
    run_if_missing(ROOT / "artifacts" / "residual_simplex" / "prob_to_prob_kl.json", [PY, "src/prob_to_prob_regression.py"])

    export_e1_e2_csvs()
    export_e3_csvs()
    export_e4_e5_csvs()
    make_e1_e2_side_by_side()

    print(f"Saved canonical outputs to: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

