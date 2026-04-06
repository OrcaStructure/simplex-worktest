#!/usr/bin/env python3
"""Residual-to-simplex linear regression pipeline.

Pipeline:
1) Load trained transformer checkpoint.
2) Run inference on many sequences from mixed Mess3 data.
3) Extract residual stream vectors right before unembedding (final_ln).
4) Build supervised dataset: residual -> true belief simplex point.
5) Fit one linear map (W,b) per Mess3 process via least squares.
6) Save datasets, fitted parameters, metrics, and trajectory plots.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hmm_process.mess3 import Mess3Process
from src.hmm_process.simplex_plot import barycentric_to_cartesian, simplex_vertices
from src.simple_transformer_residual import Config, TinyTransformer


@dataclass
class PipelineConfig:
    checkpoint_path: str = "artifacts/tiny_transformer.pt"
    dataset_path: str = "src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"
    out_dir: str = "artifacts/residual_simplex"
    device: str = "cpu"

    max_sequences: int = 6000
    train_frac: float = 0.8
    seed: int = 0
    use_cached_residual_dataset: bool = True
    reduced_dim: int = 8  # Aggressive dimensionality reduction for regression features.

    plot_sequences_per_process_linear: int = 200
    plot_sequences_per_process_ground_truth: int = 3000


def load_checkpoint(path: Path, device: torch.device) -> Tuple[TinyTransformer, Config, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def load_rows(path: Path, max_sequences: int) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= max_sequences:
                break
    if not rows:
        raise ValueError("No dataset rows loaded")
    return rows


def _process_from_row(row: dict) -> Mess3Process:
    return Mess3Process(alpha=float(row["alpha"]), x=float(row["x"]), dtype=torch.float64, device="cpu")


def build_residual_dataset(
    model: TinyTransformer,
    cfg: Config,
    rows: List[dict],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    process_ids: List[int] = []
    sequence_ids: List[int] = []
    positions: List[int] = []

    for seq_idx, row in enumerate(rows):
        tokens = [int(t) for t in row["tokens"]]
        pid = int(row["process_id"])
        if len(tokens) < cfg.seq_len:
            continue

        x_tokens = tokens[: cfg.seq_len]

        with torch.no_grad():
            inp = torch.tensor(x_tokens, dtype=torch.long, device=device).unsqueeze(0)
            _, cache = model(inp, capture_residuals=True)
            residuals = cache["final_ln"][0].detach().cpu()  # [T, d_model]

        process = _process_from_row(row)
        true_traj = process.belief_trajectory(x_tokens)  # [T+1, 3], belief after each observed x_t is index t+1

        for pos in range(cfg.seq_len):
            xs.append(residuals[pos])
            ys.append(true_traj[pos + 1].to(dtype=torch.float32))
            process_ids.append(pid)
            sequence_ids.append(seq_idx)
            positions.append(pos)

    if not xs:
        raise ValueError("No residual samples built")

    return {
        "X": torch.stack(xs, dim=0).to(dtype=torch.float32),
        "Y": torch.stack(ys, dim=0).to(dtype=torch.float32),
        "process_id": torch.tensor(process_ids, dtype=torch.long),
        "sequence_id": torch.tensor(sequence_ids, dtype=torch.long),
        "position": torch.tensor(positions, dtype=torch.long),
    }


def _fit_linear_map(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Solve [X,1] B = Y by least squares. B shape [d+1, 3].
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype)
    XA = torch.cat([X, ones], dim=1)
    B = torch.linalg.lstsq(XA, Y).solution
    W = B[:-1].T.contiguous()  # [3, d]
    b = B[-1].contiguous()  # [3]
    return W, b


def _project_to_simplex(prob_like: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z = torch.clamp(prob_like, min=eps)
    return z / z.sum(dim=-1, keepdim=True)


def _split_indices(n: int, train_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, min(n - 1, int(n * train_frac)))
    return perm[:n_train], perm[n_train:]


def reduce_features_with_pca(X_raw: torch.Tensor, reduced_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = X_raw.shape[1]
    k = max(1, min(reduced_dim, d))
    mean = X_raw.mean(dim=0, keepdim=True)
    xc = X_raw - mean
    # Principal directions in feature space.
    _, _, v = torch.pca_lowrank(xc, q=k, center=False)
    X_red = xc @ v[:, :k]
    return X_red, mean.squeeze(0), v[:, :k]


def fit_per_process_maps(
    residual_ds: Dict[str, torch.Tensor],
    train_frac: float,
    seed: int,
) -> Dict[int, dict]:
    X = residual_ds["X"]
    Y = residual_ds["Y"]
    pid = residual_ds["process_id"]

    out: Dict[int, dict] = {}
    for p in sorted(int(v) for v in pid.unique().tolist()):
        idx = torch.where(pid == p)[0]
        Xp = X[idx]
        Yp = Y[idx]
        tr, va = _split_indices(Xp.shape[0], train_frac=train_frac, seed=seed + p)

        W, b = _fit_linear_map(Xp[tr], Yp[tr])

        pred_train = Xp[tr] @ W.T + b
        pred_val = Xp[va] @ W.T + b

        pred_train_s = _project_to_simplex(pred_train)
        pred_val_s = _project_to_simplex(pred_val)

        mse_train = torch.mean((pred_train_s - Yp[tr]) ** 2).item()
        mse_val = torch.mean((pred_val_s - Yp[va]) ** 2).item()
        mae_val = torch.mean(torch.abs(pred_val_s - Yp[va])).item()

        # Cross-entropy of true beliefs under predicted simplex probs (soft-target CE).
        ce_val = (-(Yp[va] * torch.log(pred_val_s + 1e-8)).sum(dim=1)).mean().item()

        out[p] = {
            "W": W,
            "b": b,
            "metrics": {
                "num_samples": int(Xp.shape[0]),
                "num_train": int(tr.shape[0]),
                "num_val": int(va.shape[0]),
                "mse_train": mse_train,
                "mse_val": mse_val,
                "mae_val": mae_val,
                "soft_ce_val": ce_val,
            },
        }

    return out


def _build_joint_targets(Y_local: torch.Tensor, process_id: torch.Tensor, num_processes: int = 3) -> torch.Tensor:
    # Map local 3-simplex target to joint 9-simplex target with block structure.
    n = Y_local.shape[0]
    y_joint = torch.zeros((n, num_processes * 3), dtype=Y_local.dtype)
    for p in range(num_processes):
        idx = torch.where(process_id == p)[0]
        if idx.numel() == 0:
            continue
        y_joint[idx, 3 * p : 3 * (p + 1)] = Y_local[idx]
    return y_joint


def fit_joint_8_simplex_map(
    residual_ds: Dict[str, torch.Tensor],
    train_frac: float,
    seed: int,
) -> dict:
    X = residual_ds["X"]
    Y_local = residual_ds["Y"]
    pid = residual_ds["process_id"]
    Y_joint = _build_joint_targets(Y_local, pid, num_processes=3)

    tr, va = _split_indices(X.shape[0], train_frac=train_frac, seed=seed + 123)
    W, b = _fit_linear_map(X[tr], Y_joint[tr])  # W: [9,d], b: [9]

    pred_train = _project_to_simplex(X[tr] @ W.T + b)
    pred_val = _project_to_simplex(X[va] @ W.T + b)

    y_tr = Y_joint[tr]
    y_va = Y_joint[va]

    mse_train = torch.mean((pred_train - y_tr) ** 2).item()
    mse_val = torch.mean((pred_val - y_va) ** 2).item()
    mae_val = torch.mean(torch.abs(pred_val - y_va)).item()
    ce_val = (-(y_va * torch.log(pred_val + 1e-8)).sum(dim=1)).mean().item()

    # Block-level sanity: mass per process block should identify the process.
    block_val = pred_val.view(pred_val.shape[0], 3, 3).sum(dim=2)  # [N,3]
    pred_pid = torch.argmax(block_val, dim=1)
    true_pid = pid[va]
    block_acc = (pred_pid == true_pid).float().mean().item()

    true_block_mass = torch.gather(block_val, dim=1, index=true_pid.unsqueeze(1)).squeeze(1)
    true_block_mass_mean = true_block_mass.mean().item()

    return {
        "W": W,
        "b": b,
        "metrics": {
            "num_samples": int(X.shape[0]),
            "num_train": int(tr.shape[0]),
            "num_val": int(va.shape[0]),
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mae_val": mae_val,
            "soft_ce_val": ce_val,
            "process_block_acc_val": block_acc,
            "true_process_block_mass_mean_val": true_block_mass_mean,
        },
    }


def save_metrics_json(per_process: Dict[int, dict], out_path: Path) -> None:
    payload = {
        str(pid): {
            "metrics": data["metrics"],
            "W_shape": list(data["W"].shape),
            "b_shape": list(data["b"].shape),
        }
        for pid, data in per_process.items()
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_true_and_pred_trajectories_separately(
    residual_ds: Dict[str, torch.Tensor],
    per_process: Dict[int, dict],
    out_dir: Path,
    max_sequences_per_process_linear: int,
    max_sequences_per_process_ground_truth: int,
) -> None:
    import matplotlib.pyplot as plt

    X = residual_ds["X"]
    Y = residual_ds["Y"]
    pid = residual_ds["process_id"]
    sid = residual_ds["sequence_id"]
    pos = residual_ds["position"]

    verts = simplex_vertices()
    triangle = torch.vstack([verts, verts[0:1]])

    for p, pdata in per_process.items():
        W = pdata["W"]
        b = pdata["b"]

        seq_ids_all = torch.unique(sid[pid == p]).tolist()
        seq_ids_true = seq_ids_all[:max_sequences_per_process_ground_truth]
        seq_ids_pred = seq_ids_all[:max_sequences_per_process_linear]

        fig_true, ax_true = plt.subplots(figsize=(7, 6))
        fig_pred, ax_pred = plt.subplots(figsize=(7, 6))
        for ax in (ax_true, ax_pred):
            ax.plot(triangle[:, 0].numpy(), triangle[:, 1].numpy(), color="black", linewidth=1.5)

        for seq_id in seq_ids_true:
            idx = torch.where((pid == p) & (sid == seq_id))[0]
            order = torch.argsort(pos[idx])
            idx = idx[order]

            y_true = Y[idx]  # [T,3]
            xy_true = barycentric_to_cartesian(y_true.to(dtype=torch.float64))
            ax_true.scatter(xy_true[:, 0].numpy(), xy_true[:, 1].numpy(), color="#1f77b4", alpha=0.07, s=6)

        for seq_id in seq_ids_pred:
            idx = torch.where((pid == p) & (sid == seq_id))[0]
            order = torch.argsort(pos[idx])
            idx = idx[order]

            y_pred = _project_to_simplex(X[idx] @ W.T + b)
            xy_pred = barycentric_to_cartesian(y_pred.to(dtype=torch.float64))
            ax_pred.scatter(xy_pred[:, 0].numpy(), xy_pred[:, 1].numpy(), color="#d62728", alpha=0.25, s=8)

        for ax, title in (
            (ax_true, f"Process {p}: ground-truth points"),
            (ax_pred, f"Process {p}: linear-map points"),
        ):
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xlabel("simplex x")
            ax.set_ylabel("simplex y")
            ax.grid(alpha=0.2)

        out_true = out_dir / f"process_{p}_ground_truth.png"
        out_pred = out_dir / f"process_{p}_linear_map.png"
        fig_true.savefig(out_true, dpi=180, bbox_inches="tight")
        fig_pred.savefig(out_pred, dpi=180, bbox_inches="tight")
        plt.close(fig_true)
        plt.close(fig_pred)


def main() -> None:
    cfg = PipelineConfig()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_cache_path = out_dir / "residual_dataset.pt"

    if cfg.use_cached_residual_dataset and residual_cache_path.exists():
        residual_ds = torch.load(residual_cache_path, map_location="cpu")
        print(f"Loaded cached residual dataset: {residual_cache_path}")
    else:
        device = torch.device(cfg.device)
        model, model_cfg, _ = load_checkpoint(Path(cfg.checkpoint_path), device)
        rows = load_rows(Path(cfg.dataset_path), cfg.max_sequences)
        residual_ds = build_residual_dataset(model, model_cfg, rows, device)
        torch.save(residual_ds, residual_cache_path)
        print(f"Built and cached residual dataset: {residual_cache_path}")

    X_raw = residual_ds.get("X_raw", residual_ds["X"]).to(dtype=torch.float32)
    X_red, pca_mean, pca_components = reduce_features_with_pca(X_raw, reduced_dim=cfg.reduced_dim)
    residual_ds = dict(residual_ds)
    residual_ds["X_raw"] = X_raw
    residual_ds["X"] = X_red
    residual_ds["pca_mean"] = pca_mean
    residual_ds["pca_components"] = pca_components
    residual_ds["reduced_dim"] = torch.tensor([cfg.reduced_dim], dtype=torch.long)
    torch.save(residual_ds, residual_cache_path)
    torch.save(
        {"mean": pca_mean, "components": pca_components, "reduced_dim": cfg.reduced_dim},
        out_dir / "pca_reduction.pt",
    )

    per_process = fit_per_process_maps(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)
    joint_map = fit_joint_8_simplex_map(residual_ds, train_frac=cfg.train_frac, seed=cfg.seed)

    params_to_save = {
        int(pid): {
            "W": pdata["W"],
            "b": pdata["b"],
            "metrics": pdata["metrics"],
        }
        for pid, pdata in per_process.items()
    }
    torch.save(params_to_save, out_dir / "linear_maps_per_process.pt")
    save_metrics_json(per_process, out_dir / "linear_maps_metrics.json")
    torch.save(joint_map, out_dir / "linear_map_joint_8_simplex.pt")
    (out_dir / "joint_8_simplex_metrics.json").write_text(
        json.dumps(
            {
                "metrics": joint_map["metrics"],
                "W_shape": list(joint_map["W"].shape),
                "b_shape": list(joint_map["b"].shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_true_and_pred_trajectories_separately(
        residual_ds,
        per_process,
        out_dir=out_dir,
        max_sequences_per_process_linear=cfg.plot_sequences_per_process_linear,
        max_sequences_per_process_ground_truth=cfg.plot_sequences_per_process_ground_truth,
    )

    print("Saved:")
    print(f"  residual dataset: {out_dir / 'residual_dataset.pt'}")
    print(f"  pca reduction:    {out_dir / 'pca_reduction.pt'}")
    print(f"  linear maps:      {out_dir / 'linear_maps_per_process.pt'}")
    print(f"  metrics json:     {out_dir / 'linear_maps_metrics.json'}")
    print(f"  joint 8-simplex:  {out_dir / 'linear_map_joint_8_simplex.pt'}")
    print(f"  joint metrics:    {out_dir / 'joint_8_simplex_metrics.json'}")
    for pid, pdata in sorted(per_process.items()):
        m = pdata["metrics"]
        print(
            f"  process {pid}: n={m['num_samples']} mse_val={m['mse_val']:.6f} "
            f"mae_val={m['mae_val']:.6f} soft_ce_val={m['soft_ce_val']:.6f}"
        )
    jm = joint_map["metrics"]
    print(
        "  joint(8-simplex): "
        f"mse_val={jm['mse_val']:.6f} mae_val={jm['mae_val']:.6f} "
        f"soft_ce_val={jm['soft_ce_val']:.6f} "
        f"process_block_acc_val={jm['process_block_acc_val']:.6f} "
        f"true_process_block_mass_mean_val={jm['true_process_block_mass_mean_val']:.6f}"
    )


if __name__ == "__main__":
    main()
