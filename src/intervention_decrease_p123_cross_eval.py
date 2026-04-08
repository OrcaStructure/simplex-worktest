#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import align_xy, build_block_targets
from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import _fit_linear_map, load_checkpoint, load_rows


DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
CHECKPOINT = Path("artifacts/tiny_transformer.pt")
OUT_JSON = Path("artifacts/residual_simplex/intervention_decrease_p123_cross_eval.json")
OUT_CSV = Path("artifacts/residual_simplex/intervention_decrease_p123_cross_eval.csv")

MAX_SEQS = 6000
BATCH = 128
EPS = 1e-8
OPT_STEPS = 800
OPT_LR = 0.2
OPT_L2 = 0.02


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def from_alr(c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([c, torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)], dim=1)
    return torch.softmax(z, dim=1)


def build_residual_dataset_from_cache_key(model, cfg, rows: List[dict], cache_key: str) -> dict:
    xs, ys, pids, sids, poss = [], [], [], [], []
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
        residuals = cache[cache_key][0].detach().cpu()
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


def fit_block_map(ds: dict, rows: List[dict], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    block_targets = build_block_targets(rows, seq_len=seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    Wb, bb = _fit_linear_map(Xb.float(), to_alr(Yb.float()))
    return Wb, bb


def optimize_alr_shift_for_decrease(c_train: torch.Tensor, target_idx: int) -> torch.Tensor:
    # Optimize direct ALR shift d (2D) to reduce mean p_target over training decoded points.
    d = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([d], lr=OPT_LR)
    c0 = c_train.to(torch.float64)
    best = None
    for _ in range(OPT_STEPS):
        p = from_alr(c0 + d.view(1, 2)).to(torch.float64)
        loss = p[:, target_idx].mean() + OPT_L2 * (d.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = d.detach().clone()
        if best is None or float(loss.item()) < best[0]:
            best = (float(loss.item()), cur)
    if best is None:
        raise RuntimeError("Failed optimizing ALR shift")
    return best[1].to(torch.float32)


def min_norm_delta_from_alr_shift(Wb: torch.Tensor, d_alr: torch.Tensor) -> torch.Tensor:
    # Solve Wb * delta ~= d_alr with minimum-norm least squares solution.
    # Wb: [2, d_model], d_alr: [2]
    sol = torch.linalg.lstsq(Wb.to(torch.float64), d_alr.to(torch.float64)).solution
    return sol.to(torch.float32)


def evaluate_on_rows(
    model,
    cfg,
    rows_subset: List[dict],
    Wb: torch.Tensor,
    bb: torch.Tensor,
    delta: torch.Tensor | None,
) -> dict:
    nll_sum = 0.0
    tok_count = 0
    kl_b2i_sum = 0.0
    kl_i2b_sum = 0.0
    n_items = 0
    p_base_chunks: List[torch.Tensor] = []
    p_int_chunks: List[torch.Tensor] = []

    for start in range(0, len(rows_subset), BATCH):
        chunk = rows_subset[start : start + BATCH]
        x = torch.tensor([r["tokens"][: cfg.seq_len] for r in chunk], dtype=torch.long)
        y = torch.tensor([r["tokens"][1 : cfg.seq_len + 1] for r in chunk], dtype=torch.long)
        with torch.no_grad():
            logits, cache = model(x, capture_residuals=True)
            res = cache["final_ln"]  # [B,T,d]
            base_probs = torch.softmax(logits, dim=-1)
            if delta is None:
                logits_i = logits
                res_i = res
                int_probs = base_probs
            else:
                res_i = res + delta.view(1, 1, -1)
                logits_i = model.unembed(res_i)
                int_probs = torch.softmax(logits_i, dim=-1)

        nll = F.cross_entropy(logits_i.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum").item()
        nll_sum += nll
        tok_count += int(y.numel())

        pb_tok = base_probs.clamp_min(EPS)
        pi_tok = int_probs.clamp_min(EPS)
        kl_b2i = (pb_tok * (torch.log(pb_tok) - torch.log(pi_tok))).sum(dim=-1).mean().item()
        kl_i2b = (pi_tok * (torch.log(pi_tok) - torch.log(pb_tok))).sum(dim=-1).mean().item()
        kl_b2i_sum += kl_b2i * x.shape[0]
        kl_i2b_sum += kl_i2b * x.shape[0]
        n_items += x.shape[0]

        r0 = res.reshape(-1, res.shape[-1]).float()
        r1 = res_i.reshape(-1, res_i.shape[-1]).float()
        c0 = r0 @ Wb.T + bb
        c1 = r1 @ Wb.T + bb
        p0 = from_alr(c0)
        p1 = from_alr(c1)
        p_base_chunks.append(p0.detach().cpu())
        p_int_chunks.append(p1.detach().cpu())

    p_base = torch.cat(p_base_chunks, dim=0)
    p_int = torch.cat(p_int_chunks, dim=0)
    mean_base = p_base.mean(dim=0)
    mean_int = p_int.mean(dim=0)
    mean_delta = mean_int - mean_base

    return {
        "nll_per_token": float(nll_sum / max(1, tok_count)),
        "ppl": float(math.exp(nll_sum / max(1, tok_count))),
        "token_dist_kl_base_to_int": float(kl_b2i_sum / max(1, n_items)),
        "token_dist_kl_int_to_base": float(kl_i2b_sum / max(1, n_items)),
        "mean_prob_base": [float(v.item()) for v in mean_base],
        "mean_prob_int": [float(v.item()) for v in mean_int],
        "mean_prob_delta": [float(v.item()) for v in mean_delta],
    }


def main() -> None:
    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)

    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    Wb, bb = fit_block_map(ds, rows, seq_len=cfg.seq_len)

    # "Training" for steering vectors on full mixed dataset: optimize ALR shifts then map to residual deltas.
    c_train = ds["X"].float() @ Wb.T + bb
    alr_shifts: Dict[int, torch.Tensor] = {}
    deltas: Dict[int, torch.Tensor] = {}
    for j in (0, 1, 2):
        d_j = optimize_alr_shift_for_decrease(c_train, target_idx=j)
        delta_j = min_norm_delta_from_alr_shift(Wb, d_j)
        alr_shifts[j] = d_j
        deltas[j] = delta_j

    # Three corresponding datasets: one per process id.
    datasets = {j: [r for r in rows if int(r["process_id"]) == j and len(r["tokens"]) >= cfg.seq_len + 1] for j in (0, 1, 2)}

    # Cross-evaluate: each steering vector on each dataset (+ baseline for each dataset).
    baseline_by_dataset: Dict[int, dict] = {}
    cross: Dict[str, dict] = {}
    for ds_id, rows_ds in datasets.items():
        baseline_by_dataset[ds_id] = evaluate_on_rows(model, cfg, rows_ds, Wb=Wb, bb=bb, delta=None)
        for steer_id, delta in deltas.items():
            key = f"dataset_p{ds_id}__steer_dec_p{steer_id}"
            cross[key] = evaluate_on_rows(model, cfg, rows_ds, Wb=Wb, bb=bb, delta=delta)

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "max_sequences": MAX_SEQS,
            "objective": "for each j in {0,1,2}, learn steering to decrease decoded mean p_j on mixed data",
            "optimizer": {"steps": OPT_STEPS, "lr": OPT_LR, "l2": OPT_L2},
        },
        "trained_on": {
            "num_sequences": int(len(rows)),
            "num_tokens": int(ds["X"].shape[0]),
            "alr_shifts": {f"dec_p{j}": [float(v.item()) for v in alr_shifts[j]] for j in (0, 1, 2)},
            "delta_norms_l2": {f"dec_p{j}": float(torch.norm(deltas[j]).item()) for j in (0, 1, 2)},
        },
        "datasets": {f"p{j}": {"num_sequences": int(len(datasets[j]))} for j in (0, 1, 2)},
        "baseline_by_dataset": {f"p{j}": baseline_by_dataset[j] for j in (0, 1, 2)},
        "cross_eval": cross,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Compact matrix CSV: rows=dataset, cols=steering target, values are mean_prob_delta for p1,p2,p3.
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("dataset,steer_target,nll_per_token,ppl,delta_p1,delta_p2,delta_p3,token_kl_base_to_int\n")
        for ds_id in (0, 1, 2):
            base = baseline_by_dataset[ds_id]
            f.write(
                f"p{ds_id},baseline,{base['nll_per_token']:.8f},{base['ppl']:.8f},0.00000000,0.00000000,0.00000000,0.00000000\n"
            )
            for steer_id in (0, 1, 2):
                key = f"dataset_p{ds_id}__steer_dec_p{steer_id}"
                r = cross[key]
                d = r["mean_prob_delta"]
                f.write(
                    f"p{ds_id},dec_p{steer_id},{r['nll_per_token']:.8f},{r['ppl']:.8f},"
                    f"{d[0]:.8f},{d[1]:.8f},{d[2]:.8f},{r['token_dist_kl_base_to_int']:.8f}\n"
                )

    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_CSV}")
    print("delta norms:", {f"dec_p{j}": float(torch.norm(deltas[j]).item()) for j in (0, 1, 2)})


if __name__ == "__main__":
    main()
