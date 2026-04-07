#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import argparse
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
OUT_JSON_BASE = Path("artifacts/residual_simplex/intervention_eval_p1p2_subset")
OUT_CSV_BASE = Path("artifacts/residual_simplex/intervention_eval_p1p2_subset")

MAX_SEQS = 6000
ALPHA_TARGET_DEFAULT = 0.1
EPS = 1e-8
SEED = 0
TOL = 1e-8
BATCH = 128


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


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


def fit_maps(ds: dict, rows: List[dict], seq_len: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    X = ds["X"].float()
    Y = ds["Y"].float()
    pid = ds["process_id"]
    maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for p in (0, 1, 2):
        idx = torch.where(pid == p)[0]
        maps[f"p{p}"] = _fit_linear_map(X[idx], to_alr(Y[idx]))
    block_targets = build_block_targets(rows, seq_len=seq_len)
    Xb, Yb = align_xy(ds, block_targets)
    maps["block"] = _fit_linear_map(Xb.float(), to_alr(Yb.float()))
    return maps


def rowspace_basis(W: torch.Tensor, tol: float = TOL) -> torch.Tensor:
    _, s, vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    r = int((s > tol).sum().item())
    return vh[:r].T.contiguous() if r > 0 else torch.zeros((W.shape[1], 0), dtype=torch.float64)


def nullspace_basis(A: torch.Tensor, tol: float = TOL) -> torch.Tensor:
    _, s, vh = torch.linalg.svd(A.to(torch.float64), full_matrices=True)
    r = int((s > tol).sum().item())
    return vh[r:].T.contiguous()


def solve_delta(W_block: torch.Tensor, S: torch.Tensor, alpha: float) -> torch.Tensor:
    target = torch.tensor([alpha, alpha], dtype=torch.float64)
    A = W_block.to(torch.float64) @ S
    z = torch.linalg.lstsq(A, target).solution
    return (S @ z).to(torch.float32)


def make_balanced_p1p2(rows: List[dict], seq_len: int, seed: int) -> List[dict]:
    p1 = [r for r in rows if int(r["process_id"]) == 1 and len(r["tokens"]) >= seq_len + 1]
    p2 = [r for r in rows if int(r["process_id"]) == 2 and len(r["tokens"]) >= seq_len + 1]
    m = min(len(p1), len(p2))
    g = torch.Generator().manual_seed(seed)
    i1 = torch.randperm(len(p1), generator=g)[:m].tolist()
    i2 = torch.randperm(len(p2), generator=g)[:m].tolist()
    subset = [p1[i] for i in i1] + [p2[i] for i in i2]
    perm = torch.randperm(len(subset), generator=g).tolist()
    return [subset[i] for i in perm]


def eval_subset(model, cfg, rows_subset: List[dict], delta: torch.Tensor | None, maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    device = torch.device("cpu")
    nll_sum = 0.0
    tok_count = 0
    kl_b2i_sum = 0.0
    kl_i2b_sum = 0.0
    n_items = 0

    Wb, bb = maps["block"]
    W1, b1 = maps["p1"]
    W2, b2 = maps["p2"]

    block_shift = []
    p1_shift = []
    p2_shift = []

    for start in range(0, len(rows_subset), BATCH):
        chunk = rows_subset[start : start + BATCH]
        x = torch.tensor([r["tokens"][: cfg.seq_len] for r in chunk], dtype=torch.long, device=device)
        y = torch.tensor([r["tokens"][1 : cfg.seq_len + 1] for r in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, cache = model(x, capture_residuals=True)
            base_probs = torch.softmax(logits, dim=-1)
            res = cache["final_ln"]  # [B,T,d]
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

        # Symmetric reporting of distribution shift baseline vs intervention.
        if delta is not None:
            pb = base_probs.clamp_min(EPS)
            pi = int_probs.clamp_min(EPS)
            kl_b2i = (pb * (torch.log(pb) - torch.log(pi))).sum(dim=-1).mean().item()
            kl_i2b = (pi * (torch.log(pi) - torch.log(pb))).sum(dim=-1).mean().item()
            kl_b2i_sum += kl_b2i * x.shape[0]
            kl_i2b_sum += kl_i2b * x.shape[0]
            n_items += x.shape[0]

        # Map-space observed shifts.
        r0 = res.reshape(-1, res.shape[-1]).float()
        r1 = res_i.reshape(-1, res_i.shape[-1]).float()
        db = (r1 @ Wb.T + bb) - (r0 @ Wb.T + bb)
        d1 = (r1 @ W1.T + b1) - (r0 @ W1.T + b1)
        d2 = (r1 @ W2.T + b2) - (r0 @ W2.T + b2)
        block_shift.append(db)
        p1_shift.append(d1)
        p2_shift.append(d2)

    def summarize_shift(chunks: List[torch.Tensor]) -> dict:
        d = torch.cat(chunks, dim=0)
        c1 = float(d[:, 0].mean().item())
        c2 = float(d[:, 1].mean().item())
        return {"delta_c1_mean": c1, "delta_c2_mean": c2, "mean_shift": 0.5 * (c1 + c2), "coord_gap_abs": abs(c1 - c2)}

    out = {
        "nll_per_token": float(nll_sum / max(1, tok_count)),
        "ppl": float(math.exp(nll_sum / max(1, tok_count))),
        "block_shift": summarize_shift(block_shift),
        "p1_shift": summarize_shift(p1_shift),
        "p2_shift": summarize_shift(p2_shift),
    }
    if delta is not None and n_items > 0:
        out["token_dist_kl_base_to_int"] = float(kl_b2i_sum / n_items)
        out["token_dist_kl_int_to_base"] = float(kl_i2b_sum / n_items)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA_TARGET_DEFAULT)
    args = parser.parse_args()
    alpha_target = float(args.alpha)

    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    maps = fit_maps(ds, rows, seq_len=cfg.seq_len)

    Wb, _ = maps["block"]
    Wp0, _ = maps["p0"]
    Wp1, _ = maps["p1"]
    Wp2, _ = maps["p2"]

    S_a = rowspace_basis(Wb)
    delta_a = solve_delta(Wb, S_a, alpha_target)

    A_other = torch.cat([Wp0.float(), Wp1.float(), Wp2.float()], dim=0)
    S_b = nullspace_basis(A_other)
    delta_b = solve_delta(Wb, S_b, alpha_target)

    subset = make_balanced_p1p2(rows, seq_len=cfg.seq_len, seed=SEED)
    baseline = eval_subset(model, cfg, subset, delta=None, maps=maps)
    res_a = eval_subset(model, cfg, subset, delta=delta_a, maps=maps)
    res_b = eval_subset(model, cfg, subset, delta=delta_b, maps=maps)

    suffix = f"_alpha{alpha_target:g}"
    out_json = OUT_JSON_BASE.with_name(OUT_JSON_BASE.name + suffix + ".json")
    out_csv = OUT_CSV_BASE.with_name(OUT_CSV_BASE.name + suffix + ".csv")

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "subset": "process_id in {1,2}, balanced 50/50",
            "subset_size_sequences": int(len(subset)),
            "layer": "final_ln",
            "coords": "ALR",
            "target_shift": [alpha_target, alpha_target],
        },
        "baseline_no_intervention": baseline,
        "method_A_rowspace_block": res_a,
        "method_B_nullspace_others": res_b,
        "deltas": {
            "delta_A_l2": float(torch.norm(delta_a).item()),
            "delta_B_l2": float(torch.norm(delta_b).item()),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Compact CSV summary.
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("method,nll_per_token,ppl,block_mean_shift,p1_mean_shift,p2_mean_shift,token_kl_base_to_int,token_kl_int_to_base\n")
        f.write(
            f"baseline,{baseline['nll_per_token']:.8f},{baseline['ppl']:.8f},"
            f"{baseline['block_shift']['mean_shift']:.8f},{baseline['p1_shift']['mean_shift']:.8f},"
            f"{baseline['p2_shift']['mean_shift']:.8f},,\n"
        )
        f.write(
            f"A_rowspace_block,{res_a['nll_per_token']:.8f},{res_a['ppl']:.8f},"
            f"{res_a['block_shift']['mean_shift']:.8f},{res_a['p1_shift']['mean_shift']:.8f},"
            f"{res_a['p2_shift']['mean_shift']:.8f},{res_a['token_dist_kl_base_to_int']:.8f},{res_a['token_dist_kl_int_to_base']:.8f}\n"
        )
        f.write(
            f"B_nullspace_others,{res_b['nll_per_token']:.8f},{res_b['ppl']:.8f},"
            f"{res_b['block_shift']['mean_shift']:.8f},{res_b['p1_shift']['mean_shift']:.8f},"
            f"{res_b['p2_shift']['mean_shift']:.8f},{res_b['token_dist_kl_base_to_int']:.8f},{res_b['token_dist_kl_int_to_base']:.8f}\n"
        )

    print(f"saved {out_json}")
    print(f"saved {out_csv}")
    print("\nsubset:", out["config"]["subset"], "N=", out["config"]["subset_size_sequences"])
    print(f"baseline nll={baseline['nll_per_token']:.6f} ppl={baseline['ppl']:.6f}")
    for name, res in (("A(rowspace block)", res_a), ("B(nullspace others)", res_b)):
        print(
            f"{name}: nll={res['nll_per_token']:.6f} ppl={res['ppl']:.6f} "
            f"block_mean_shift={res['block_shift']['mean_shift']:.6f} "
            f"p1/p2_mean_shift={res['p1_shift']['mean_shift']:.6f}/{res['p2_shift']['mean_shift']:.6f}"
        )


if __name__ == "__main__":
    main()
