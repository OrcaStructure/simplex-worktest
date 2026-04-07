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
OUT_BASE = Path("artifacts/residual_simplex/intervention_eval_pair_adaptive")

MAX_SEQS = 6000
SEQ_SEED = 0
EPS = 1e-8
TOL = 1e-8
BATCH = 128

# Adaptive policy target for decoded block p3 after intervention.
TARGET_P3 = 0.20
# Safety cap for per-token alpha.
ALPHA_MAX = 2.0
# Fixed baselines to compare against.
FIXED_ALPHAS = (0.1, 0.5, 1.0)


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
        residuals = cache["final_ln"][0].detach().cpu()
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


def solve_delta_unit(W_block: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # Solve for delta giving approximately +1,+1 ALR shift in block map.
    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    A = W_block.to(torch.float64) @ S
    z = torch.linalg.lstsq(A, target).solution
    return (S @ z).to(torch.float32)


def make_balanced_pair(rows: List[dict], seq_len: int, seed: int, p_a: int, p_b: int) -> List[dict]:
    a_rows = [r for r in rows if int(r["process_id"]) == p_a and len(r["tokens"]) >= seq_len + 1]
    b_rows = [r for r in rows if int(r["process_id"]) == p_b and len(r["tokens"]) >= seq_len + 1]
    m = min(len(a_rows), len(b_rows))
    g = torch.Generator().manual_seed(seed)
    i1 = torch.randperm(len(a_rows), generator=g)[:m].tolist()
    i2 = torch.randperm(len(b_rows), generator=g)[:m].tolist()
    subset = [a_rows[i] for i in i1] + [b_rows[i] for i in i2]
    perm = torch.randperm(len(subset), generator=g).tolist()
    return [subset[i] for i in perm]


def alpha_from_block_p3(p3: torch.Tensor, target_p3: float, alpha_max: float) -> torch.Tensor:
    # Choose smallest nonnegative alpha that would map p3 -> target_p3 under common ALR shift.
    # p3' = p3 / (exp(alpha)*(1-p3) + p3)
    tp = torch.full_like(p3, float(target_p3))
    need = p3 > tp
    numer = p3 * (1.0 - tp)
    denom = tp * (1.0 - p3)
    a = torch.log(torch.clamp(numer / torch.clamp(denom, min=1e-12), min=1e-12))
    a = torch.where(need, a, torch.zeros_like(a))
    return torch.clamp(a, min=0.0, max=alpha_max)


def eval_policy(
    model,
    cfg,
    rows_subset: List[dict],
    maps: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    delta_unit: torch.Tensor,
    mode: str,
) -> dict:
    """
    mode:
      - baseline
      - fixed:<alpha>
      - adaptive
    """
    device = torch.device("cpu")
    nll_sum = 0.0
    tok_count = 0
    kl_b2i_sum = 0.0
    kl_i2b_sum = 0.0
    n_items = 0
    alpha_vals: List[torch.Tensor] = []

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

            if mode == "baseline":
                res_i = res
                alphas = torch.zeros((res.shape[0], res.shape[1]), dtype=res.dtype)
            elif mode.startswith("fixed:"):
                a = float(mode.split(":", 1)[1])
                alphas = torch.full((res.shape[0], res.shape[1]), a, dtype=res.dtype)
                res_i = res + alphas.unsqueeze(-1) * delta_unit.view(1, 1, -1)
            elif mode == "adaptive":
                c = res.reshape(-1, res.shape[-1]).float() @ Wb.T + bb  # [B*T,2]
                p = from_alr(c)
                p3 = p[:, 2]
                a = alpha_from_block_p3(p3, target_p3=TARGET_P3, alpha_max=ALPHA_MAX)
                alphas = a.view(res.shape[0], res.shape[1]).to(res.dtype)
                res_i = res + alphas.unsqueeze(-1) * delta_unit.view(1, 1, -1)
            else:
                raise ValueError(f"unknown mode: {mode}")

            logits_i = model.unembed(res_i)
            int_probs = torch.softmax(logits_i, dim=-1)

        alpha_vals.append(alphas.reshape(-1).detach().cpu())

        nll = F.cross_entropy(logits_i.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum").item()
        nll_sum += nll
        tok_count += int(y.numel())

        if mode != "baseline":
            pb = base_probs.clamp_min(EPS)
            pi = int_probs.clamp_min(EPS)
            kl_b2i = (pb * (torch.log(pb) - torch.log(pi))).sum(dim=-1).mean().item()
            kl_i2b = (pi * (torch.log(pi) - torch.log(pb))).sum(dim=-1).mean().item()
            kl_b2i_sum += kl_b2i * x.shape[0]
            kl_i2b_sum += kl_i2b * x.shape[0]
            n_items += x.shape[0]

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

    a_all = torch.cat(alpha_vals, dim=0)
    out = {
        "mode": mode,
        "nll_per_token": float(nll_sum / max(1, tok_count)),
        "ppl": float(math.exp(nll_sum / max(1, tok_count))),
        "alpha_mean": float(a_all.mean().item()),
        "alpha_median": float(a_all.median().item()),
        "alpha_nonzero_frac": float((a_all > 0).float().mean().item()),
        "block_shift": summarize_shift(block_shift),
        "p1_shift": summarize_shift(p1_shift),
        "p2_shift": summarize_shift(p2_shift),
    }
    if mode != "baseline" and n_items > 0:
        out["token_dist_kl_base_to_int"] = float(kl_b2i_sum / n_items)
        out["token_dist_kl_int_to_base"] = float(kl_i2b_sum / n_items)
    return out


def main() -> None:
    p_a, p_b = 1, 2  # default pair
    if len(sys.argv) >= 2 and "," in sys.argv[1]:
        s1, s2 = sys.argv[1].split(",", 1)
        p_a, p_b = int(s1), int(s2)

    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    maps = fit_maps(ds, rows, seq_len=cfg.seq_len)

    Wb, _ = maps["block"]
    Wp0, _ = maps["p0"]
    Wp1, _ = maps["p1"]
    Wp2, _ = maps["p2"]
    A_other = torch.cat([Wp0.float(), Wp1.float(), Wp2.float()], dim=0)
    S_b = nullspace_basis(A_other)
    delta_unit = solve_delta_unit(Wb, S_b)  # "clean" block direction

    subset = make_balanced_pair(rows, seq_len=cfg.seq_len, seed=SEQ_SEED, p_a=p_a, p_b=p_b)

    baseline = eval_policy(model, cfg, subset, maps, delta_unit, "baseline")
    fixed = {str(a): eval_policy(model, cfg, subset, maps, delta_unit, f"fixed:{a}") for a in FIXED_ALPHAS}
    adaptive = eval_policy(model, cfg, subset, maps, delta_unit, "adaptive")

    tag = f"_{p_a}{p_b}"
    out_json = OUT_BASE.with_name(OUT_BASE.name + tag + ".json")
    out_csv = OUT_BASE.with_name(OUT_BASE.name + tag + ".csv")

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "subset": f"process_id in {{{p_a},{p_b}}}, balanced 50/50",
            "subset_size_sequences": int(len(subset)),
            "layer": "final_ln",
            "coords": "ALR",
            "adaptive_target_p3": TARGET_P3,
            "adaptive_alpha_max": ALPHA_MAX,
            "fixed_alphas": list(FIXED_ALPHAS),
        },
        "baseline": baseline,
        "fixed": fixed,
        "adaptive": adaptive,
        "delta_unit_l2": float(torch.norm(delta_unit).item()),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("mode,nll_per_token,ppl,alpha_mean,alpha_median,alpha_nonzero_frac,block_mean_shift,p1_mean_shift,p2_mean_shift,token_kl_base_to_int,token_kl_int_to_base\n")
        def line(name: str, r: dict):
            f.write(
                f"{name},{r['nll_per_token']:.8f},{r['ppl']:.8f},"
                f"{r['alpha_mean']:.8f},{r['alpha_median']:.8f},{r['alpha_nonzero_frac']:.8f},"
                f"{r['block_shift']['mean_shift']:.8f},{r['p1_shift']['mean_shift']:.8f},{r['p2_shift']['mean_shift']:.8f},"
                f"{r.get('token_dist_kl_base_to_int','')},{r.get('token_dist_kl_int_to_base','')}\n"
            )
        line("baseline", baseline)
        for a in FIXED_ALPHAS:
            line(f"fixed_{a}", fixed[str(a)])
        line("adaptive", adaptive)

    print(f"saved {out_json}")
    print(f"saved {out_csv}")
    print(f"\nsubset N={len(subset)}")
    print(f"baseline nll={baseline['nll_per_token']:.6f} ppl={baseline['ppl']:.6f}")
    for a in FIXED_ALPHAS:
        r = fixed[str(a)]
        print(
            f"fixed {a}: nll={r['nll_per_token']:.6f} ppl={r['ppl']:.6f} "
            f"alpha_mean={r['alpha_mean']:.4f} block_shift={r['block_shift']['mean_shift']:.4f}"
        )
    print(
        f"adaptive: nll={adaptive['nll_per_token']:.6f} ppl={adaptive['ppl']:.6f} "
        f"alpha_mean={adaptive['alpha_mean']:.4f} nz={adaptive['alpha_nonzero_frac']:.3f} "
        f"block_shift={adaptive['block_shift']['mean_shift']:.4f}"
    )


if __name__ == "__main__":
    main()
