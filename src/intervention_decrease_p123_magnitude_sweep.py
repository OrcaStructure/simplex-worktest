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
OUT_JSON = Path("artifacts/residual_simplex/intervention_decrease_p123_magnitude_sweep.json")
OUT_CSV = Path("artifacts/residual_simplex/intervention_decrease_p123_magnitude_sweep.csv")

MAX_SEQS = 6000
BATCH = 128
EPS = 1e-8
OPT_STEPS = 800
OPT_LR = 0.2
OPT_L2 = 0.02
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def from_alr(c: torch.Tensor) -> torch.Tensor:
    z = torch.cat([c, torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)], dim=1)
    return torch.softmax(z, dim=1)


def build_residual_dataset_from_cache_key(model, cfg, rows: List[dict], cache_key: str) -> dict:
    xs, ys, pids = [], [], []
    model.eval()
    for row in rows:
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
    return {
        "X": torch.stack(xs, dim=0),
        "Y": torch.stack(ys, dim=0),
        "process_id": torch.tensor(pids, dtype=torch.long),
    }


def fit_block_map(ds: dict, rows: List[dict], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    block_targets = build_block_targets(rows, seq_len=seq_len)
    # align_xy needs sequence_id/position, so rebuild minimal dataset with those keys by reusing another approach
    # Here we derive block fit by fitting directly against decoded per-token block targets from rows order.
    # For consistency with prior experiments, do full aligned path via a temporary expanded structure.
    # Build expanded aligned arrays manually.
    xs = []
    ys = []
    cursor = 0
    for seq_idx, row in enumerate(rows):
        toks = [int(t) for t in row["tokens"]]
        if len(toks) < seq_len:
            continue
        for pos in range(seq_len):
            xs.append(ds["X"][cursor + pos])
            ys.append(block_targets[(seq_idx, pos)])
        cursor += seq_len
    Xb = torch.stack(xs, dim=0).float()
    Yb = torch.stack(ys, dim=0).float()
    Wb, bb = _fit_linear_map(Xb, to_alr(Yb))
    return Wb, bb


def optimize_alr_shift_for_decrease(c_train: torch.Tensor, target_idx: int) -> torch.Tensor:
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
    return best[1].to(torch.float32)


def min_norm_delta_from_alr_shift(Wb: torch.Tensor, d_alr: torch.Tensor) -> torch.Tensor:
    sol = torch.linalg.lstsq(Wb.to(torch.float64), d_alr.to(torch.float64)).solution
    return sol.to(torch.float32)


def eval_on_rows(model, cfg, rows_subset: List[dict], Wb: torch.Tensor, bb: torch.Tensor, delta: torch.Tensor | None) -> dict:
    nll_sum = 0.0
    tok_count = 0
    p_base_chunks = []
    p_int_chunks = []
    for start in range(0, len(rows_subset), BATCH):
        chunk = rows_subset[start : start + BATCH]
        x = torch.tensor([r["tokens"][: cfg.seq_len] for r in chunk], dtype=torch.long)
        y = torch.tensor([r["tokens"][1 : cfg.seq_len + 1] for r in chunk], dtype=torch.long)
        with torch.no_grad():
            logits, cache = model(x, capture_residuals=True)
            res = cache["final_ln"]
            if delta is None:
                logits_i = logits
                res_i = res
            else:
                res_i = res + delta.view(1, 1, -1)
                logits_i = model.unembed(res_i)
        nll_sum += F.cross_entropy(logits_i.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum").item()
        tok_count += int(y.numel())
        r0 = res.reshape(-1, res.shape[-1]).float()
        r1 = res_i.reshape(-1, res_i.shape[-1]).float()
        p0 = from_alr(r0 @ Wb.T + bb)
        p1 = from_alr(r1 @ Wb.T + bb)
        p_base_chunks.append(p0.cpu())
        p_int_chunks.append(p1.cpu())
    p_base = torch.cat(p_base_chunks, dim=0)
    p_int = torch.cat(p_int_chunks, dim=0)
    delta_prob = p_int.mean(dim=0) - p_base.mean(dim=0)
    return {
        "nll_per_token": float(nll_sum / max(1, tok_count)),
        "ppl": float(math.exp(nll_sum / max(1, tok_count))),
        "mean_prob_delta": [float(v.item()) for v in delta_prob],
    }


def main() -> None:
    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    ds = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    Wb, bb = fit_block_map(ds, rows, seq_len=cfg.seq_len)

    c_train = ds["X"].float() @ Wb.T + bb
    deltas = {}
    for j in (0, 1, 2):
        d_alr = optimize_alr_shift_for_decrease(c_train, target_idx=j)
        deltas[j] = min_norm_delta_from_alr_shift(Wb, d_alr)

    rows_by_pid = {j: [r for r in rows if int(r["process_id"]) == j and len(r["tokens"]) >= cfg.seq_len + 1] for j in (0, 1, 2)}
    baselines = {j: eval_on_rows(model, cfg, rows_by_pid[j], Wb, bb, delta=None) for j in (0, 1, 2)}

    records = []
    for j in (0, 1, 2):
        base = baselines[j]
        for s in SCALES:
            delta = None if s == 0.0 else (deltas[j] * float(s))
            r = eval_on_rows(model, cfg, rows_by_pid[j], Wb, bb, delta=delta)
            records.append(
                {
                    "process": j,
                    "steer": f"dec_p{j}",
                    "scale": s,
                    "nll_per_token": r["nll_per_token"],
                    "ppl": r["ppl"],
                    "ppl_delta_vs_base": r["ppl"] - base["ppl"],
                    "delta_p1": r["mean_prob_delta"][0],
                    "delta_p2": r["mean_prob_delta"][1],
                    "delta_p3": r["mean_prob_delta"][2],
                }
            )

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "scales": SCALES,
            "note": "matched pairs only: dec_pj evaluated on process-j dataset",
        },
        "baselines": baselines,
        "delta_norms": {f"dec_p{j}": float(torch.norm(deltas[j]).item()) for j in (0, 1, 2)},
        "records": records,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("process,steer,scale,nll_per_token,ppl,ppl_delta_vs_base,delta_p1,delta_p2,delta_p3\n")
        for r in records:
            f.write(
                f"p{r['process']},{r['steer']},{r['scale']:.2f},{r['nll_per_token']:.8f},{r['ppl']:.8f},"
                f"{r['ppl_delta_vs_base']:.8f},{r['delta_p1']:.8f},{r['delta_p2']:.8f},{r['delta_p3']:.8f}\n"
            )

    print(f"saved {OUT_JSON}")
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
