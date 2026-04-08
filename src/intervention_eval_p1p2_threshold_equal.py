#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
OUT_BASE = Path("artifacts/residual_simplex/intervention_eval_threshold_equal")

MAX_SEQS = 6000
SEQ_SEED = 0
EPS = 1e-8
BATCH = 128


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


def optimize_direction(
    W_target: torch.Tensor,
    W_others: torch.Tensor,
    steps: int = 1200,
    lr: float = 0.08,
    lambda_gap: float = 2.0,
    beta_other: float = 0.2,
) -> torch.Tensor:
    d = W_target.shape[1]
    u = torch.randn(d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)

    best_obj = -1e18
    best_u = None
    for _ in range(steps):
        u_hat = u / (u.norm() + 1e-12)
        c = W_target.to(torch.float64) @ u_hat  # [2]
        other_act = W_others.to(torch.float64) @ u_hat  # [k]
        obj = torch.minimum(c[0], c[1]) - lambda_gap * (c[0] - c[1]).pow(2) + beta_other * other_act.pow(2).mean()
        loss = -obj
        opt.zero_grad()
        loss.backward()
        opt.step()

        obj_val = float(obj.detach().item())
        if obj_val > best_obj:
            best_obj = obj_val
            best_u = u_hat.detach().clone()

    if best_u is None:
        raise RuntimeError("Failed to optimize steering direction")
    return best_u.to(torch.float32)


def nullspace_basis(A: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    _, s, vh = torch.linalg.svd(A.to(torch.float64), full_matrices=True)
    r = int((s > tol).sum().item())
    return vh[r:].T.contiguous()


def scale_to_threshold(W_target: torch.Tensor, u: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, dict]:
    c = (W_target @ u).to(torch.float64)
    # Allow sign flip if both coordinates are negative.
    if c[0] < 0 and c[1] < 0:
        u = -u
        c = -c
    c_min = float(torch.minimum(c[0], c[1]).item())
    if c_min <= 0.0:
        raise RuntimeError(f"Cannot scale to positive threshold; base shift coords are {c.tolist()}")
    alpha = float(threshold / c_min)
    delta = (alpha * u).to(torch.float32)
    c_scaled = (W_target @ delta).to(torch.float64)
    info = {
        "alpha": alpha,
        "base_coords": [float(c[0].item()), float(c[1].item())],
        "scaled_coords": [float(c_scaled[0].item()), float(c_scaled[1].item())],
        "scaled_gap_abs": float(abs(c_scaled[0].item() - c_scaled[1].item())),
    }
    return delta, info


def forward_with_optional_steering(model, x: torch.Tensor, delta_l0: torch.Tensor | None, delta_final: torch.Tensor | None):
    bsz, t = x.shape
    pos = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, -1)
    h = model.token_emb(x) + model.pos_emb(pos)
    mask = model._causal_mask(t, x.device)

    layer0_after = None
    for layer_i, block in enumerate(model.blocks):
        a = block.ln1(h)
        attn_out, _ = block.attn(a, a, a, attn_mask=mask, need_weights=False)
        h = h + attn_out
        m = block.ln2(h)
        mlp_out = block.mlp(m)
        h = h + mlp_out
        if layer_i == 0:
            if delta_l0 is not None:
                h = h + delta_l0.view(1, 1, -1)
            layer0_after = h

    final_res = model.final_ln(h)
    if delta_final is not None:
        final_res = final_res + delta_final.view(1, 1, -1)
    logits = model.unembed(final_res)
    return logits, {"layer_0_after_mlp": layer0_after, "final_ln": final_res}


def summarize_shift(chunks: List[torch.Tensor], threshold: float) -> dict:
    d = torch.cat(chunks, dim=0)
    c1 = float(d[:, 0].mean().item())
    c2 = float(d[:, 1].mean().item())
    both_ge = float(((d[:, 0] >= threshold) & (d[:, 1] >= threshold)).float().mean().item())
    return {
        "delta_c1_mean": c1,
        "delta_c2_mean": c2,
        "mean_shift": 0.5 * (c1 + c2),
        "coord_gap_abs": abs(c1 - c2),
        "frac_tokens_both_ge_threshold": both_ge,
    }


def eval_policy(
    model,
    cfg,
    rows_subset: List[dict],
    maps_final: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    delta_l0: torch.Tensor | None,
    delta_final: torch.Tensor | None,
    threshold: float,
    p_a: int,
    p_b: int,
) -> dict:
    device = torch.device("cpu")
    nll_sum = 0.0
    tok_count = 0
    kl_b2i_sum = 0.0
    kl_i2b_sum = 0.0
    n_items = 0

    Wb, bb = maps_final["block"]
    W1, b1 = maps_final["p1"]
    W2, b2 = maps_final["p2"]

    block_shift = []
    p1_shift = []
    p2_shift = []
    pair_mass_base_all = []
    pair_mass_int_all = []
    pair_logodds_base_all = []
    pair_logodds_int_all = []

    for start in range(0, len(rows_subset), BATCH):
        chunk = rows_subset[start : start + BATCH]
        x = torch.tensor([r["tokens"][: cfg.seq_len] for r in chunk], dtype=torch.long, device=device)
        y = torch.tensor([r["tokens"][1 : cfg.seq_len + 1] for r in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            base_logits, base_cache = forward_with_optional_steering(model, x, delta_l0=None, delta_final=None)
            int_logits, int_cache = forward_with_optional_steering(model, x, delta_l0=delta_l0, delta_final=delta_final)
            base_probs = torch.softmax(base_logits, dim=-1)
            int_probs = torch.softmax(int_logits, dim=-1)

        nll = F.cross_entropy(int_logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum").item()
        nll_sum += nll
        tok_count += int(y.numel())

        pb = base_probs.clamp_min(EPS)
        pi = int_probs.clamp_min(EPS)
        kl_b2i = (pb * (torch.log(pb) - torch.log(pi))).sum(dim=-1).mean().item()
        kl_i2b = (pi * (torch.log(pi) - torch.log(pb))).sum(dim=-1).mean().item()
        kl_b2i_sum += kl_b2i * x.shape[0]
        kl_i2b_sum += kl_i2b * x.shape[0]
        n_items += x.shape[0]

        r0 = base_cache["final_ln"].reshape(-1, base_cache["final_ln"].shape[-1]).float()
        r1 = int_cache["final_ln"].reshape(-1, int_cache["final_ln"].shape[-1]).float()
        c0 = r0 @ Wb.T + bb
        c1 = r1 @ Wb.T + bb
        db = c1 - c0
        d1 = (r1 @ W1.T + b1) - (r0 @ W1.T + b1)
        d2 = (r1 @ W2.T + b2) - (r0 @ W2.T + b2)
        block_shift.append(db)
        p1_shift.append(d1)
        p2_shift.append(d2)

        p0 = from_alr(c0)
        p1p = from_alr(c1)
        outside = ({0, 1, 2} - {p_a, p_b}).pop()
        pair_mass_base = p0[:, p_a] + p0[:, p_b]
        pair_mass_int = p1p[:, p_a] + p1p[:, p_b]
        outside_base = p0[:, outside]
        outside_int = p1p[:, outside]
        pair_logodds_base = torch.log((pair_mass_base + EPS) / (outside_base + EPS))
        pair_logodds_int = torch.log((pair_mass_int + EPS) / (outside_int + EPS))
        pair_mass_base_all.append(pair_mass_base.detach().cpu())
        pair_mass_int_all.append(pair_mass_int.detach().cpu())
        pair_logodds_base_all.append(pair_logodds_base.detach().cpu())
        pair_logodds_int_all.append(pair_logodds_int.detach().cpu())

    pair_mass_base = torch.cat(pair_mass_base_all)
    pair_mass_int = torch.cat(pair_mass_int_all)
    pair_logodds_base = torch.cat(pair_logodds_base_all)
    pair_logodds_int = torch.cat(pair_logodds_int_all)
    out = {
        "nll_per_token": float(nll_sum / max(1, tok_count)),
        "ppl": float(math.exp(nll_sum / max(1, tok_count))),
        "block_shift": summarize_shift(block_shift, threshold=threshold),
        "p1_shift": summarize_shift(p1_shift, threshold=threshold),
        "p2_shift": summarize_shift(p2_shift, threshold=threshold),
        "token_dist_kl_base_to_int": float(kl_b2i_sum / max(1, n_items)),
        "token_dist_kl_int_to_base": float(kl_i2b_sum / max(1, n_items)),
        "pair_belief": {
            "pair_mass_base_mean": float(pair_mass_base.mean().item()),
            "pair_mass_int_mean": float(pair_mass_int.mean().item()),
            "pair_mass_delta": float((pair_mass_int - pair_mass_base).mean().item()),
            "pair_logodds_base_mean": float(pair_logodds_base.mean().item()),
            "pair_logodds_int_mean": float(pair_logodds_int.mean().item()),
            "pair_logodds_delta": float((pair_logodds_int - pair_logodds_base).mean().item()),
        },
    }
    return out


def _random_unit(d: int) -> torch.Tensor:
    v = torch.randn(d, dtype=torch.float32)
    return v / (v.norm() + 1e-12)


def random_vector_in_subspace(basis: torch.Tensor, target_norm: float) -> torch.Tensor:
    # basis: [d, k], columns span the subspace.
    if basis.shape[1] == 0:
        raise RuntimeError("Requested random nullspace vector, but nullspace has dimension 0")
    z = torch.randn(basis.shape[1], dtype=torch.float64)
    v = (basis.to(torch.float64) @ z).to(torch.float32)
    v = v / (v.norm() + 1e-12)
    return v * float(target_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="1,2", help="process pair like 1,2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lambda-gap", type=float, default=2.0)
    parser.add_argument("--beta-other", type=float, default=0.2)
    args = parser.parse_args()

    p_a, p_b = (int(v) for v in args.pair.split(",", 1))
    threshold = float(args.threshold)

    model, cfg, _ = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    rows = load_rows(DATASET_PATH, MAX_SEQS)

    # Fit maps for final and first-layer residual streams.
    ds_final = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="final_ln")
    ds_l0 = build_residual_dataset_from_cache_key(model, cfg, rows, cache_key="layer_0_after_mlp")
    maps_final = fit_maps(ds_final, rows, seq_len=cfg.seq_len)
    maps_l0 = fit_maps(ds_l0, rows, seq_len=cfg.seq_len)

    # Optimize direction with "not in nullspace of other maps" pressure.
    Wb_f, _ = maps_final["block"]
    A_other_f = torch.cat([maps_final["p0"][0], maps_final["p1"][0], maps_final["p2"][0]], dim=0)
    u_f = optimize_direction(
        W_target=Wb_f.float(),
        W_others=A_other_f.float(),
        steps=args.steps,
        lr=args.lr,
        lambda_gap=args.lambda_gap,
        beta_other=args.beta_other,
    )
    delta_f, info_f = scale_to_threshold(Wb_f.float(), u_f, threshold=threshold)

    Wb_l0, _ = maps_l0["block"]
    A_other_l0 = torch.cat([maps_l0["p0"][0], maps_l0["p1"][0], maps_l0["p2"][0]], dim=0)
    u_l0 = optimize_direction(
        W_target=Wb_l0.float(),
        W_others=A_other_l0.float(),
        steps=args.steps,
        lr=args.lr,
        lambda_gap=args.lambda_gap,
        beta_other=args.beta_other,
    )
    delta_l0, info_l0 = scale_to_threshold(Wb_l0.float(), u_l0, threshold=threshold)

    subset = make_balanced_pair(rows, seq_len=cfg.seq_len, seed=SEQ_SEED, p_a=p_a, p_b=p_b)
    baseline = eval_policy(
        model, cfg, subset, maps_final, delta_l0=None, delta_final=None, threshold=threshold, p_a=p_a, p_b=p_b
    )
    final_only = eval_policy(
        model, cfg, subset, maps_final, delta_l0=None, delta_final=delta_f, threshold=threshold, p_a=p_a, p_b=p_b
    )
    both_layers = eval_policy(
        model, cfg, subset, maps_final, delta_l0=delta_l0, delta_final=delta_f, threshold=threshold, p_a=p_a, p_b=p_b
    )

    # Random nullspace controls with matched L2 norms.
    N_f = nullspace_basis(A_other_f.float())
    N_l0 = nullspace_basis(A_other_l0.float())
    delta_f_rand_null = random_vector_in_subspace(N_f, target_norm=float(torch.norm(delta_f).item()))
    delta_l0_rand_null = random_vector_in_subspace(N_l0, target_norm=float(torch.norm(delta_l0).item()))

    control_final_only = eval_policy(
        model,
        cfg,
        subset,
        maps_final,
        delta_l0=None,
        delta_final=delta_f_rand_null,
        threshold=threshold,
        p_a=p_a,
        p_b=p_b,
    )
    control_both_layers = eval_policy(
        model,
        cfg,
        subset,
        maps_final,
        delta_l0=delta_l0_rand_null,
        delta_final=delta_f_rand_null,
        threshold=threshold,
        p_a=p_a,
        p_b=p_b,
    )

    tag = f"_{p_a}{p_b}_thr{threshold:g}"
    out_json = OUT_BASE.with_name(OUT_BASE.name + tag + ".json")
    out_csv = OUT_BASE.with_name(OUT_BASE.name + tag + ".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "checkpoint": str(CHECKPOINT),
            "subset": f"process_id in {{{p_a},{p_b}}}, balanced 50/50",
            "subset_size_sequences": int(len(subset)),
            "threshold": threshold,
            "objective": "maximize min(delta,epsilon) - lambda*(delta-epsilon)^2 + beta*||A_other u||^2 with ||u||=1",
            "optimizer": {
                "steps": int(args.steps),
                "lr": float(args.lr),
                "lambda_gap": float(args.lambda_gap),
                "beta_other": float(args.beta_other),
            },
        },
        "steering": {
            "final_ln": {
                "l2": float(torch.norm(delta_f).item()),
                **info_f,
            },
            "layer_0_after_mlp": {
                "l2": float(torch.norm(delta_l0).item()),
                **info_l0,
            },
        },
        "baseline": baseline,
        "final_only": final_only,
        "both_layers": both_layers,
        "nullspace_random_controls_same_norm": {
            "final_only": {
                "final_norm_l2": float(torch.norm(delta_f_rand_null).item()),
                "metrics": control_final_only,
            },
            "both_layers": {
                "layer0_norm_l2": float(torch.norm(delta_l0_rand_null).item()),
                "final_norm_l2": float(torch.norm(delta_f_rand_null).item()),
                "metrics": control_both_layers,
            },
        },
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("mode,nll_per_token,ppl,block_mean_shift,block_gap_abs,block_frac_ge_thr,p1_mean_shift,p2_mean_shift,pair_mass_delta,pair_logodds_delta,token_kl_base_to_int,token_kl_int_to_base\n")
        for mode, r in (
            ("baseline", baseline),
            ("final_only", final_only),
            ("both_layers", both_layers),
            ("control_final_only", control_final_only),
            ("control_both_layers", control_both_layers),
        ):
            f.write(
                f"{mode},{r['nll_per_token']:.8f},{r['ppl']:.8f},"
                f"{r['block_shift']['mean_shift']:.8f},{r['block_shift']['coord_gap_abs']:.8f},{r['block_shift']['frac_tokens_both_ge_threshold']:.8f},"
                f"{r['p1_shift']['mean_shift']:.8f},{r['p2_shift']['mean_shift']:.8f},"
                f"{r['pair_belief']['pair_mass_delta']:.8f},{r['pair_belief']['pair_logodds_delta']:.8f},"
                f"{r['token_dist_kl_base_to_int']:.8f},{r['token_dist_kl_int_to_base']:.8f}\n"
            )

    print(f"saved {out_json}")
    print(f"saved {out_csv}")
    print(f"\nsubset N={len(subset)} threshold={threshold}")
    for mode, r in (("baseline", baseline), ("final_only", final_only), ("both_layers", both_layers)):
        print(
            f"{mode}: nll={r['nll_per_token']:.6f} ppl={r['ppl']:.6f} "
            f"block_mean={r['block_shift']['mean_shift']:.4f} "
            f"gap={r['block_shift']['coord_gap_abs']:.4f} "
            f"frac>=thr={r['block_shift']['frac_tokens_both_ge_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
