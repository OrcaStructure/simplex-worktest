#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.block_simplex_regression import build_block_targets, extract_specs
from src.hmm_process.mess3 import Mess3Process
from src.residual_simplex_regression import _fit_linear_map, _project_to_simplex, _split_indices, load_rows


DATASET_PATH = Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl")
OUT_JSON = Path("artifacts/residual_simplex/prob_to_prob_kl.json")

MAX_SEQS = 6000
SEQ_LEN = 12
TRAIN_FRAC = 0.8
SEED = 0
EPS = 1e-8

NAMES = ("p0", "p1", "p2", "block")


def mean_kl(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = EPS) -> float:
    y = y_true.clamp_min(eps)
    p = y_pred.clamp_min(eps)
    return float((y * (torch.log(y) - torch.log(p))).sum(dim=1).mean().item())


def build_all_targets(rows: List[dict], seq_len: int) -> Dict[str, torch.Tensor]:
    specs = extract_specs(rows)
    procs = [Mess3Process(alpha=a, x=x, dtype=torch.float64, device="cpu") for a, x in specs]
    block_t = build_block_targets(rows, seq_len=seq_len)

    all_p0: List[torch.Tensor] = []
    all_p1: List[torch.Tensor] = []
    all_p2: List[torch.Tensor] = []
    all_bk: List[torch.Tensor] = []

    for seq_idx, row in enumerate(rows):
        tokens = [int(t) for t in row["tokens"]]
        if len(tokens) < seq_len:
            continue
        toks = tokens[:seq_len]
        trajs = [p.belief_trajectory(toks).to(torch.float32) for p in procs]
        for pos in range(seq_len):
            all_p0.append(trajs[0][pos + 1])
            all_p1.append(trajs[1][pos + 1])
            all_p2.append(trajs[2][pos + 1])
            all_bk.append(block_t[(seq_idx, pos)])

    return {
        "p0": torch.stack(all_p0, dim=0),
        "p1": torch.stack(all_p1, dim=0),
        "p2": torch.stack(all_p2, dim=0),
        "block": torch.stack(all_bk, dim=0),
    }


def fit_source_to_target_kl(src_prob: torch.Tensor, tgt_prob: torch.Tensor, split_seed: int) -> dict:
    tr, va = _split_indices(src_prob.shape[0], train_frac=TRAIN_FRAC, seed=split_seed)
    w, b = _fit_linear_map(src_prob[tr], tgt_prob[tr])  # [3,3], [3]
    pred_raw = src_prob[va] @ w.T + b
    pred_prob = _project_to_simplex(pred_raw)
    kl = mean_kl(tgt_prob[va], pred_prob)
    mse_prob = float(((pred_prob - tgt_prob[va]) ** 2).mean().item())
    return {
        "kl_val": kl,
        "prob_mse_val": mse_prob,
        "num_train": int(tr.numel()),
        "num_val": int(va.numel()),
    }


def main() -> None:
    rows = load_rows(DATASET_PATH, MAX_SEQS)
    targets = build_all_targets(rows, seq_len=SEQ_LEN)

    out = {
        "config": {
            "dataset_path": str(DATASET_PATH),
            "max_seqs": MAX_SEQS,
            "seq_len": SEQ_LEN,
            "train_frac": TRAIN_FRAC,
            "seed": SEED,
            "coordinates": "probability -> linear -> simplex-projection -> KL on probs",
        },
        "samples": int(targets["p0"].shape[0]),
        "pairs": {},
    }

    for i, src in enumerate(NAMES):
        for j, tgt in enumerate(NAMES):
            out["pairs"][f"{src}_to_{tgt}"] = fit_source_to_target_kl(
                targets[src], targets[tgt], split_seed=SEED + 100 * i + j
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"saved {OUT_JSON}")
    print(f"samples={out['samples']}")
    for src in NAMES:
        row = []
        for tgt in NAMES:
            row.append(f"{out['pairs'][f'{src}_to_{tgt}']['kl_val']:.4f}")
        print(f"{src} -> [{', '.join(row)}]  (targets p0,p1,p2,block)")


if __name__ == "__main__":
    main()

