#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.simple_transformer_residual import Config, TinyTransformer, eval_loss, load_mess3_jsonl, split_dataset


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyTransformer, Config, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def eval_per_process(
    model: TinyTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    process_ids: torch.Tensor,
    *,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    unique_processes = sorted(int(p) for p in process_ids.unique().tolist())
    g = torch.Generator().manual_seed(seed)

    for pid in unique_processes:
        idx = torch.where(process_ids == pid)[0]
        if idx.numel() == 0:
            continue
        x_p = x[idx]
        y_p = y[idx]

        losses = []
        for _ in range(eval_batches):
            pick = torch.randint(0, x_p.shape[0], (batch_size,), generator=g)
            xb = x_p[pick].to(device)
            yb = y_p[pick].to(device)
            logits, _ = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), yb.reshape(-1))
            losses.append(float(loss.item()))

        out[str(pid)] = sum(losses) / len(losses)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained tiny transformer checkpoint on a JSONL dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    model, cfg, _ = load_model(Path(args.checkpoint), device)
    data = load_mess3_jsonl(Path(args.dataset), seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
    tr, va, te = split_dataset(data, train_frac=args.train_frac, seed=args.seed)

    metrics = {
        "checkpoint": str(Path(args.checkpoint)),
        "dataset": str(Path(args.dataset)),
        "train_size": int(tr.size),
        "val_size": int(va.size),
        "test_size": int(te.size),
        "val_loss": float(eval_loss(model, va, args.batch_size, args.eval_batches, device, args.seed + 1000)),
        "test_loss": float(eval_loss(model, te, args.batch_size, args.eval_batches, device, args.seed + 2000)),
    }

    # Optional per-process metrics if process_id exists in JSONL rows.
    process_ids = []
    with Path(args.dataset).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            toks = row.get("tokens", [])
            if not isinstance(toks, list) or len(toks) < cfg.seq_len + 1:
                continue
            process_ids.append(int(row.get("process_id", -1)))

    if process_ids and min(process_ids) >= 0:
        pid_t = torch.tensor(process_ids, dtype=torch.long)
        # split_dataset shuffles by permutation; rebuild same split indices deterministically.
        n = pid_t.shape[0]
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n, generator=g)
        n_train = max(1, min(n - 1, int(n * args.train_frac)))
        n_eval = n - n_train
        n_val = max(1, n_eval // 2)
        n_test = n_eval - n_val
        if n_test == 0:
            n_test = 1
            n_val = max(1, n_val - 1)
        test_idx = perm[n_train + n_val :]

        metrics["test_loss_per_process"] = eval_per_process(
            model,
            te.x,
            te.y,
            pid_t[test_idx],
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + 3000,
        )

    text = json.dumps(metrics, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote metrics to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
