#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process.mixed_dataset import generate_equal_mixture_dataset, write_jsonl


@dataclass
class DatasetVariant:
    name: str
    specs: List[List[float]]
    sequences_per_process: int
    steps: int
    seed: int
    shuffle: bool


@dataclass
class ModelVariant:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    d_ff_multiplier: float


def ensure_int(value: Any, name: str, *, min_value: int = 1) -> int:
    out = int(value)
    if out < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {out}")
    return out


def parse_dataset_variant(v: Dict[str, Any]) -> DatasetVariant:
    name = str(v["name"])
    sequences_per_process = ensure_int(v.get("sequences_per_process", 5000), "sequences_per_process")
    steps = ensure_int(v.get("steps", 64), "steps")
    seed = int(v.get("seed", 0))
    shuffle = bool(v.get("shuffle", True))

    specs = v.get("specs")
    if specs is None:
        num_processes = ensure_int(v.get("num_processes", 3), "num_processes")
        alpha_range = v.get("alpha_range", [0.65, 0.95])
        x_range = v.get("x_range", [0.03, 0.25])
        a0, a1 = float(alpha_range[0]), float(alpha_range[1])
        x0, x1 = float(x_range[0]), float(x_range[1])
        if num_processes == 1:
            specs = [[0.5 * (a0 + a1), 0.5 * (x0 + x1)]]
        else:
            specs = []
            for i in range(num_processes):
                t = i / (num_processes - 1)
                # Use a slight nonlinearity to avoid near-duplicate processes at high counts.
                tt = math.sqrt(t)
                specs.append([a0 + (a1 - a0) * tt, x0 + (x1 - x0) * t])

    parsed_specs: List[List[float]] = []
    for item in specs:
        if len(item) != 2:
            raise ValueError(f"dataset {name}: each spec must be [alpha, x]")
        alpha = float(item[0])
        x = float(item[1])
        parsed_specs.append([alpha, x])

    return DatasetVariant(
        name=name,
        specs=parsed_specs,
        sequences_per_process=sequences_per_process,
        steps=steps,
        seed=seed,
        shuffle=shuffle,
    )


def parse_model_variant(v: Dict[str, Any]) -> ModelVariant:
    name = str(v["name"])
    n_layers = ensure_int(v.get("n_layers", 2), "n_layers")
    d_model = ensure_int(v.get("d_model", 48), "d_model")
    n_heads = ensure_int(v.get("n_heads", 4), "n_heads")
    d_ff_multiplier = float(v.get("d_ff_multiplier", 2.0))

    if d_model % n_heads != 0:
        raise ValueError(f"model {name}: d_model={d_model} must be divisible by n_heads={n_heads}")
    if d_ff_multiplier < 1.0:
        raise ValueError(f"model {name}: d_ff_multiplier must be >= 1.0")

    return ModelVariant(
        name=name,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff_multiplier=d_ff_multiplier,
    )


def run_cmd(cmd: List[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model sweeps over dataset variants and architecture variants")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep JSON config")
    parser.add_argument("--out-root", type=str, default="artifacts/sweeps")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    global_cfg = cfg.get("global", {})
    run_name = str(global_cfg.get("run_name", datetime.now().strftime("sweep_%Y%m%d_%H%M%S")))
    out_root = Path(args.out_root) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_variants = [parse_dataset_variant(v) for v in cfg["dataset_variants"]]
    model_variants = [parse_model_variant(v) for v in cfg["model_variants"]]

    device = str(global_cfg.get("device", "cuda"))
    train_steps = ensure_int(global_cfg.get("train_steps", 2400), "train_steps")
    batch_size = ensure_int(global_cfg.get("batch_size", 32), "batch_size")
    lr = float(global_cfg.get("lr", 3e-4))
    eval_batches = ensure_int(global_cfg.get("eval_batches", 30), "eval_batches")
    train_frac = float(global_cfg.get("train_frac", 0.9))
    seq_len = ensure_int(global_cfg.get("seq_len", 12), "seq_len")
    vocab_size = ensure_int(global_cfg.get("vocab_size", 3), "vocab_size")
    base_seed = int(global_cfg.get("seed", 0))
    use_wandb = bool(global_cfg.get("use_wandb", False))
    wandb_mode = str(global_cfg.get("wandb_mode", "disabled"))

    datasets_dir = out_root / "datasets"
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    eval_dir = out_root / "eval"

    dataset_manifest = []
    for ds in dataset_variants:
        ds_path = datasets_dir / f"{ds.name}.jsonl"
        rows = generate_equal_mixture_dataset(
            specs=[(float(a), float(x)) for a, x in ds.specs],
            sequences_per_process=ds.sequences_per_process,
            steps=ds.steps,
            seed=ds.seed,
            shuffle=ds.shuffle,
            include_states=False,
        )
        write_jsonl(rows, ds_path)
        dataset_manifest.append(
            {
                "name": ds.name,
                "path": str(ds_path),
                "num_processes": len(ds.specs),
                "specs": ds.specs,
                "rows": len(rows),
                "steps": ds.steps,
            }
        )

    train_runs: List[Dict[str, Any]] = []
    for dsi, ds in enumerate(dataset_variants):
        for mi, mv in enumerate(model_variants):
            run_id = f"train_{ds.name}__{mv.name}"
            checkpoint_path = ckpt_dir / f"{run_id}.pt"
            run_seed = base_seed + 1000 * dsi + 10 * mi
            d_ff = int(round(mv.d_model * mv.d_ff_multiplier))

            train_meta = {
                "run_id": run_id,
                "dataset_name": ds.name,
                "model_name": mv.name,
                "dataset_path": str(datasets_dir / f"{ds.name}.jsonl"),
                "checkpoint_path": str(checkpoint_path),
                "seed": run_seed,
                "n_layers": mv.n_layers,
                "d_model": mv.d_model,
                "n_heads": mv.n_heads,
                "d_ff": d_ff,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
            }
            train_runs.append(train_meta)

            if args.skip_training:
                continue

            cmd = [
                args.python,
                "src/simple_transformer_residual.py",
                "--device",
                device,
                "--dataset-path",
                train_meta["dataset_path"],
                "--train-steps",
                str(train_steps),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--eval-batches",
                str(eval_batches),
                "--train-frac",
                str(train_frac),
                "--seq-len",
                str(seq_len),
                "--vocab-size",
                str(vocab_size),
                "--d-model",
                str(mv.d_model),
                "--n-heads",
                str(mv.n_heads),
                "--d-ff",
                str(d_ff),
                "--n-layers",
                str(mv.n_layers),
                "--seed",
                str(run_seed),
                "--save-model-path",
                str(checkpoint_path),
                "--wandb-mode",
                wandb_mode,
            ]
            cmd.append("--use-wandb" if use_wandb else "--no-wandb")
            run_cmd(cmd, log_path=logs_dir / f"{run_id}.log")

    # Evaluate each trained checkpoint on each dataset variant.
    eval_rows: List[Dict[str, Any]] = []
    if not args.skip_eval:
        for run in train_runs:
            for ds in dataset_variants:
                eval_name = f"{run['run_id']}__on__{ds.name}"
                eval_out = eval_dir / f"{eval_name}.json"
                cmd = [
                    args.python,
                    "src/sweeps/eval_transformer_checkpoint.py",
                    "--checkpoint",
                    run["checkpoint_path"],
                    "--dataset",
                    str(datasets_dir / f"{ds.name}.jsonl"),
                    "--device",
                    device,
                    "--train-frac",
                    str(train_frac),
                    "--eval-batches",
                    str(eval_batches),
                    "--batch-size",
                    str(batch_size),
                    "--seed",
                    str(run["seed"]),
                    "--out",
                    str(eval_out),
                ]
                run_cmd(cmd, log_path=logs_dir / f"{eval_name}.log")
                payload = json.loads(eval_out.read_text(encoding="utf-8"))
                eval_rows.append(
                    {
                        "train_dataset": run["dataset_name"],
                        "model_name": run["model_name"],
                        "eval_dataset": ds.name,
                        "val_loss": float(payload["val_loss"]),
                        "test_loss": float(payload["test_loss"]),
                    }
                )

    manifest = {
        "config_path": str(cfg_path),
        "run_name": run_name,
        "global": global_cfg,
        "datasets": dataset_manifest,
        "train_runs": train_runs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if eval_rows:
        csv_path = out_root / "cross_eval_losses.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(eval_rows)

    print(f"Sweep artifacts: {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    if eval_rows:
        print(f"Cross-eval CSV: {out_root / 'cross_eval_losses.csv'}")


if __name__ == "__main__":
    main()
