# Sweep Runbook (GPU)

This runbook adds a reproducible way to sweep:
- number of transformer layers (`n_layers`)
- model width (`d_model`) down to small limits
- number of mixed Mess3 processes (e.g. 2, 3, 8+)

It trains on each dataset variant, then evaluates each trained checkpoint on every dataset variant (cross-eval matrix).

## 1) Edit sweep config

Use:
- `configs/model_sweep_example.json`

Key sections:
- `global`: shared training/eval settings
- `dataset_variants`: process-count variants
- `model_variants`: architecture variants

Notes:
- `d_model` must be divisible by `n_heads`.
- For low-dimensional tests, reduce `d_model` (e.g. 12) and keep `n_heads` compatible.
- Practical floor: keep `d_model >= 6` so attention heads and residual capacity are non-degenerate.
- `num_processes` can be `2` or much larger.

## 2) Run the full sweep

From repo root:

```bash
python src/sweeps/run_model_sweep.py --config configs/model_sweep_example.json
```

Outputs are written under:
- `artifacts/sweeps/<run_name>/manifest.json`
- `artifacts/sweeps/<run_name>/cross_eval_losses.csv`
- `artifacts/sweeps/<run_name>/checkpoints/*.pt`
- `artifacts/sweeps/<run_name>/logs/*.log`
- `artifacts/sweeps/<run_name>/eval/*.json`

## 3) Generate only datasets

If you want ad-hoc data generation first:

```bash
python src/hmm_process/make_mixed_mess3_dataset_sweep.py \
  --num-processes 8 \
  --spec-mode linspace \
  --alpha-min 0.70 --alpha-max 0.95 \
  --x-min 0.03 --x-max 0.25 \
  --sequences-per-process 8000 \
  --steps 64 \
  --seed 2 \
  --out artifacts/datasets/mess3_mix_p8.jsonl
```

## 4) Evaluate one checkpoint on one dataset

```bash
python src/sweeps/eval_transformer_checkpoint.py \
  --checkpoint artifacts/sweeps/<run_name>/checkpoints/<ckpt>.pt \
  --dataset artifacts/sweeps/<run_name>/datasets/<dataset>.jsonl \
  --device cuda \
  --out artifacts/sweeps/<run_name>/eval/single_eval.json
```

## 5) What to send back after GPU run

Share these files so I can analyze and produce report tables/figures:
- `artifacts/sweeps/<run_name>/manifest.json`
- `artifacts/sweeps/<run_name>/cross_eval_losses.csv`
- optionally `artifacts/sweeps/<run_name>/eval/*.json` for per-process breakdown
