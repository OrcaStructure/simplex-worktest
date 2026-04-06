#!/usr/bin/env python3
"""Train a tiny transformer and inspect residual-stream internals.

Primary mode is training on Mess3 mixed JSONL data with rows like:
  {"process_id": 0, "alpha": 0.8, "x": 0.1, "tokens": [0,1,2,...]}

All key defaults are hard-coded in one section near the top of this file.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None


# =========================
# Hard-coded defaults
# =========================
DEFAULT_SEED = 0
DEFAULT_DEVICE = "cuda"  # GPU default
DEFAULT_USE_SYNTHETIC = False
DEFAULT_DATASET_PATH = "src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"
DEFAULT_TRAIN_FRAC = 0.9
DEFAULT_EVAL_BATCHES = 20

DEFAULT_TRAIN_STEPS = 800
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 3e-3
DEFAULT_VOCAB_SIZE = 3
DEFAULT_SEQ_LEN = 12
DEFAULT_D_MODEL = 48
DEFAULT_N_HEADS = 4
DEFAULT_D_FF = 96
DEFAULT_N_LAYERS = 2

DEFAULT_PROMPT = "0,1,2"
DEFAULT_GENERATE = 10
DEFAULT_TOP_K = 3

DEFAULT_USE_WANDB = True
DEFAULT_WANDB_PROJECT = "mess3-transformer"
DEFAULT_WANDB_ENTITY = None
DEFAULT_WANDB_MODE = "online"  # online | offline | disabled
DEFAULT_WANDB_RUN_NAME = "mess3-run"


@dataclass
class Config:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = DEFAULT_D_MODEL
    n_heads: int = DEFAULT_N_HEADS
    d_ff: int = DEFAULT_D_FF
    n_layers: int = DEFAULT_N_LAYERS
    seq_len: int = DEFAULT_SEQ_LEN
    dropout: float = 0.0


@dataclass
class TokenDataset:
    x: torch.Tensor  # [N, T]
    y: torch.Tensor  # [N, T]

    @property
    def size(self) -> int:
        return int(self.x.shape[0])


def init_wandb(args: argparse.Namespace, cfg: Config):
    if not args.use_wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("wandb is not installed; continuing without wandb logging.")
        return None

    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "seed": args.seed,
                "device": args.device,
                "dataset_path": args.dataset_path,
                "train_frac": args.train_frac,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "vocab_size": cfg.vocab_size,
                "seq_len": cfg.seq_len,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "n_layers": cfg.n_layers,
                "use_synthetic": args.use_synthetic,
            },
        )
    except Exception as exc:
        print(f"wandb init failed ({type(exc).__name__}): {exc}. Continuing without wandb.")
        return None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        capture: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cache: Dict[str, torch.Tensor] = {}

        a = self.ln1(x)
        attn_out, _ = self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        if capture:
            cache["after_attn"] = x.detach().clone()

        m = self.ln2(x)
        mlp_out = self.mlp(m)
        x = x + mlp_out
        if capture:
            cache["after_mlp"] = x.detach().clone()

        return x, cache


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        idx: torch.Tensor,
        capture_residuals: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, t = idx.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"Input length {t} exceeds configured seq_len={self.cfg.seq_len}")

        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(idx) + self.pos_emb(pos)

        cache: Dict[str, torch.Tensor] = {}
        if capture_residuals:
            cache["embed"] = x.detach().clone()

        mask = self._causal_mask(t, idx.device)
        for layer_i, block in enumerate(self.blocks):
            x, block_cache = block(x, mask, capture=capture_residuals)
            if capture_residuals:
                cache[f"layer_{layer_i}_after_attn"] = block_cache["after_attn"]
                cache[f"layer_{layer_i}_after_mlp"] = block_cache["after_mlp"]

        x = self.final_ln(x)
        logits = self.unembed(x)
        if capture_residuals:
            cache["final_ln"] = x.detach().clone()

        return logits, cache


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        start = random.randrange(vocab_size)
        step = random.randrange(1, max(2, min(7, vocab_size)))
        seq = [(start + i * step) % vocab_size for i in range(seq_len + 1)]
        x[b] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        y[b] = torch.tensor(seq[1:], dtype=torch.long, device=device)

    return x, y


def load_mess3_jsonl(path: Path, seq_len: int, vocab_size: int) -> TokenDataset:
    xs: List[List[int]] = []
    ys: List[List[int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            tokens = row.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError(f"Line {line_no}: missing list field 'tokens'")
            if len(tokens) < seq_len + 1:
                continue

            chunk = [int(t) for t in tokens[: seq_len + 1]]
            if any(t < 0 or t >= vocab_size for t in chunk):
                raise ValueError(f"Line {line_no}: token outside [0,{vocab_size - 1}]")

            xs.append(chunk[:-1])
            ys.append(chunk[1:])

    if len(xs) == 0:
        raise ValueError(f"No usable sequences in {path}. Need tokens length >= {seq_len + 1}.")

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return TokenDataset(x=x, y=y)


def split_dataset(dataset: TokenDataset, train_frac: float, seed: int) -> Tuple[TokenDataset, TokenDataset]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")

    n = dataset.size
    if n < 2:
        raise ValueError("Dataset needs at least 2 sequences for split")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, min(n - 1, int(n * train_frac)))
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:]

    return TokenDataset(dataset.x[tr_idx], dataset.y[tr_idx]), TokenDataset(dataset.x[va_idx], dataset.y[va_idx])


def sample_batch(dataset: TokenDataset, batch_size: int, device: torch.device, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, dataset.size, (batch_size,), generator=generator)
    return dataset.x[idx].to(device), dataset.y[idx].to(device)


@torch.no_grad()
def eval_loss(model: TinyTransformer, dataset: TokenDataset, batch_size: int, eval_batches: int, device: torch.device, seed: int) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses: List[float] = []
    for _ in range(eval_batches):
        x, y = sample_batch(dataset, batch_size=batch_size, device=device, generator=g)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1))
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(
    model: TinyTransformer,
    prompt: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, torch.Tensor]]:
    model.eval()
    tokens = prompt[:]
    latest_cache: Dict[str, torch.Tensor] = {}

    for _ in range(max_new_tokens):
        idx = torch.tensor(tokens[-model.cfg.seq_len :], dtype=torch.long, device=device).unsqueeze(0)
        logits, cache = model(idx, capture_residuals=True)
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        tokens.append(next_id)
        latest_cache = cache

    return tokens, latest_cache


@torch.no_grad()
def residual_report(model: TinyTransformer, cache: Dict[str, torch.Tensor], top_k: int = 5) -> str:
    if not cache:
        return "No residual cache available."

    lines: List[str] = []
    residual_keys = ["embed"]
    for i in range(model.cfg.n_layers):
        residual_keys.append(f"layer_{i}_after_attn")
        residual_keys.append(f"layer_{i}_after_mlp")

    lines.append("Residual stream norms at final token:")
    for k in residual_keys:
        if k not in cache:
            continue
        vec = cache[k][0, -1]
        lines.append(f"  {k:20s} norm={vec.norm().item():.4f}")

    lines.append("\nLogit-lens style top tokens from each residual state:")
    for k in residual_keys:
        if k not in cache:
            continue
        vec = cache[k][0, -1]
        logits = model.unembed(model.final_ln(vec))
        vals, ids = torch.topk(logits, k=min(top_k, logits.numel()))
        top = ", ".join([f"{int(i.item())}:{float(v.item()):.2f}" for v, i in zip(vals, ids)])
        lines.append(f"  {k:20s} {top}")

    return "\n".join(lines)


def parse_prompt(prompt: str, vocab_size: int) -> List[int]:
    if not prompt.strip():
        return [0, 1, 2]
    toks = [int(t.strip()) for t in prompt.split(",") if t.strip()]
    for t in toks:
        if t < 0 or t >= vocab_size:
            raise ValueError(f"Prompt token {t} must be in [0, {vocab_size - 1}]")
    return toks


def train(
    model: TinyTransformer,
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    train_data: TokenDataset | None,
    val_data: TokenDataset | None,
    eval_batches: int,
    seed: int,
    use_synthetic: bool,
    wandb_run,
) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    for step in range(1, train_steps + 1):
        if use_synthetic:
            x, y = make_synthetic_batch(batch_size, model.cfg.seq_len, model.cfg.vocab_size, device)
        else:
            assert train_data is not None
            x, y = sample_batch(train_data, batch_size=batch_size, device=device, generator=g)

        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % max(1, train_steps // 10) == 0 or step == 1:
            msg = f"step={step:5d} train_loss={loss.item():.4f}"
            log_payload = {"step": step, "train/loss": float(loss.item())}
            if (not use_synthetic) and val_data is not None:
                val = eval_loss(
                    model,
                    val_data,
                    batch_size=batch_size,
                    eval_batches=eval_batches,
                    device=device,
                    seed=seed + step,
                )
                msg += f" val_loss={val:.4f}"
                log_payload["val/loss"] = float(val)
            print(msg)
            if wandb_run is not None:
                wandb_run.log(log_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny transformer with residual-stream analysis")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cpu", "cuda", "mps"])

    parser.add_argument("--use-synthetic", action="store_true", default=DEFAULT_USE_SYNTHETIC, help="Use synthetic arithmetic data instead of Mess3 JSONL")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to mixed Mess3 JSONL dataset",
    )
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)

    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--generate", type=int, default=DEFAULT_GENERATE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=DEFAULT_USE_WANDB)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-mode", type=str, default=DEFAULT_WANDB_MODE, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name", type=str, default=DEFAULT_WANDB_RUN_NAME)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    cfg = Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    )

    wandb_run = init_wandb(args, cfg)

    train_data: TokenDataset | None = None
    val_data: TokenDataset | None = None
    if not args.use_synthetic:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        all_data = load_mess3_jsonl(dataset_path, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        train_data, val_data = split_dataset(all_data, train_frac=args.train_frac, seed=args.seed)
        print(
            f"Loaded dataset: total={all_data.size} train={train_data.size} val={val_data.size} "
            f"seq_len={cfg.seq_len} vocab={cfg.vocab_size}"
        )

    model = TinyTransformer(cfg).to(device)
    print("Training model...")
    train(
        model,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        train_data=train_data,
        val_data=val_data,
        eval_batches=args.eval_batches,
        seed=args.seed,
        use_synthetic=args.use_synthetic,
        wandb_run=wandb_run,
    )

    prompt = parse_prompt(args.prompt, cfg.vocab_size)
    print(f"\nPrompt tokens: {prompt}")
    generated, cache = generate(model, prompt, args.generate, device)
    print(f"Generated tokens: {generated}")

    report = residual_report(model, cache, top_k=args.top_k)
    print("\n=== Residual Analysis ===")
    print(report)

    if wandb_run is not None:
        wandb_run.log({"inference/generated_len": len(generated)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
