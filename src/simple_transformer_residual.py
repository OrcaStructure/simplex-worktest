#!/usr/bin/env python3
"""Train a tiny transformer and inspect residual-stream internals.

This script:
1) Builds synthetic token sequences.
2) Trains a small causal transformer for next-token prediction.
3) Runs generation/inference.
4) Retroactively analyzes residual stream states captured during inference.

Usage examples:
  python src/simple_transformer_residual.py --train-steps 600 --device cpu
  python src/simple_transformer_residual.py --prompt "1,3,5" --generate 4
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 3
    seq_len: int = 12
    dropout: float = 0.0


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
        # True entries are blocked positions for nn.MultiheadAttention.
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


def make_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create arithmetic-progression-like token sequences mod vocab."""
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for b in range(batch_size):
        start = random.randrange(vocab_size)
        step = random.randrange(1, min(7, vocab_size))
        seq = [(start + i * step) % vocab_size for i in range(seq_len + 1)]
        x[b] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        y[b] = torch.tensor(seq[1:], dtype=torch.long, device=device)

    return x, y


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
def residual_report(
    model: TinyTransformer,
    cache: Dict[str, torch.Tensor],
    top_k: int = 5,
) -> str:
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
        vec = cache[k][0, -1]  # [d_model]
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


def train(model: TinyTransformer, steps: int, batch_size: int, lr: float, device: torch.device) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        x, y = make_batch(batch_size, model.cfg.seq_len, model.cfg.vocab_size, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % max(1, steps // 10) == 0 or step == 1:
            print(f"step={step:4d} loss={loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny transformer with residual-stream analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="1,3,5")
    parser.add_argument("--generate", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type != "cpu" and not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("CUDA requested but not available")

    cfg = Config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    )

    model = TinyTransformer(cfg).to(device)
    print("Training model...")
    train(model, args.train_steps, args.batch_size, args.lr, device)

    prompt = parse_prompt(args.prompt, cfg.vocab_size)
    print(f"\\nPrompt tokens: {prompt}")
    generated, cache = generate(model, prompt, args.generate, device)
    print(f"Generated tokens: {generated}")

    report = residual_report(model, cache, top_k=args.top_k)
    print("\\n=== Residual Analysis ===")
    print(report)


if __name__ == "__main__":
    main()
