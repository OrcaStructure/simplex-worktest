from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch

from .mess3 import Mess3Process


@dataclass(frozen=True)
class BaselineMetrics:
    nll: float
    bits_per_token: float
    perplexity: float
    num_tokens: int


def _to_metrics(total_nll: float, token_count: int) -> BaselineMetrics:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    nll = total_nll / token_count
    return BaselineMetrics(
        nll=nll,
        bits_per_token=nll / math.log(2.0),
        perplexity=math.exp(nll),
        num_tokens=token_count,
    )


def load_dataset_rows(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if "tokens" not in row or not isinstance(row["tokens"], list):
                raise ValueError(f"Line {line_no}: missing list field 'tokens'")
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def rows_to_sequences(rows: Sequence[dict], vocab_size: int) -> List[List[int]]:
    seqs: List[List[int]] = []
    for i, row in enumerate(rows):
        toks = [int(t) for t in row["tokens"]]
        if not toks:
            continue
        if any(t < 0 or t >= vocab_size for t in toks):
            raise ValueError(f"Row {i}: token outside [0,{vocab_size - 1}]")
        seqs.append(toks)
    if not seqs:
        raise ValueError("No valid sequences found")
    return seqs


def split_rows(rows: Sequence[dict], train_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    n = len(rows)
    if n < 2:
        raise ValueError("Need at least 2 rows")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = max(1, min(n - 1, int(n * train_frac)))
    tr = [rows[i] for i in perm[:n_train]]
    va = [rows[i] for i in perm[n_train:]]
    return tr, va


def fit_unigram(sequences: Sequence[Sequence[int]], vocab_size: int, smoothing: float = 1.0) -> torch.Tensor:
    counts = torch.full((vocab_size,), float(smoothing), dtype=torch.float64)
    for seq in sequences:
        for tok in seq:
            counts[int(tok)] += 1.0
    return counts / counts.sum()


def evaluate_unigram(sequences: Sequence[Sequence[int]], probs: torch.Tensor) -> BaselineMetrics:
    total_nll = 0.0
    n = 0
    for seq in sequences:
        for tok in seq:
            p = float(probs[int(tok)].item())
            total_nll += -math.log(p)
            n += 1
    return _to_metrics(total_nll, n)


def fit_markov_1step(
    sequences: Sequence[Sequence[int]],
    vocab_size: int,
    smoothing: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # p(x0), p(xt | xt-1)
    init_counts = torch.full((vocab_size,), float(smoothing), dtype=torch.float64)
    trans_counts = torch.full((vocab_size, vocab_size), float(smoothing), dtype=torch.float64)

    for seq in sequences:
        if not seq:
            continue
        init_counts[int(seq[0])] += 1.0
        for t in range(1, len(seq)):
            trans_counts[int(seq[t - 1]), int(seq[t])] += 1.0

    init_probs = init_counts / init_counts.sum()
    trans_probs = trans_counts / trans_counts.sum(dim=1, keepdim=True)
    return init_probs, trans_probs


def evaluate_markov_1step(
    sequences: Sequence[Sequence[int]],
    init_probs: torch.Tensor,
    trans_probs: torch.Tensor,
) -> BaselineMetrics:
    total_nll = 0.0
    n = 0
    for seq in sequences:
        if not seq:
            continue
        total_nll += -math.log(float(init_probs[int(seq[0])].item()))
        n += 1
        for t in range(1, len(seq)):
            total_nll += -math.log(float(trans_probs[int(seq[t - 1]), int(seq[t])].item()))
            n += 1
    return _to_metrics(total_nll, n)


def _extract_mess3_specs(rows: Sequence[dict]) -> List[Tuple[float, float]]:
    seen = {}
    for r in rows:
        key = (float(r["alpha"]), float(r["x"]))
        seen[key] = None
    specs = list(seen.keys())
    specs.sort()
    return specs


def evaluate_oracle_mess3_mixture(
    sequences: Sequence[Sequence[int]],
    specs: Sequence[Tuple[float, float]],
) -> BaselineMetrics:
    if len(specs) == 0:
        raise ValueError("Need at least one Mess3 spec")
    processes = [Mess3Process(alpha=a, x=x, dtype=torch.float64, device="cpu") for a, x in specs]
    m = len(processes)

    total_nll = 0.0
    n = 0
    for seq in sequences:
        process_weights = torch.full((m,), 1.0 / m, dtype=torch.float64)
        beliefs = [p.stationary_distribution().clone() for p in processes]

        for obs in seq:
            pred_tok = torch.zeros(3, dtype=torch.float64)
            evidences = torch.zeros(m, dtype=torch.float64)
            new_beliefs: List[torch.Tensor] = []

            for i, p in enumerate(processes):
                tok_probs = p.predictive_token_probs(beliefs[i])
                pred_tok += process_weights[i] * tok_probs
                new_belief, evidence = p.update_belief(beliefs[i], int(obs))
                evidences[i] = evidence
                new_beliefs.append(new_belief)

            p_obs = float(pred_tok[int(obs)].item())
            total_nll += -math.log(p_obs)
            n += 1

            # Bayesian update over process identity.
            process_weights = process_weights * evidences
            process_weights = process_weights / process_weights.sum()
            beliefs = new_beliefs

    return _to_metrics(total_nll, n)


def evaluate_baselines_from_rows(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    vocab_size: int = 3,
    smoothing: float = 1.0,
) -> Tuple[BaselineMetrics, BaselineMetrics, BaselineMetrics]:
    train_seqs = rows_to_sequences(train_rows, vocab_size=vocab_size)
    val_seqs = rows_to_sequences(val_rows, vocab_size=vocab_size)

    uni_probs = fit_unigram(train_seqs, vocab_size=vocab_size, smoothing=smoothing)
    unigram = evaluate_unigram(val_seqs, uni_probs)

    init_probs, trans_probs = fit_markov_1step(train_seqs, vocab_size=vocab_size, smoothing=smoothing)
    markov = evaluate_markov_1step(val_seqs, init_probs, trans_probs)

    specs = _extract_mess3_specs(train_rows)
    oracle = evaluate_oracle_mess3_mixture(val_seqs, specs=specs)

    return unigram, markov, oracle

