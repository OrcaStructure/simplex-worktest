from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch

from .mess3 import Mess3Process


Mess3Spec = Tuple[float, float]


def generate_equal_mixture_dataset(
    specs: Sequence[Mess3Spec],
    sequences_per_process: int,
    steps: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    include_states: bool = False,
) -> List[dict]:
    """Create a dataset with equal numbers of sequences from each Mess3 process.

    Each example has token vocabulary {0,1,2}. Output rows include:
      process_id, alpha, x, tokens
    and optionally:
      hidden_states
    """
    if len(specs) <= 0:
        raise ValueError("specs must contain at least one Mess3 parameter pair")
    if sequences_per_process <= 0:
        raise ValueError("sequences_per_process must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    rows: List[dict] = []
    base_gen = torch.Generator().manual_seed(seed)

    for process_id, (alpha, x) in enumerate(specs):
        process = Mess3Process(alpha=alpha, x=x)
        # Deterministic per-process stream derived from base generator.
        process_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=base_gen).item())
        gen = torch.Generator().manual_seed(process_seed)

        for _ in range(sequences_per_process):
            init = process.stationary_distribution()
            init_state = process.sample_initial_state(belief=init, generator=gen)
            states, tokens = process.sample_sequence(initial_state=init_state, steps=steps, generator=gen)
            row = {
                "process_id": process_id,
                "alpha": float(alpha),
                "x": float(x),
                "tokens": [int(t) for t in tokens],
            }
            if include_states:
                row["hidden_states"] = [int(s) for s in states]
            rows.append(row)

    if shuffle:
        perm = torch.randperm(len(rows), generator=base_gen).tolist()
        rows = [rows[i] for i in perm]

    return rows


def write_jsonl(rows: Iterable[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out
