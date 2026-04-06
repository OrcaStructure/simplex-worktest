from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process.mixed_dataset import generate_equal_mixture_dataset, write_jsonl


def parse_specs(specs_str: str) -> List[Tuple[float, float]]:
    # Format: "0.8,0.1;0.7,0.15;0.9,0.05"
    specs: List[Tuple[float, float]] = []
    for item in specs_str.split(";"):
        item = item.strip()
        if not item:
            continue
        a_s, x_s = [part.strip() for part in item.split(",")]
        specs.append((float(a_s), float(x_s)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one mixed dataset from three Mess3 processes")
    parser.add_argument(
        "--specs",
        type=str,
        default="0.8,0.1;0.7,0.15;0.9,0.05",
        help="Three alpha,x pairs separated by semicolons",
    )
    parser.add_argument("--sequences-per-process", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=64, help="tokens per sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--include-states", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="src/hmm_process/artifacts/mess3_mixed_dataset.jsonl",
    )
    args = parser.parse_args()

    specs = parse_specs(args.specs)
    rows = generate_equal_mixture_dataset(
        specs=specs,
        sequences_per_process=args.sequences_per_process,
        steps=args.steps,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        include_states=args.include_states,
    )
    out = write_jsonl(rows, args.out)

    total = len(rows)
    print(f"Wrote {total} sequences to {out}")
    print(f"Sequences per process: {args.sequences_per_process}")
    print(f"Tokens per sequence: {args.steps}")
    print(f"Total tokens: {total * args.steps}")


if __name__ == "__main__":
    main()

