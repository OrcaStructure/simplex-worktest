#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process.mixed_dataset import generate_equal_mixture_dataset, write_jsonl


def parse_specs_json(path: Path) -> List[Tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("specs json must be a list of [alpha, x] pairs")
    out = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("every specs entry must be [alpha, x]")
        out.append((float(item[0]), float(item[1])))
    if not out:
        raise ValueError("spec list is empty")
    return out


def make_linspace_specs(num_processes: int, alpha_min: float, alpha_max: float, x_min: float, x_max: float) -> List[Tuple[float, float]]:
    if num_processes == 1:
        return [((alpha_min + alpha_max) * 0.5, (x_min + x_max) * 0.5)]
    specs = []
    for i in range(num_processes):
        t = i / (num_processes - 1)
        tt = math.sqrt(t)
        specs.append((alpha_min + (alpha_max - alpha_min) * tt, x_min + (x_max - x_min) * t))
    return specs


def make_random_specs(num_processes: int, alpha_min: float, alpha_max: float, x_min: float, x_max: float, seed: int) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(num_processes):
        out.append((rng.uniform(alpha_min, alpha_max), rng.uniform(x_min, x_max)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed Mess3 datasets with arbitrary process count")
    parser.add_argument("--num-processes", type=int, default=3)
    parser.add_argument("--specs-json", type=str, default="", help="Optional JSON file containing [[alpha,x], ...]")
    parser.add_argument("--spec-mode", type=str, default="linspace", choices=["linspace", "random"])
    parser.add_argument("--alpha-min", type=float, default=0.7)
    parser.add_argument("--alpha-max", type=float, default=0.95)
    parser.add_argument("--x-min", type=float, default=0.03)
    parser.add_argument("--x-max", type=float, default=0.25)
    parser.add_argument("--sequences-per-process", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--include-states", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if args.num_processes <= 0:
        raise ValueError("--num-processes must be positive")

    if args.specs_json:
        specs = parse_specs_json(Path(args.specs_json))
    else:
        if args.spec_mode == "linspace":
            specs = make_linspace_specs(args.num_processes, args.alpha_min, args.alpha_max, args.x_min, args.x_max)
        else:
            specs = make_random_specs(args.num_processes, args.alpha_min, args.alpha_max, args.x_min, args.x_max, args.seed)

    rows = generate_equal_mixture_dataset(
        specs=specs,
        sequences_per_process=args.sequences_per_process,
        steps=args.steps,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        include_states=args.include_states,
    )
    out = write_jsonl(rows, args.out)

    summary = {
        "out": str(out),
        "rows": len(rows),
        "num_processes": len(specs),
        "specs": [[float(a), float(x)] for a, x in specs],
        "sequences_per_process": int(args.sequences_per_process),
        "steps": int(args.steps),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
