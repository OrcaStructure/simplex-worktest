from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process.baselines import (
    evaluate_baselines_from_rows,
    load_dataset_rows,
    split_rows,
)


def _fmt(name: str, nll: float, bpt: float, ppl: float) -> str:
    return f"{name:18s} nll={nll:.6f}  bits/token={bpt:.6f}  perplexity={ppl:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unigram, 1-step Markov, and oracle Mess3 baselines")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="src/hmm_process/artifacts/mess3_mixed_dataset.jsonl",
    )
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_dataset_rows(args.dataset_path)
    train_rows, val_rows = split_rows(rows, train_frac=args.train_frac, seed=args.seed)

    unigram, markov, oracle = evaluate_baselines_from_rows(
        train_rows=train_rows,
        val_rows=val_rows,
        vocab_size=args.vocab_size,
        smoothing=args.smoothing,
    )

    print(f"Rows: total={len(rows)} train={len(train_rows)} val={len(val_rows)}")
    print(_fmt("unigram", unigram.nll, unigram.bits_per_token, unigram.perplexity))
    print(_fmt("markov_1step", markov.nll, markov.bits_per_token, markov.perplexity))
    print(_fmt("oracle_mess3_mix", oracle.nll, oracle.bits_per_token, oracle.perplexity))

    # Sanity checks (soft checks; printed rather than hard failing).
    print("\nSanity checks:")
    print(f"  markov <= unigram: {markov.nll <= unigram.nll}")
    print(f"  oracle <= markov: {oracle.nll <= markov.nll}")
    print(f"  oracle <= unigram: {oracle.nll <= unigram.nll}")


if __name__ == "__main__":
    main()
