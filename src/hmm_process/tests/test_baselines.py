import torch

from src.hmm_process import Mess3Process
from src.hmm_process.baselines import (
    evaluate_baselines_from_rows,
    fit_markov_1step,
    fit_unigram,
    evaluate_markov_1step,
    evaluate_unigram,
)


def _make_rows(n: int = 300, steps: int = 32):
    specs = [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05)]
    rows = []
    g = torch.Generator().manual_seed(7)
    for pid, (a, x) in enumerate(specs):
        p = Mess3Process(alpha=a, x=x)
        for _ in range(n):
            init = p.stationary_distribution()
            s0 = p.sample_initial_state(init, generator=g)
            _, obs = p.sample_sequence(initial_state=s0, steps=steps, generator=g)
            rows.append({"process_id": pid, "alpha": a, "x": x, "tokens": obs})
    return rows


def test_basic_models_produce_finite_metrics() -> None:
    rows = _make_rows(n=40, steps=20)
    split = int(0.8 * len(rows))
    train = rows[:split]
    val = rows[split:]

    uni, mkv, orc = evaluate_baselines_from_rows(train, val, vocab_size=3)
    assert uni.nll > 0
    assert mkv.nll > 0
    assert orc.nll > 0
    assert torch.isfinite(torch.tensor([uni.nll, mkv.nll, orc.nll])).all()


def test_markov_better_than_unigram_on_generated_data() -> None:
    rows = _make_rows(n=25, steps=24)
    split = int(0.8 * len(rows))
    train_rows, val_rows = rows[:split], rows[split:]

    train = [r["tokens"] for r in train_rows]
    val = [r["tokens"] for r in val_rows]

    up = fit_unigram(train, vocab_size=3)
    um = evaluate_unigram(val, up)

    p0, p1 = fit_markov_1step(train, vocab_size=3)
    mm = evaluate_markov_1step(val, p0, p1)

    assert mm.nll <= um.nll
