from pathlib import Path

import torch

from src.block_simplex_regression import block_posteriors_for_tokens, build_block_targets, extract_specs
from src.hmm_process import Mess3Process
from src.residual_simplex_regression import load_rows


def test_extract_specs_respects_process_id_order() -> None:
    rows = load_rows(Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"), 300)
    specs = extract_specs(rows)
    assert specs == [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05)]


def test_block_targets_are_valid_simplex_points() -> None:
    rows = load_rows(Path("src/hmm_process/artifacts/mess3_mixed_dataset.jsonl"), 40)
    targets = build_block_targets(rows, seq_len=12)

    sample = torch.stack([targets[k] for k in list(targets.keys())[:120]], dim=0)
    assert torch.all(sample >= 0)
    assert torch.allclose(sample.sum(dim=1), torch.ones(sample.shape[0]), atol=1e-6)


def test_first_step_block_posterior_matches_bayes_rule() -> None:
    specs = [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05)]
    tok = 2
    traj = block_posteriors_for_tokens([tok], specs)

    evidences = []
    for alpha, x in specs:
        p = Mess3Process(alpha=alpha, x=x)
        b0 = p.stationary_distribution()
        _, e = p.update_belief(b0, tok)
        evidences.append(float(e))

    expected = torch.tensor(evidences, dtype=torch.float64)
    expected = expected / expected.sum()
    assert torch.allclose(traj[0].to(torch.float64), expected, atol=1e-6)
