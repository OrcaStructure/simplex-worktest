from collections import Counter

from src.hmm_process import generate_equal_mixture_dataset


def test_equal_mixture_counts_and_vocab() -> None:
    specs = [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05)]
    rows = generate_equal_mixture_dataset(
        specs=specs,
        sequences_per_process=12,
        steps=20,
        seed=3,
        shuffle=True,
        include_states=False,
    )

    assert len(rows) == 36
    counts = Counter(row["process_id"] for row in rows)
    assert counts == {0: 12, 1: 12, 2: 12}

    for row in rows:
        assert len(row["tokens"]) == 20
        assert all(tok in {0, 1, 2} for tok in row["tokens"])
        assert "hidden_states" not in row


def test_include_states_shape() -> None:
    specs = [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05)]
    rows = generate_equal_mixture_dataset(
        specs=specs,
        sequences_per_process=2,
        steps=7,
        seed=1,
        shuffle=False,
        include_states=True,
    )
    assert len(rows) == 6
    for row in rows:
        assert len(row["tokens"]) == 7
        assert len(row["hidden_states"]) == 8
        assert all(s in {0, 1, 2} for s in row["hidden_states"])


def test_supports_arbitrary_process_count() -> None:
    specs = [(0.8, 0.1), (0.7, 0.15), (0.9, 0.05), (0.85, 0.12), (0.75, 0.2)]
    rows = generate_equal_mixture_dataset(
        specs=specs,
        sequences_per_process=3,
        steps=9,
        seed=11,
        shuffle=False,
        include_states=False,
    )
    assert len(rows) == 15
    counts = Counter(row["process_id"] for row in rows)
    assert counts == {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
