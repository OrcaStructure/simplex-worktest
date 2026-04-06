import torch

from src.hmm_process import Mess3Process


def test_labeled_matrices_match_definition() -> None:
    alpha = 0.8
    x = 0.1
    p = Mess3Process(alpha=alpha, x=x)
    beta = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x

    t0_expected = torch.tensor(
        [
            [alpha * y, beta * x, beta * x],
            [alpha * x, beta * y, beta * x],
            [alpha * x, beta * x, beta * y],
        ],
        dtype=torch.float64,
    )
    t1_expected = torch.tensor(
        [
            [beta * y, alpha * x, beta * x],
            [beta * x, alpha * y, beta * x],
            [beta * x, alpha * x, beta * y],
        ],
        dtype=torch.float64,
    )
    t2_expected = torch.tensor(
        [
            [beta * y, beta * x, alpha * x],
            [beta * x, beta * y, alpha * x],
            [beta * x, beta * x, alpha * y],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(p.token_matrix(0), t0_expected)
    assert torch.allclose(p.token_matrix(1), t1_expected)
    assert torch.allclose(p.token_matrix(2), t2_expected)


def test_net_transition_matrix_and_stationary_uniform() -> None:
    p = Mess3Process(alpha=0.73, x=0.17)
    t = p.transition_matrix()

    expected = torch.tensor(
        [
            [1.0 - 2.0 * p.x, p.x, p.x],
            [p.x, 1.0 - 2.0 * p.x, p.x],
            [p.x, p.x, 1.0 - 2.0 * p.x],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(t, expected)
    assert torch.allclose(t.sum(dim=1), torch.ones(3, dtype=torch.float64))

    pi = p.stationary_distribution()
    ones = p.right_ones_eigenvector()
    assert torch.allclose(pi @ t, pi)
    assert torch.allclose(t @ ones, ones)


def test_belief_update_returns_simplex_point_and_evidence() -> None:
    p = Mess3Process(alpha=0.8, x=0.1)
    b0 = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    b1, evidence = p.update_belief(b0, token=1)

    assert b1.shape == (3,)
    assert torch.all(b1 >= 0)
    assert torch.isclose(b1.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert evidence > 0

    manual_unnorm = (b0 / b0.sum()) @ p.token_matrix(1)
    assert torch.isclose(evidence, manual_unnorm.sum())
    assert torch.allclose(b1, manual_unnorm / manual_unnorm.sum())


def test_forward_filter_and_sampling_sanity() -> None:
    p = Mess3Process(alpha=0.9, x=0.05)
    initial = p.stationary_distribution()

    beliefs, ll = p.forward_filter([0, 2, 1, 1, 0], initial)
    assert beliefs.shape == (5, 3)
    assert torch.isfinite(ll)
    assert torch.allclose(beliefs.sum(dim=1), torch.ones(5, dtype=torch.float64))

    gen = torch.Generator().manual_seed(3)
    states, obs = p.sample_sequence(initial_state=0, steps=20, generator=gen)
    assert len(states) == 21
    assert len(obs) == 20
    assert all(s in {0, 1, 2} for s in states)
    assert all(o in {0, 1, 2} for o in obs)


def test_generated_trajectories_start_from_uniform_prior() -> None:
    p = Mess3Process(alpha=0.85, x=0.08)
    init = p.stationary_distribution()
    gen = torch.Generator().manual_seed(11)

    trajectories, obs_list, state_list = p.generate_belief_trajectories(
        num_sequences=4,
        steps=6,
        initial_belief=init,
        generator=gen,
    )

    assert len(trajectories) == 4
    assert len(obs_list) == 4
    assert len(state_list) == 4
    for traj, obs, states in zip(trajectories, obs_list, state_list):
        assert traj.shape == (7, 3)
        assert torch.allclose(traj[0], init)
        assert torch.allclose(traj.sum(dim=1), torch.ones(7, dtype=torch.float64))
        assert len(obs) == 6
        assert len(states) == 7


def test_invalid_params_rejected() -> None:
    for alpha, x in [(-0.1, 0.1), (1.1, 0.1), (0.5, -0.1), (0.5, 0.6)]:
        try:
            Mess3Process(alpha=alpha, x=x)
            assert False, f"expected ValueError for alpha={alpha}, x={x}"
        except ValueError:
            pass
