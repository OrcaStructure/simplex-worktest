from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Mess3Params:
    alpha: float
    x: float

    @property
    def beta(self) -> float:
        return (1.0 - self.alpha) / 2.0

    @property
    def y(self) -> float:
        return 1.0 - 2.0 * self.x


class Mess3Process:
    """Mess3 HMM with 3 hidden states and tokens {0,1,2}.

    Labeled matrices follow the equations provided by the user:
      T^(0), T^(1), T^(2), with beta=(1-alpha)/2 and y=1-2x.

    Interpretation:
      T^(k)[i, j] = P(next_state=j, emit_token=k | current_state=i)

    Belief updates live in the 2-simplex over hidden states:
      b' = normalize(b @ T^(k))
    """

    def __init__(
        self,
        alpha: float,
        x: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.params = Mess3Params(alpha=alpha, x=x)
        self.dtype = dtype
        self.device = torch.device(device)
        self.tokens = [0, 1, 2]
        self.num_states = 3

        self._validate_params()
        self.token_matrices = self._build_labeled_matrices()
        self._validate_probabilities()

    def _validate_params(self) -> None:
        a = self.params.alpha
        x = self.params.x
        if not (0.0 <= a <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if not (0.0 <= x <= 0.5):
            raise ValueError("x must be in [0,0.5] so y=1-2x is non-negative")

    def _build_labeled_matrices(self) -> torch.Tensor:
        a = self.params.alpha
        b = self.params.beta
        x = self.params.x
        y = self.params.y

        t0 = torch.tensor(
            [
                [a * y, b * x, b * x],
                [a * x, b * y, b * x],
                [a * x, b * x, b * y],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        t1 = torch.tensor(
            [
                [b * y, a * x, b * x],
                [b * x, a * y, b * x],
                [b * x, a * x, b * y],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        t2 = torch.tensor(
            [
                [b * y, b * x, a * x],
                [b * x, b * y, a * x],
                [b * x, b * x, a * y],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        return torch.stack([t0, t1, t2], dim=0)

    def _validate_probabilities(self) -> None:
        if torch.any(self.token_matrices < 0):
            raise ValueError("Mess3 produced negative entries; check alpha/x range")
        row_totals = self.transition_matrix().sum(dim=1)
        ones = torch.ones_like(row_totals)
        if not torch.allclose(row_totals, ones, atol=1e-10):
            raise ValueError("Rows of net transition matrix must sum to 1")

    @property
    def alpha(self) -> float:
        return self.params.alpha

    @property
    def beta(self) -> float:
        return self.params.beta

    @property
    def x(self) -> float:
        return self.params.x

    @property
    def y(self) -> float:
        return self.params.y

    def token_matrix(self, token: int) -> torch.Tensor:
        if token not in self.tokens:
            raise ValueError(f"token must be one of {self.tokens}")
        return self.token_matrices[token]

    def transition_matrix(self) -> torch.Tensor:
        return self.token_matrices.sum(dim=0)

    def stationary_distribution(self) -> torch.Tensor:
        # For Mess3 as defined, stationary left eigenvector is uniform.
        return torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=self.dtype, device=self.device)

    def right_ones_eigenvector(self) -> torch.Tensor:
        return torch.ones(3, dtype=self.dtype, device=self.device)

    def normalize_belief(self, belief: torch.Tensor) -> torch.Tensor:
        if belief.ndim != 1 or belief.shape[0] != self.num_states:
            raise ValueError("belief must have shape [3]")
        if torch.any(belief < 0):
            raise ValueError("belief entries must be non-negative")
        total = belief.sum()
        if total <= 0:
            raise ValueError("belief must have positive mass")
        return belief / total

    def predictive_token_probs(self, belief: torch.Tensor) -> torch.Tensor:
        b = self.normalize_belief(belief)
        next_by_token = torch.einsum("i,kij->kj", b, self.token_matrices)
        return next_by_token.sum(dim=1)

    def update_belief(self, belief: torch.Tensor, token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.normalize_belief(belief)
        unnormalized = b @ self.token_matrix(token)
        evidence = unnormalized.sum()
        if evidence <= 0:
            raise ValueError(f"Observed token {token} has zero probability under current belief")
        posterior = unnormalized / evidence
        return posterior, evidence

    def forward_filter(
        self, observations: Sequence[int], initial_belief: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beliefs: List[torch.Tensor] = []
        belief = self.normalize_belief(initial_belief)
        log_likelihood = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        for tok in observations:
            belief, evidence = self.update_belief(belief, tok)
            beliefs.append(belief)
            log_likelihood = log_likelihood + torch.log(evidence)

        if len(beliefs) == 0:
            return torch.empty((0, self.num_states), dtype=self.dtype, device=self.device), log_likelihood
        return torch.stack(beliefs, dim=0), log_likelihood

    def sample_step(self, state: int, generator: torch.Generator | None = None) -> Tuple[int, int]:
        if state < 0 or state >= self.num_states:
            raise ValueError("state must be in {0,1,2}")
        probs = self.token_matrices[:, state, :].reshape(-1)
        choice = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        token = choice // self.num_states
        dst = choice % self.num_states
        return token, dst

    def sample_initial_state(
        self, belief: torch.Tensor | None = None, generator: torch.Generator | None = None
    ) -> int:
        if belief is None:
            belief = self.stationary_distribution()
        b = self.normalize_belief(belief)
        return int(torch.multinomial(b, num_samples=1, generator=generator).item())

    def sample_sequence(
        self, initial_state: int, steps: int, generator: torch.Generator | None = None
    ) -> Tuple[List[int], List[int]]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        states = [initial_state]
        obs: List[int] = []
        state = initial_state
        for _ in range(steps):
            token, state = self.sample_step(state, generator=generator)
            obs.append(token)
            states.append(state)
        return states, obs

    def belief_trajectory(
        self,
        observations: Sequence[int],
        initial_belief: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Posterior trajectory in the 2-simplex, including the initial point."""
        if initial_belief is None:
            initial_belief = self.stationary_distribution()
        belief = self.normalize_belief(initial_belief)
        traj: List[torch.Tensor] = [belief]
        for tok in observations:
            belief, _ = self.update_belief(belief, tok)
            traj.append(belief)
        return torch.stack(traj, dim=0)

    def generate_belief_trajectories(
        self,
        num_sequences: int,
        steps: int,
        initial_belief: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tuple[List[torch.Tensor], List[List[int]], List[List[int]]]:
        """Simulate sequences and return belief trajectories from an initial belief.

        Returns:
          trajectories: each tensor has shape [steps + 1, 3] (includes initial belief)
          observations: emitted token sequences
          hidden_states: hidden-state paths used to generate tokens
        """
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")

        if initial_belief is None:
            initial_belief = self.stationary_distribution()
        init = self.normalize_belief(initial_belief)

        trajectories: List[torch.Tensor] = []
        observations: List[List[int]] = []
        hidden_states: List[List[int]] = []

        for _ in range(num_sequences):
            initial_state = self.sample_initial_state(init, generator=generator)
            states, obs = self.sample_sequence(initial_state=initial_state, steps=steps, generator=generator)
            traj = self.belief_trajectory(obs, initial_belief=init)
            trajectories.append(traj)
            observations.append(obs)
            hidden_states.append(states)

        return trajectories, observations, hidden_states

    def plot_sampled_belief_trajectories(
        self,
        num_sequences: int,
        steps: int,
        initial_belief: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        save_path: str | None = None,
        title: str | None = None,
    ) -> Tuple[List[torch.Tensor], List[List[int]], List[List[int]]]:
        """Generate and plot posterior trajectories on the 2-simplex.

        Default initial belief is uniform [1/3, 1/3, 1/3].
        """
        trajectories, observations, states = self.generate_belief_trajectories(
            num_sequences=num_sequences,
            steps=steps,
            initial_belief=initial_belief,
            generator=generator,
        )
        from .simplex_plot import plot_belief_trajectories_on_simplex

        if title is None:
            title = f"Mess3 trajectories (alpha={self.alpha}, x={self.x})"
        plot_belief_trajectories_on_simplex(
            trajectories=trajectories,
            observations=observations,
            title=title,
            save_path=save_path,
        )
        return trajectories, observations, states
