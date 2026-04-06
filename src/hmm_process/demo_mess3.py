from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hmm_process import Mess3Process, plot_belief_trajectories_on_simplex


def main() -> None:
    parser = argparse.ArgumentParser(description="Mess3 process demo: matrices, filtering, sampling, and simplex trajectories")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--obs", type=str, default="0,1,2,1,0", help="comma-separated tokens in {0,1,2}")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-path", type=str, default="src/hmm_process/artifacts/mess3_trajectories.png")
    args = parser.parse_args()

    p = Mess3Process(alpha=args.alpha, x=args.x)

    print(f"alpha={p.alpha:.4f}, beta={p.beta:.4f}, x={p.x:.4f}, y={p.y:.4f}")
    for tok in p.tokens:
        print(f"\nT^({tok}) =\n{p.token_matrix(tok)}")
    print(f"\nNet T =\n{p.transition_matrix()}")
    print(f"\nStationary pi = {p.stationary_distribution()}")

    initial = p.stationary_distribution()
    observations = [int(s.strip()) for s in args.obs.split(",") if s.strip()]
    beliefs, ll = p.forward_filter(observations, initial)
    print(f"\nobservations = {observations}")
    print(f"log-likelihood = {float(ll):.6f}")
    for t in range(beliefs.shape[0]):
        print(f"belief[{t}] = {beliefs[t]}")

    gen = torch.Generator().manual_seed(args.seed)
    states, obs = p.sample_sequence(initial_state=0, steps=args.sample_steps, generator=gen)
    print(f"\nsampled states = {states}")
    print(f"sampled tokens = {obs}")

    trajectories, obs_list, _ = p.generate_belief_trajectories(
        num_sequences=args.num_trajectories,
        steps=args.sample_steps,
        initial_belief=initial,
        generator=gen,
    )
    plot_belief_trajectories_on_simplex(
        trajectories,
        observations=obs_list,
        title=f"Mess3 trajectories (alpha={p.alpha}, x={p.x})",
        save_path=args.plot_path,
    )
    print(f"\nSaved simplex plot to: {args.plot_path}")


if __name__ == "__main__":
    main()
