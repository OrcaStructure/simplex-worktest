#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.simple_transformer_residual import Config, TinyTransformer


EPS = 1e-8


@dataclass
class RunSpec:
    name: str
    vocab_size: int
    num_processes: int
    sequences_per_process: int
    steps: int
    sparse_emission: bool
    sparse_process_id: int
    sparse_state_id: int
    sparse_allowed_tokens: List[int]
    seed: int


@dataclass
class ModelSpec:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    seq_len: int


class GenericTokenProcess:
    """3-state HMM represented by token-labeled transition matrices T^k[i,j]."""

    def __init__(self, token_matrices: torch.Tensor):
        # token_matrices: [V, 3, 3], nonnegative, row-stochastic over (k,j) for each i
        if token_matrices.ndim != 3 or token_matrices.shape[1:] != (3, 3):
            raise ValueError("token_matrices must have shape [vocab,3,3]")
        if torch.any(token_matrices < 0):
            raise ValueError("token_matrices must be nonnegative")
        self.token_matrices = token_matrices.to(dtype=torch.float64, device="cpu")
        self.vocab_size = int(token_matrices.shape[0])

        row_totals = self.transition_matrix().sum(dim=1)
        if not torch.allclose(row_totals, torch.ones_like(row_totals), atol=1e-8):
            raise ValueError("Rows of transition matrix must sum to 1")

    def transition_matrix(self) -> torch.Tensor:
        return self.token_matrices.sum(dim=0)

    def stationary_distribution(self, max_iter: int = 5000) -> torch.Tensor:
        # Power iteration on left distribution.
        b = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
        t = self.transition_matrix()
        for _ in range(max_iter):
            b_next = b @ t
            if torch.max(torch.abs(b_next - b)).item() < 1e-12:
                return b_next / b_next.sum()
            b = b_next
        return b / b.sum()

    def normalize_belief(self, b: torch.Tensor) -> torch.Tensor:
        bb = b.clamp_min(0)
        s = bb.sum()
        if s <= 0:
            raise ValueError("belief has no mass")
        return bb / s

    def update_belief(self, belief: torch.Tensor, token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.normalize_belief(belief)
        u = b @ self.token_matrices[int(token)]
        evidence = u.sum()
        if evidence <= 0:
            raise ValueError(f"token {token} has zero evidence")
        return u / evidence, evidence

    def sample_step(self, state: int, g: torch.Generator) -> Tuple[int, int]:
        probs = self.token_matrices[:, state, :].reshape(-1)
        pick = int(torch.multinomial(probs, num_samples=1, generator=g).item())
        token = pick // 3
        nxt = pick % 3
        return token, nxt

    def sample_sequence(self, initial_state: int, steps: int, g: torch.Generator) -> Tuple[List[int], List[int]]:
        states = [int(initial_state)]
        obs: List[int] = []
        s = int(initial_state)
        for _ in range(steps):
            tok, s = self.sample_step(s, g)
            obs.append(tok)
            states.append(s)
        return states, obs

    def belief_trajectory(self, observations: List[int], initial_belief: torch.Tensor | None = None) -> torch.Tensor:
        b = self.stationary_distribution() if initial_belief is None else self.normalize_belief(initial_belief)
        out = [b]
        for tok in observations:
            b, _ = self.update_belief(b, int(tok))
            out.append(b)
        return torch.stack(out, dim=0)


def ensure_int(x: Any, name: str, min_value: int = 1) -> int:
    v = int(x)
    if v < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return v


def run_cmd(cmd: List[str], log_path: Path, *, verbose: bool, step_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n=== {step_name} ===")
        print(f"$ {' '.join(cmd)}")
        print(f"log: {log_path}")

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            if verbose:
                print(line.rstrip())
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}. See {log_path}")
    if verbose:
        print(f"=== done: {step_name} ===")


def sample_token_matrices(
    vocab_size: int,
    *,
    seed: int,
    sparse_state_id: int | None,
    sparse_allowed_tokens: List[int] | None,
) -> torch.Tensor:
    """Create [V,3,3] labeled matrices, row-stochastic over (token,next_state)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros((vocab_size, 3, 3), dtype=torch.float64)

    for s in range(3):
        allowed = list(range(vocab_size))
        if sparse_state_id is not None and s == sparse_state_id:
            if not sparse_allowed_tokens:
                raise ValueError("sparse_allowed_tokens cannot be empty")
            allowed = sorted(set(int(x) for x in sparse_allowed_tokens))
            if min(allowed) < 0 or max(allowed) >= vocab_size:
                raise ValueError("sparse allowed token index out of range")

        # random mass over allowed tokens x 3 next states
        logits = torch.rand((len(allowed), 3), generator=g, dtype=torch.float64)
        probs = logits / logits.sum()

        for i, tok in enumerate(allowed):
            t[tok, s, :] = probs[i]

    return t


def generate_dataset_for_run(run: RunSpec, out_dir: Path, *, verbose: bool) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / f"{run.name}.jsonl"
    meta_path = out_dir / f"{run.name}_metadata.json"

    process_specs = []
    rows: List[dict] = []

    if verbose:
        print(f"\n--- building dataset {run.name} ---")
        print(
            f"vocab={run.vocab_size} processes={run.num_processes} "
            f"seq_per_process={run.sequences_per_process} steps={run.steps} seed={run.seed}"
        )
        if run.sparse_emission:
            print(
                "sparse condition: "
                f"process_id={run.sparse_process_id}, state_id={run.sparse_state_id}, "
                f"allowed_tokens={run.sparse_allowed_tokens}"
            )

    for pid in range(run.num_processes):
        sparse_state = None
        sparse_tokens = None
        if run.sparse_emission and pid == run.sparse_process_id:
            sparse_state = run.sparse_state_id
            sparse_tokens = run.sparse_allowed_tokens

        tm = sample_token_matrices(
            run.vocab_size,
            seed=run.seed + 100 * pid,
            sparse_state_id=sparse_state,
            sparse_allowed_tokens=sparse_tokens,
        )
        proc = GenericTokenProcess(tm)
        process_specs.append(
            {
                "process_id": pid,
                "token_matrices": tm.tolist(),
                "sparse_state_id": sparse_state,
                "sparse_allowed_tokens": sparse_tokens,
            }
        )

        g = torch.Generator().manual_seed(run.seed + 10_000 + pid)
        init_b = proc.stationary_distribution()

        for _ in range(run.sequences_per_process):
            init_state = int(torch.multinomial(init_b, num_samples=1, generator=g).item())
            _, tokens = proc.sample_sequence(initial_state=init_state, steps=run.steps, g=g)
            rows.append(
                {
                    "process_id": pid,
                    "tokens": [int(x) for x in tokens],
                }
            )

    # Shuffle rows deterministically.
    gg = torch.Generator().manual_seed(run.seed)
    perm = torch.randperm(len(rows), generator=gg).tolist()
    rows = [rows[i] for i in perm]

    with dataset_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    meta = {
        "run": run.__dict__,
        "num_states": 3,
        "num_processes": run.num_processes,
        "vocab_size": run.vocab_size,
        "process_specs": process_specs,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"dataset rows={len(rows)} written: {dataset_path}")
        print(f"dataset metadata: {meta_path}")
    return dataset_path, meta_path


def load_rows(path: Path, max_sequences: int | None = None) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_sequences is not None and len(rows) >= max_sequences:
                break
    return rows


def load_checkpoint(path: Path, device: torch.device) -> Tuple[TinyTransformer, Config]:
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def build_residual_dataset(
    model: TinyTransformer,
    cfg: Config,
    rows: List[dict],
    process_lookup: Dict[int, GenericTokenProcess],
    cache_key: str,
    *,
    device: torch.device,
    eval_batch_size: int,
    verbose: bool,
    progress_every: int,
    progress_label: str,
) -> dict:
    xs = []
    ys = []
    pids = []
    sids = []
    poss = []

    prepared: List[Tuple[int, int, List[int]]] = []
    for seq_idx, row in enumerate(rows):
        toks = [int(t) for t in row["tokens"]]
        pid = int(row["process_id"])
        if len(toks) < cfg.seq_len:
            continue
        prepared.append((seq_idx, pid, toks[: cfg.seq_len]))

    total = len(prepared)
    if verbose:
        print(f"[{progress_label}] extracting residuals for {total} sequences with batch_size={eval_batch_size} on {device}")

    for start in range(0, total, eval_batch_size):
        chunk = prepared[start : start + eval_batch_size]
        batch_tokens = torch.tensor([x for _, _, x in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            _, cache = model(batch_tokens, capture_residuals=True)
        residuals_batch = cache[cache_key].detach().cpu()  # [B,T,d]

        for b_idx, (seq_idx, pid, x_tokens) in enumerate(chunk):
            residuals = residuals_batch[b_idx]
            proc = process_lookup[pid]
            traj = proc.belief_trajectory(x_tokens)

            for pos in range(cfg.seq_len):
                xs.append(residuals[pos].to(torch.float32))
                ys.append(traj[pos + 1].to(torch.float32))
                pids.append(pid)
                sids.append(seq_idx)
                poss.append(pos)

        done = min(start + len(chunk), total)
        if verbose and (done % progress_every == 0 or done == total):
            print(f"[{progress_label}] residual extraction progress: {done}/{total}")

    return {
        "X": torch.stack(xs, dim=0),
        "Y": torch.stack(ys, dim=0),
        "process_id": torch.tensor(pids, dtype=torch.long),
        "sequence_id": torch.tensor(sids, dtype=torch.long),
        "position": torch.tensor(poss, dtype=torch.long),
    }


def _fit_linear_map(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype)
    XA = torch.cat([X, ones], dim=1)
    B = torch.linalg.lstsq(XA, Y).solution
    return B[:-1].T.contiguous(), B[-1].contiguous()


def _split_indices(n: int, train_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = max(1, min(n - 1, int(n * train_frac)))
    return perm[:n_train], perm[n_train:]


def _build_joint_targets(Y: torch.Tensor, pid: torch.Tensor, num_processes: int) -> torch.Tensor:
    out = torch.zeros((Y.shape[0], num_processes * 3), dtype=Y.dtype)
    for p in range(num_processes):
        idx = torch.where(pid == p)[0]
        if idx.numel() > 0:
            out[idx, 3 * p : 3 * (p + 1)] = Y[idx]
    return out


def to_alr(y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    yy = y.clamp_min(eps)
    return torch.log(yy[:, :-1] / yy[:, -1:])


def from_alr(c: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((c.shape[0], 1), dtype=c.dtype, device=c.device)
    return torch.softmax(torch.cat([c, z], dim=1), dim=1)


def mean_kl(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = EPS) -> float:
    y = y_true.clamp_min(eps)
    p = y_pred.clamp_min(eps)
    return float((y * (torch.log(y) - torch.log(p))).sum(dim=1).mean().item())


def fit_eval_alr_kl(X: torch.Tensor, Y: torch.Tensor, seed: int, train_frac: float) -> float:
    tr, va = _split_indices(X.shape[0], train_frac, seed)
    C = to_alr(Y)
    W, b = _fit_linear_map(X[tr], C[tr])
    pred = from_alr(X[va] @ W.T + b)
    return mean_kl(Y[va], pred)


def build_block_targets(rows: List[dict], process_lookup: Dict[int, GenericTokenProcess], seq_len: int) -> Dict[Tuple[int, int], torch.Tensor]:
    num_processes = len(process_lookup)
    pri = torch.full((num_processes,), 1.0 / num_processes, dtype=torch.float64)
    init_beliefs = {pid: p.stationary_distribution() for pid, p in process_lookup.items()}

    targets: Dict[Tuple[int, int], torch.Tensor] = {}
    for seq_idx, row in enumerate(rows):
        tokens = [int(t) for t in row["tokens"][:seq_len]]
        beliefs = {pid: b.clone() for pid, b in init_beliefs.items()}
        w = pri.clone()
        for pos, tok in enumerate(tokens):
            evid = torch.zeros(num_processes, dtype=torch.float64)
            new_b: Dict[int, torch.Tensor] = {}
            for pid in range(num_processes):
                b_new, e = process_lookup[pid].update_belief(beliefs[pid], tok)
                new_b[pid] = b_new
                evid[pid] = e
            w = (w * evid).clamp_min(1e-18)
            w = w / w.sum()
            beliefs = new_b
            targets[(seq_idx, pos)] = w.to(torch.float32)
    return targets


def align_xy(ds: dict, targets: Dict[Tuple[int, int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = ds["sequence_id"].tolist()
    pos = ds["position"].tolist()
    keep = []
    ys = []
    for i, (s, p) in enumerate(zip(seq, pos)):
        key = (int(s), int(p))
        if key in targets:
            keep.append(i)
            ys.append(targets[key])
    idx = torch.tensor(keep, dtype=torch.long)
    return ds["X"][idx], torch.stack(ys, dim=0)


def rowspace_basis(W: torch.Tensor, tol: float = 1e-8) -> Tuple[torch.Tensor, int]:
    _, s, vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    r = int((s > tol).sum().item())
    q = vh[:r].T.contiguous() if r > 0 else torch.zeros((W.shape[1], 0), dtype=torch.float64)
    return q, r


def subspace_stats(q1: torch.Tensor, q2: torch.Tensor) -> dict:
    if q1.shape[1] == 0 or q2.shape[1] == 0:
        return {"k": 0, "mean_angle_deg": None, "min_angle_deg": None, "max_angle_deg": None, "avg_cos2": None}
    s = torch.linalg.svdvals(q1.T @ q2).clamp(0.0, 1.0)
    ang = torch.rad2deg(torch.acos(s))
    k = min(q1.shape[1], q2.shape[1])
    return {
        "k": int(k),
        "mean_angle_deg": float(ang.mean().item()),
        "min_angle_deg": float(ang.min().item()),
        "max_angle_deg": float(ang.max().item()),
        "avg_cos2": float((s.pow(2).sum() / k).item()),
    }


def eval_case(
    ckpt_path: Path,
    rows: List[dict],
    process_lookup: Dict[int, GenericTokenProcess],
    cache_key: str,
    train_frac: float,
    seed: int,
    eval_device: torch.device,
    eval_batch_size: int,
    verbose: bool,
    progress_every: int,
    progress_label: str,
) -> dict:
    model, cfg = load_checkpoint(ckpt_path, eval_device)
    ds = build_residual_dataset(
        model,
        cfg,
        rows,
        process_lookup,
        cache_key,
        device=eval_device,
        eval_batch_size=eval_batch_size,
        verbose=verbose,
        progress_every=progress_every,
        progress_label=progress_label,
    )
    ds["X"] = ds["X"].float()

    pid = ds["process_id"]
    per_process = {}
    process_maps = {}
    process_bases = {}
    process_ranks = {}
    for p in sorted(int(v) for v in pid.unique().tolist()):
        idx = torch.where(pid == p)[0]
        per_process[str(p)] = fit_eval_alr_kl(ds["X"][idx], ds["Y"][idx], seed=seed + p, train_frac=train_frac)
        c = to_alr(ds["Y"][idx])
        W, _ = _fit_linear_map(ds["X"][idx], c)
        process_maps[f"p{p}"] = W
        q, r = rowspace_basis(W)
        process_bases[f"p{p}"] = q
        process_ranks[f"p{p}"] = r

    y_joint = _build_joint_targets(ds["Y"], pid, num_processes=len(process_lookup))
    joint_kl = fit_eval_alr_kl(ds["X"], y_joint, seed=seed + 123, train_frac=train_frac)

    block_targets = build_block_targets(rows, process_lookup, seq_len=cfg.seq_len)
    xb, yb = align_xy(ds, block_targets)
    block_kl = fit_eval_alr_kl(xb, yb, seed=seed + 777, train_frac=train_frac)
    wb, _ = _fit_linear_map(xb, to_alr(yb))
    process_maps["block"] = wb
    q, r = rowspace_basis(wb)
    process_bases["block"] = q
    process_ranks["block"] = r

    names = sorted(process_maps.keys())
    within = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            within[f"{a}_vs_{b}"] = subspace_stats(process_bases[a], process_bases[b])

    return {
        "input_dim": int(ds["X"].shape[1]),
        "per_process_kl_val": per_process,
        "joint_simplex_kl_val": float(joint_kl),
        "block_simplex_kl_val": float(block_kl),
        "rowspace_ranks": process_ranks,
        "rowspace_within_case": within,
        "_maps_for_cross": {k: v.tolist() for k, v in process_maps.items()},
    }


def summarize_cross_case(trained: dict, control: dict) -> dict:
    out = {}
    keys = sorted(set(trained["_maps_for_cross"].keys()) & set(control["_maps_for_cross"].keys()))
    for k in keys:
        wt = torch.tensor(trained["_maps_for_cross"][k], dtype=torch.float64)
        wc = torch.tensor(control["_maps_for_cross"][k], dtype=torch.float64)
        qt, _ = rowspace_basis(wt)
        qc, _ = rowspace_basis(wc)
        out[f"trained_vs_control_{k}"] = subspace_stats(qt, qc)
    return out


def parse_run_spec(d: Dict[str, Any]) -> RunSpec:
    return RunSpec(
        name=str(d["name"]),
        vocab_size=ensure_int(d.get("vocab_size", 3), "vocab_size", min_value=2),
        num_processes=ensure_int(d.get("num_processes", 3), "num_processes", min_value=2),
        sequences_per_process=ensure_int(d.get("sequences_per_process", 5000), "sequences_per_process"),
        steps=ensure_int(d.get("steps", 64), "steps", min_value=4),
        sparse_emission=bool(d.get("sparse_emission", False)),
        sparse_process_id=int(d.get("sparse_process_id", 0)),
        sparse_state_id=int(d.get("sparse_state_id", 0)),
        sparse_allowed_tokens=[int(x) for x in d.get("sparse_allowed_tokens", [0])],
        seed=int(d.get("seed", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep over vocabulary size and run KL/orthogonality experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="artifacts/sweeps_vocab")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Reduce terminal logging")
    args = parser.parse_args()
    verbose = not args.quiet

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    global_cfg = cfg.get("global", {})

    run_name = str(global_cfg.get("run_name", "vocab_token_sweep"))
    out_root = Path(args.out_root) / run_name
    datasets_dir = out_root / "datasets"
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    metrics_dir = out_root / "metrics"

    model_spec = ModelSpec(
        n_layers=ensure_int(global_cfg.get("n_layers", 2), "n_layers"),
        d_model=ensure_int(global_cfg.get("d_model", 48), "d_model"),
        n_heads=ensure_int(global_cfg.get("n_heads", 4), "n_heads"),
        d_ff=ensure_int(global_cfg.get("d_ff", 96), "d_ff"),
        seq_len=ensure_int(global_cfg.get("seq_len", 12), "seq_len"),
    )
    if model_spec.d_model % model_spec.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    train_steps = int(global_cfg.get("train_steps", 2400))
    batch_size = int(global_cfg.get("batch_size", 64))
    eval_batches = int(global_cfg.get("eval_batches", 40))
    train_frac = float(global_cfg.get("train_frac", 0.9))
    lr = float(global_cfg.get("lr", 3e-4))
    device = str(global_cfg.get("device", "cuda"))
    eval_device_str = str(global_cfg.get("eval_device", device))
    eval_batch_size = int(global_cfg.get("eval_batch_size", 256))
    eval_progress_every = int(global_cfg.get("eval_progress_every", 2000))
    base_seed = int(global_cfg.get("seed", 0))
    eval_device = torch.device(eval_device_str)
    if eval_device.type == "cuda" and not torch.cuda.is_available():
        print("eval_device=cuda requested but unavailable; falling back to CPU")
        eval_device = torch.device("cpu")

    runs = [parse_run_spec(d) for d in cfg["runs"]]
    if verbose:
        print("=== vocab token sweep config ===")
        print(f"config: {args.config}")
        print(f"run_name: {run_name}")
        print(
            f"model: n_layers={model_spec.n_layers} d_model={model_spec.d_model} "
            f"n_heads={model_spec.n_heads} d_ff={model_spec.d_ff} seq_len={model_spec.seq_len}"
        )
        print(
            f"train: steps={train_steps} batch={batch_size} eval_batches={eval_batches} "
            f"train_frac={train_frac} lr={lr} device={device} base_seed={base_seed}"
        )
        print(
            f"analysis/eval: eval_device={eval_device} eval_batch_size={eval_batch_size} "
            f"eval_progress_every={eval_progress_every}"
        )
        print(f"num_runs={len(runs)}")
        for rr in runs:
            print(
                f"  - {rr.name}: vocab={rr.vocab_size} sparse={rr.sparse_emission} "
                f"proc={rr.num_processes} seq/proc={rr.sequences_per_process}"
            )

    summary_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "config_path": str(Path(args.config)),
        "run_name": run_name,
        "global": global_cfg,
        "runs": [],
    }

    for i, run in enumerate(runs):
        if verbose:
            print("\n############################################################")
            print(f"RUN {i+1}/{len(runs)} :: {run.name}")
            print("############################################################")
        run_seed = base_seed + 1000 * i
        run_out = out_root / run.name
        run_out.mkdir(parents=True, exist_ok=True)

        dataset_path, meta_path = generate_dataset_for_run(run, datasets_dir, verbose=verbose)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        process_lookup = {
            int(ps["process_id"]): GenericTokenProcess(torch.tensor(ps["token_matrices"], dtype=torch.float64))
            for ps in meta["process_specs"]
        }

        trained_ckpt = ckpt_dir / f"{run.name}_trained.pt"
        control_ckpt = ckpt_dir / f"{run.name}_control_random_init.pt"

        if not args.skip_training:
            cmd_train = [
                args.python,
                "src/simple_transformer_residual.py",
                "--device",
                device,
                "--dataset-path",
                str(dataset_path),
                "--vocab-size",
                str(run.vocab_size),
                "--seq-len",
                str(model_spec.seq_len),
                "--d-model",
                str(model_spec.d_model),
                "--n-heads",
                str(model_spec.n_heads),
                "--d-ff",
                str(model_spec.d_ff),
                "--n-layers",
                str(model_spec.n_layers),
                "--train-steps",
                str(train_steps),
                "--batch-size",
                str(batch_size),
                "--eval-batches",
                str(eval_batches),
                "--train-frac",
                str(train_frac),
                "--lr",
                str(lr),
                "--seed",
                str(run_seed),
                "--no-wandb",
                "--save-model-path",
                str(trained_ckpt),
            ]
            run_cmd(
                cmd_train,
                logs_dir / f"{run.name}_train.log",
                verbose=verbose,
                step_name=f"{run.name}: train trained checkpoint",
            )

            # Control: same architecture/vocab, untrained random init checkpoint.
            cmd_control = [
                args.python,
                "src/simple_transformer_residual.py",
                "--device",
                device,
                "--dataset-path",
                str(dataset_path),
                "--vocab-size",
                str(run.vocab_size),
                "--seq-len",
                str(model_spec.seq_len),
                "--d-model",
                str(model_spec.d_model),
                "--n-heads",
                str(model_spec.n_heads),
                "--d-ff",
                str(model_spec.d_ff),
                "--n-layers",
                str(model_spec.n_layers),
                "--train-steps",
                "0",
                "--batch-size",
                str(batch_size),
                "--eval-batches",
                str(eval_batches),
                "--train-frac",
                str(train_frac),
                "--lr",
                str(lr),
                "--seed",
                str(run_seed + 77),
                "--no-wandb",
                "--save-model-path",
                str(control_ckpt),
            ]
            run_cmd(
                cmd_control,
                logs_dir / f"{run.name}_control.log",
                verbose=verbose,
                step_name=f"{run.name}: create control random-init checkpoint",
            )

        rows = load_rows(dataset_path)

        run_metrics = {
            "run_name": run.name,
            "vocab_size": run.vocab_size,
            "sparse_emission": run.sparse_emission,
            "cases": {},
        }
        for cache_tag, cache_key in (("final_ln", "final_ln"), ("layer1", "layer_0_after_mlp")):
            if verbose:
                print(f"\n[{run.name}] evaluating cache={cache_tag} ({cache_key})")
            trained = eval_case(
                trained_ckpt,
                rows,
                process_lookup,
                cache_key,
                train_frac=train_frac,
                seed=run_seed,
                eval_device=eval_device,
                eval_batch_size=eval_batch_size,
                verbose=verbose,
                progress_every=eval_progress_every,
                progress_label=f"{run.name}:{cache_tag}:trained",
            )
            control = eval_case(
                control_ckpt,
                rows,
                process_lookup,
                cache_key,
                train_frac=train_frac,
                seed=run_seed,
                eval_device=eval_device,
                eval_batch_size=eval_batch_size,
                verbose=verbose,
                progress_every=eval_progress_every,
                progress_label=f"{run.name}:{cache_tag}:control",
            )
            cross = summarize_cross_case(trained, control)
            if verbose:
                tr_pp = float(sum(trained["per_process_kl_val"].values()) / len(trained["per_process_kl_val"]))
                ct_pp = float(sum(control["per_process_kl_val"].values()) / len(control["per_process_kl_val"]))
                print(
                    f"[{run.name}][{cache_tag}] perproc_kl mean trained={tr_pp:.6f} control={ct_pp:.6f}"
                )
                print(
                    f"[{run.name}][{cache_tag}] joint_kl trained={trained['joint_simplex_kl_val']:.6f} "
                    f"control={control['joint_simplex_kl_val']:.6f}"
                )
                print(
                    f"[{run.name}][{cache_tag}] block_kl trained={trained['block_simplex_kl_val']:.6f} "
                    f"control={control['block_simplex_kl_val']:.6f}"
                )
                p0 = cross.get("trained_vs_control_p0", {})
                blk = cross.get("trained_vs_control_block", {})
                print(
                    f"[{run.name}][{cache_tag}] cross-map cos2 p0={p0.get('avg_cos2')} "
                    f"block={blk.get('avg_cos2')}"
                )

            # Drop private map payload from persisted output.
            trained.pop("_maps_for_cross", None)
            control.pop("_maps_for_cross", None)

            run_metrics["cases"][cache_tag] = {
                "trained": trained,
                "control_random_init": control,
                "cross_case_same_map": cross,
            }

            summary_rows.append(
                {
                    "run_name": run.name,
                    "vocab_size": run.vocab_size,
                    "sparse_emission": int(run.sparse_emission),
                    "cache": cache_tag,
                    "trained_joint_kl": trained["joint_simplex_kl_val"],
                    "control_joint_kl": control["joint_simplex_kl_val"],
                    "trained_block_kl": trained["block_simplex_kl_val"],
                    "control_block_kl": control["block_simplex_kl_val"],
                    "trained_perproc_kl_mean": float(sum(trained["per_process_kl_val"].values()) / len(trained["per_process_kl_val"])),
                    "control_perproc_kl_mean": float(sum(control["per_process_kl_val"].values()) / len(control["per_process_kl_val"])),
                    "cross_map_p0_avg_cos2": cross.get("trained_vs_control_p0", {}).get("avg_cos2"),
                    "cross_map_block_avg_cos2": cross.get("trained_vs_control_block", {}).get("avg_cos2"),
                }
            )

        metrics_path = metrics_dir / f"{run.name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        if verbose:
            print(f"[{run.name}] metrics written: {metrics_path}")

        manifest["runs"].append(
            {
                "name": run.name,
                "vocab_size": run.vocab_size,
                "sparse_emission": run.sparse_emission,
                "dataset_path": str(dataset_path),
                "metadata_path": str(meta_path),
                "trained_checkpoint": str(trained_ckpt),
                "control_checkpoint": str(control_ckpt),
                "metrics_json": str(metrics_path),
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if summary_rows:
        csv_path = out_root / "summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        if verbose:
            print(f"summary rows: {len(summary_rows)}")

    print(f"Saved sweep outputs in {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    print(f"Summary: {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
