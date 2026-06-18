from __future__ import annotations
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    import gymnasium as gym  # noqa: F401 (imported for env registration)
except ModuleNotFoundError:  # pragma: no cover
    import gym  # type: ignore[no-redef]  # noqa: F401
from sbx import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


ALGO_REGISTRY = {
    "ppo": PPO,
    "sac": SAC,
}

STEPS_PER_GENERATION = 300
TOTAL_GENERATIONS = 500
TOTAL_TIMESTEPS = STEPS_PER_GENERATION * TOTAL_GENERATIONS  # 150,000
SEEDS = (0, 1, 2, 3, 4)


@dataclass
class AlgoRunConfig:
    algo_name: str
    env_id: str
    total_timesteps: int
    n_envs: int
    seeds: Iterable[int]
    log_root: Path = Path("runs")
    algo_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkPlan:
    env_id: str
    algorithms: Iterable[str]
    total_timesteps: int
    n_envs: int
    seeds: Iterable[int]
    algo_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)


DEFAULT_BENCHMARKS = [
    BenchmarkPlan(
        env_id="CartPole-v1",
        algorithms=("ppo",),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=4,
        seeds=SEEDS,
        algo_kwargs={
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "learning_rate": 2.5e-4,
                "n_steps": 128,
                "batch_size": 512,
                "n_epochs": 4,
            },
        },
    ),
    BenchmarkPlan(
         # env_id="LunarLanderContinuous-v2",
        env_id="LunarLanderContinuous-v3",
        algorithms=("ppo", "sac"),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=4,
        seeds=SEEDS,
        algo_kwargs={
            "ppo": {
                "gamma": 0.999,
                "gae_lambda": 0.98,
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "n_steps": 1024,
                "batch_size": 64,
                "n_epochs": 4,
            },
            "sac": {
                "gamma": 0.999,
                "learning_rate": 3e-4,
                "buffer_size": int(1e6),
                "tau": 0.005,
                "batch_size": 256,
                "ent_coef": "auto",
                "train_freq": (1, "step"),
                "gradient_steps": 1,
            },
        },
    ),
    BenchmarkPlan(
        env_id="BipedalWalker-v3",
        algorithms=("ppo", "sac"),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=4,
        seeds=SEEDS,
        algo_kwargs={
            "ppo": {
                "gamma": 0.999,
                "gae_lambda": 0.95,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "clip_range": 0.18,
            },
            "sac": {
                "gamma": 0.999,
                "learning_rate": 3e-4,
                "buffer_size": int(1e6),
                "tau": 0.005,
                "batch_size": 256,
                "ent_coef": "auto",
                "train_freq": (1, "step"),
                "gradient_steps": 1,
            },
        },
    ),
    BenchmarkPlan(
        env_id="Hopper-v4",
        algorithms=("ppo", "sac"),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=2,
        seeds=SEEDS,
        algo_kwargs={
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 64,
                "n_epochs": 4,
            },
            "sac": {
                "gamma": 0.99,
                "learning_rate": 3e-4,
                "buffer_size": int(1e6),
                "tau": 0.005,
                "batch_size": 256,
                "ent_coef": "auto",
                "train_freq": (1, "step"),
                "gradient_steps": 1,
            },
        },
    ),
]


class EpisodeLogger(BaseCallback):
    def __init__(self, seed: int, method: str, env_id: str, steps_per_generation: int = STEPS_PER_GENERATION):
        super().__init__()
        self.seed = seed
        self.method = method
        self.env_id = env_id
        self.steps_per_generation = steps_per_generation
        self.records: List[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                generation = ((self.num_timesteps - 1) // self.steps_per_generation) + 1
                self.records.append(
                    {
                        "method": self.method,
                        "env_id": self.env_id,
                        "seed": self.seed,
                        "timesteps": self.num_timesteps,
                        "generation": generation,
                        "episode_reward": ep["r"],
                        "episode_length": ep["l"],
                    }
                )
        return True

    def save_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "env_id",
                    "seed",
                    "timesteps",
                    "generation",
                    "episode_reward",
                    "episode_length",
                ],
            )
            writer.writeheader()
            writer.writerows(self.records)


def slugify_env(env_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", env_id).lower()


def _train_seed(
    algo_name: str,
    env_id: str,
    total_timesteps: int,
    n_envs: int,
    seed: int,
    log_root: Path,
    algo_kwargs: dict[str, Any],
    steps_per_generation: int = STEPS_PER_GENERATION,
) -> Tuple[Path, int]:
    algo_kwargs = dict(algo_kwargs or {})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"[{algo_name} | seed {seed}] training start for {total_timesteps} timesteps on {env_id}...")
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    callback = EpisodeLogger(seed, algo_name, env_id, steps_per_generation=steps_per_generation)

    algo_cls = ALGO_REGISTRY[algo_name]
    model = algo_cls("MlpPolicy", env, seed=seed, verbose=0, **algo_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()

    exp_dir = log_root / f"{algo_name}_{slugify_env(env_id)}"
    csv_path = exp_dir / f"seed_{seed}" / "learning_curve.csv"
    callback.save_csv(csv_path)
    return csv_path, int(model.num_timesteps)


def aggregate_csv(csv_paths: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "env_id",
                "seed",
                "timesteps",
                "generation",
                "episode_reward",
                "episode_length",
            ],
        )
        writer.writeheader()
        for path in csv_paths:
            with path.open() as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    writer.writerow(row)


def create_statistics_csv(learning_curves_csv: Path, stats_csv: Path, steps_per_generation: int):
    """Create a statistics CSV with mean/std for timesteps and generations."""
    import numpy as np
    from collections import defaultdict

    # Read the aggregated learning curves
    data = []
    with learning_curves_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'method': row['method'],
                'env_id': row['env_id'],
                'seed': int(row['seed']),
                'timesteps': int(float(row['timesteps'])),
                'generation': int(float(row.get('generation', 0))),
                'episode_reward': float(row['episode_reward']),
                'episode_length': float(row['episode_length']),
            })

    # Group by method, env_id, and either timesteps or generation
    timestep_stats = defaultdict(lambda: defaultdict(list))
    generation_stats = defaultdict(lambda: defaultdict(list))

    for row in data:
        key = (row['method'], row['env_id'])
        timestep_stats[key][row['timesteps']].append(row['episode_reward'])
        generation_stats[key][row['generation']].append(row['episode_reward'])

    # Compute stats for timesteps
    timestep_rows = []
    for (method, env_id), ts_rewards in timestep_stats.items():
        for ts, rewards in ts_rewards.items():
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            timestep_rows.append({
                'method': method,
                'env_id': env_id,
                'x_type': 'timesteps',
                'x_value': ts,
                'mean_reward': mean_r,
                'std_reward': std_r,
                'num_episodes': len(rewards),
            })

    # Compute stats for generations
    generation_rows = []
    for (method, env_id), gen_rewards in generation_stats.items():
        for gen, rewards in gen_rewards.items():
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            generation_rows.append({
                'method': method,
                'env_id': env_id,
                'x_type': 'generations',
                'x_value': gen,
                'mean_reward': mean_r,
                'std_reward': std_r,
                'num_episodes': len(rewards),
            })

    # Write to stats CSV
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    with stats_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'env_id', 'x_type', 'x_value', 'mean_reward', 'std_reward', 'num_episodes'
        ])
        writer.writeheader()
        writer.writerows(timestep_rows + generation_rows)


def run_algorithm(cfg: AlgoRunConfig, steps_per_generation: int = STEPS_PER_GENERATION):
    cfg.log_root.mkdir(parents=True, exist_ok=True)
    exp_dir = cfg.log_root / f"{cfg.algo_name}_{slugify_env(cfg.env_id)}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(cfg.seeds)
    with ProcessPoolExecutor(max_workers=len(seeds)) as ex:
        futures = {
            ex.submit(
                _train_seed,
                cfg.algo_name,
                cfg.env_id,
                cfg.total_timesteps,
                cfg.n_envs,
                seed,
                cfg.log_root,
                cfg.algo_kwargs,
                steps_per_generation,
            ): seed
            for seed in seeds
        }
        results = [f.result() for f in futures]

    csv_paths = [r[0] for r in results]
    total_env_steps = sum(r[1] for r in results)

    aggregate_csv(csv_paths, exp_dir / "learning_curves.csv")
    create_statistics_csv(exp_dir / "learning_curves.csv", exp_dir / "statistics.csv", steps_per_generation)
    print(f"[{cfg.algo_name} | {cfg.env_id}] Finished.")
    for p, steps in results:
        print(f"- {p} | env steps: {steps}")
    print(f"Aggregated: {exp_dir / 'learning_curves.csv'}")
    print(f"Statistics: {exp_dir / 'statistics.csv'}")
    print(f"Total environment interactions across seeds: {total_env_steps}")


def run_benchmarks(benchmarks: Iterable[BenchmarkPlan], steps_per_generation: int = STEPS_PER_GENERATION):
    for bench in benchmarks:
        for algo_name in bench.algorithms:
            cfg = AlgoRunConfig(
                algo_name=algo_name,
                env_id=bench.env_id,
                total_timesteps=bench.total_timesteps,
                n_envs=bench.n_envs,
                seeds=bench.seeds,
                algo_kwargs=bench.algo_kwargs.get(algo_name, {}),
            )
            run_algorithm(cfg, steps_per_generation=steps_per_generation)


def main():
    run_benchmarks(DEFAULT_BENCHMARKS, steps_per_generation=STEPS_PER_GENERATION)


if __name__ == "__main__":
    main()