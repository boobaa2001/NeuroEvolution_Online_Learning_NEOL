from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    env_name: str
    max_episode_steps: int
    num_inputs: int
    num_outputs: int
    config_base: str = "config_base"
    config_plastic: str = "config_plastic"


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "cartpole": BenchmarkSpec(
        name="cartpole",
        env_name="CartPole-v1",
        max_episode_steps=500,
        num_inputs=4,
        num_outputs=2,
    ),
    "bipedalwalker": BenchmarkSpec(
        name="bipedalwalker",
        env_name="BipedalWalker-v3",
        max_episode_steps=1600,
        num_inputs=24,
        num_outputs=4,
    ),
    "lunarlander": BenchmarkSpec(
        name="lunarlander",
        env_name="LunarLander-v3",
        max_episode_steps=1000,
        num_inputs=8,
        num_outputs=4,
    ),
    "hopper": BenchmarkSpec(
        name="hopper",
        env_name="Hopper-v5",
        max_episode_steps=1000,
        num_inputs=11,
        num_outputs=3,
    ),
}

ALIASES = {
    "cart": "cartpole",
    "cart-pole": "cartpole",
    "bipedal": "bipedalwalker",
    "bipedal-walker": "bipedalwalker",
    "bipedal_walker": "bipedalwalker",
    "lunar": "lunarlander",
    "lunar-lander": "lunarlander",
    "lunar_lander": "lunarlander",
}


def available_benchmark_names() -> Iterable[str]:
    return sorted([*BENCHMARKS, "custom"])


def resolve_benchmark_name(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def build_benchmark(
    name: str,
    *,
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    env_name: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    config_base: str = "config_base",
    config_plastic: str = "config_plastic",
) -> BenchmarkSpec:
    key = resolve_benchmark_name(name)

    if key not in BENCHMARKS:
        if env_name is None:
            known = ", ".join(available_benchmark_names())
            raise ValueError(
                f"Unknown benchmark '{name}'. Available: {known}. "
                "For a custom Gymnasium env, pass --env-name, "
                "--num-inputs, --num-outputs, and --max-episode-steps."
            )

        missing = []
        if num_inputs is None:
            missing.append("--num-inputs")
        if num_outputs is None:
            missing.append("--num-outputs")
        if max_episode_steps is None:
            missing.append("--max-episode-steps")

        if missing:
            raise ValueError(
                "Custom benchmarks require "
                f"{', '.join(missing)} when --env-name is provided."
            )

        return BenchmarkSpec(
            name=key,
            env_name=env_name,
            max_episode_steps=max_episode_steps,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            config_base=config_base,
            config_plastic=config_plastic,
        )

    spec = BENCHMARKS[key]
    return replace(
        spec,
        num_inputs=num_inputs if num_inputs is not None else spec.num_inputs,
        num_outputs=num_outputs if num_outputs is not None else spec.num_outputs,
        env_name=env_name if env_name is not None else spec.env_name,
        max_episode_steps=(
            max_episode_steps
            if max_episode_steps is not None
            else spec.max_episode_steps
        ),
        config_base=config_base,
        config_plastic=config_plastic,
    )


def make_env(benchmark: BenchmarkSpec):
    import gymnasium as gym

    try:
        return gym.make(
            benchmark.env_name,
            max_episode_steps=benchmark.max_episode_steps,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not create Gymnasium env '{benchmark.env_name}' because "
            f"the optional dependency '{exc.name}' is not installed. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc


def validate_benchmark_env(benchmark: BenchmarkSpec) -> None:
    import numpy as np
    from gymnasium import spaces

    env = make_env(benchmark)

    try:
        observation_space = env.observation_space
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise NotImplementedError(
                f"Unsupported observation space: {observation_space}"
            )

        actual_inputs = int(np.prod(observation_space.shape))

        if isinstance(action_space, spaces.Box):
            actual_outputs = int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            actual_outputs = int(action_space.n)
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

        if benchmark.num_inputs != actual_inputs:
            raise ValueError(
                f"{benchmark.env_name} observation dimension is {actual_inputs}, "
                f"but num_inputs={benchmark.num_inputs}"
            )

        if benchmark.num_outputs != actual_outputs:
            raise ValueError(
                f"{benchmark.env_name} action dimension is {actual_outputs}, "
                f"but num_outputs={benchmark.num_outputs}"
            )
    finally:
        env.close()
