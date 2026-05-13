#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from neol.benchmark import available_benchmark_names, build_benchmark
from neol.runner import run_experiment
from neol.settings import DEFAULT_TRAINING, TrainingSettings


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _add_bool_argument(parser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=f"Disable {name.replace('-', ' ')}.",
    )
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NEOL on a Gymnasium benchmark.",
    )
    train = parser.add_argument_group("training")
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="cartpole",
        help=(
            "Benchmark name. Use a built-in name or any custom name with "
            "--env-name."
        ),
    )
    parser.add_argument(
        "--num-inputs",
        type=_positive_int,
        help="Override observation/input dimension for the selected benchmark.",
    )
    parser.add_argument(
        "--num-outputs",
        type=_positive_int,
        help="Override policy output/action dimension for the selected benchmark.",
    )
    parser.add_argument(
        "--env-name",
        help=(
            "Override or define the Gymnasium env id, for example "
            "MountainCar-v0."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=_positive_int,
        help="Override Gymnasium max_episode_steps.",
    )
    parser.add_argument(
        "--config-base",
        default="config_base",
        help="Base NEAT config template.",
    )
    parser.add_argument(
        "--config-plastic",
        default="config_plastic",
        help="Plastic-rule config override template.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List built-in benchmark names and exit.",
    )
    train.add_argument(
        "--gens",
        type=_positive_int,
        default=DEFAULT_TRAINING.gens,
        help=f"Number of generations. Default: {DEFAULT_TRAINING.gens}.",
    )
    train.add_argument(
        "--pop-size",
        type=_positive_int,
        default=DEFAULT_TRAINING.pop_size,
        help=f"NEAT population size. Default: {DEFAULT_TRAINING.pop_size}.",
    )
    train.add_argument(
        "--repeat-per-gen",
        type=_positive_int,
        default=DEFAULT_TRAINING.repeat_per_gen,
        help=(
            "Rollout repeats per genome evaluation. "
            f"Default: {DEFAULT_TRAINING.repeat_per_gen}."
        ),
    )
    train.add_argument(
        "--num-runs",
        type=_positive_int,
        default=DEFAULT_TRAINING.num_runs,
        help=f"Number of seeds per rule. Default: {DEFAULT_TRAINING.num_runs}.",
    )
    _add_bool_argument(
        train,
        "write-back",
        DEFAULT_TRAINING.write_back,
        (
            "Write plasticity-updated weights back to genomes. "
            f"Default: {DEFAULT_TRAINING.write_back}."
        ),
    )
    _add_bool_argument(
        train,
        "chunk-parallel",
        DEFAULT_TRAINING.enable_chunk_parallel,
        (
            "Enable chunked parallel genome evaluation within each task. "
            f"Default: {DEFAULT_TRAINING.enable_chunk_parallel}."
        ),
    )
    train.add_argument(
        "--chunk-workers",
        type=_positive_int,
        default=DEFAULT_TRAINING.chunk_workers,
        help="Worker processes per task for chunk evaluation. Default: auto.",
    )
    train.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=DEFAULT_TRAINING.chunk_size,
        help=f"Genome chunk size for worker dispatch. Default: {DEFAULT_TRAINING.chunk_size}.",
    )
    return parser.parse_args()


def build_training_settings(args) -> TrainingSettings:
    return TrainingSettings(
        gens=args.gens,
        pop_size=args.pop_size,
        repeat_per_gen=args.repeat_per_gen,
        num_runs=args.num_runs,
        write_back=args.write_back,
        enable_chunk_parallel=args.chunk_parallel,
        chunk_workers=args.chunk_workers,
        chunk_size=args.chunk_size,
    )


def main():
    args = parse_args()

    if args.list_benchmarks:
        print("\n".join(available_benchmark_names()))
        return

    try:
        benchmark = build_benchmark(
            args.benchmark,
            num_inputs=args.num_inputs,
            num_outputs=args.num_outputs,
            env_name=args.env_name,
            max_episode_steps=args.max_episode_steps,
            config_base=args.config_base,
            config_plastic=args.config_plastic,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    settings = build_training_settings(args)

    print(
        "Running "
        f"{benchmark.name} ({benchmark.env_name}) "
        f"with inputs={benchmark.num_inputs}, outputs={benchmark.num_outputs}, "
        f"gens={settings.gens}, pop_size={settings.pop_size}, "
        f"num_runs={settings.num_runs}",
        flush=True,
    )
    run_experiment(benchmark, settings)


if __name__ == "__main__":
    main()
