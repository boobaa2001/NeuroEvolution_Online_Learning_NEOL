from __future__ import annotations

import configparser
import contextlib
import os
import tempfile
import time

from neol.benchmark import BenchmarkSpec
from neol.paths import OUTPUT_DIR, resolve_repo_path
from neol.settings import DEFAULT_TRAINING, TrainingSettings


PLASTIC_RULES = {"Hebb", "Oja", "BCM"}
REQUIRED_SECTIONS = (
    "NEAT",
    "DefaultGenome",
    "DefaultSpeciesSet",
    "DefaultStagnation",
    "DefaultReproduction",
)


def _read_config_file(path: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def _validate_config_file(path: str) -> None:
    parser = _read_config_file(path)
    _validate_config_parser(parser, path)


def _validate_config_parser(
    parser: configparser.ConfigParser,
    source: str,
) -> None:
    missing = [section for section in REQUIRED_SECTIONS if not parser.has_section(section)]

    if missing:
        missing_text = ", ".join(missing)
        raise configparser.NoSectionError(
            f"{missing_text} in config file: {source}"
        )


def _config_file_matches(
    path: str,
    benchmark: BenchmarkSpec,
    settings: TrainingSettings,
) -> bool:
    if not os.path.isfile(path):
        return False

    try:
        parser = _read_config_file(path)
        _validate_config_parser(parser, path)

        return (
            parser.get("NEAT", "pop_size") == str(settings.pop_size)
            and parser.get("DefaultGenome", "num_inputs") == str(benchmark.num_inputs)
            and parser.get("DefaultGenome", "num_outputs") == str(benchmark.num_outputs)
        )
    except (configparser.Error, OSError):
        return False


@contextlib.contextmanager
def _config_write_lock(path: str):
    lock_path = f"{path}.lock"
    fd = None

    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(0.05)

    try:
        yield
    finally:
        os.close(fd)
        os.remove(lock_path)


def _apply_benchmark_dimensions(
    parser: configparser.ConfigParser,
    benchmark: BenchmarkSpec,
) -> None:
    parser.set("DefaultGenome", "num_inputs", str(benchmark.num_inputs))
    parser.set("DefaultGenome", "num_outputs", str(benchmark.num_outputs))


def _apply_training_settings(
    parser: configparser.ConfigParser,
    settings: TrainingSettings,
) -> None:
    parser.set("NEAT", "pop_size", str(settings.pop_size))


def _runtime_config_path(
    benchmark: BenchmarkSpec,
    tag: str,
    settings: TrainingSettings,
) -> str:
    base_path = resolve_repo_path(benchmark.config_base)

    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Base config file not found: {base_path}")

    parser = _read_config_file(base_path)

    if tag in PLASTIC_RULES:
        override_path = resolve_repo_path(benchmark.config_plastic)

        if not os.path.isfile(override_path):
            raise FileNotFoundError(f"Plastic config file not found: {override_path}")

        parser.read(override_path)

    _validate_config_parser(parser, base_path)
    _apply_benchmark_dimensions(parser, benchmark)
    _apply_training_settings(parser, settings)

    out_dir = os.path.join(OUTPUT_DIR, "_generated_configs")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        (
            f"{benchmark.name}_{tag}_"
            f"{benchmark.num_inputs}in_{benchmark.num_outputs}out_"
            f"pop{settings.pop_size}.cfg"
        ),
    )

    if _config_file_matches(out_path, benchmark, settings):
        return out_path

    with _config_write_lock(out_path):
        if _config_file_matches(out_path, benchmark, settings):
            return out_path

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_path)}.",
            suffix=".tmp",
            dir=out_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as f:
                parser.write(f)

            os.replace(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    return out_path


def get_config_path(
    tag: str,
    benchmark: BenchmarkSpec,
    settings: TrainingSettings = DEFAULT_TRAINING,
) -> str:
    if tag != "NEAT" and tag not in PLASTIC_RULES:
        raise ValueError(f"Unknown rule tag: {tag}")

    config_path = _runtime_config_path(benchmark, tag, settings)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file for tag '{tag}' not found: {config_path}"
        )

    _validate_config_file(config_path)
    return config_path


def make_config(
    tag: str,
    benchmark: BenchmarkSpec,
    settings: TrainingSettings = DEFAULT_TRAINING,
) -> Config:
    from neat.config import Config
    from neat.genome import DefaultGenome
    from neat.reproduction import DefaultReproduction
    from neat.species import DefaultSpeciesSet
    from neat.stagnation import DefaultStagnation

    cfg = Config(
        DefaultGenome,
        DefaultReproduction,
        DefaultSpeciesSet,
        DefaultStagnation,
        get_config_path(tag, benchmark, settings),
    )
    cfg.pop_size = settings.pop_size
    return cfg
