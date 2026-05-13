# NEOL

NEOL is a small experiment runner for comparing standard NEAT against
reward-modulated online plasticity rules on Gymnasium control tasks.

The current rule set is:

- `NEAT`: static feed-forward NEAT policy
- `Hebb`: reward-modulated Hebbian plasticity
- `Oja`: reward-modulated Oja plasticity
- `BCM`: reward-modulated BCM plasticity

Each run trains all four rules for the selected benchmark and writes curves,
winners, seeds, and generated NEAT config files under `results/`.

## Requirements

- Python 3.10 or newer
- `pip`
- System packages needed by the selected Gymnasium environments

The dependency file installs Gymnasium with Box2D and MuJoCo extras:

```bash
pip install -r requirements.txt
```

If Box2D or MuJoCo installation fails on your platform, start with CartPole
first, then install the optional environment dependencies separately.

## Installation

Clone the repository and install the dependencies from the repository root:

```bash
git clone <repo-url>
cd NEOL

python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run a tiny smoke test that finishes quickly:

```bash
python main.py smoke --env-name CartPole-v1 --num-inputs 4 --num-outputs 2 --max-episode-steps 10 --gens 1 --pop-size 12 --num-runs 1 --chunk-workers 2 --chunk-size 2
```

Run the default CartPole experiment:

```bash
python main.py cartpole
```

Run CartPole with explicit training settings:

```bash
python main.py cartpole --gens 50 --pop-size 100 --num-runs 3
```

List available built-in benchmark names:

```bash
python main.py --list-benchmarks
```

## Built-in Benchmarks

| Name | Gymnasium environment | Inputs | Outputs | Default max steps |
| --- | --- | ---: | ---: | ---: |
| `cartpole` | `CartPole-v1` | 4 | 2 | 500 |
| `lunarlander` | `LunarLander-v3` | 8 | 4 | 1000 |
| `bipedalwalker` | `BipedalWalker-v3` | 24 | 4 | 1600 |
| `hopper` | `Hopper-v5` | 11 | 3 | 1000 |

Aliases such as `cart`, `lunar`, `bipedal`, and `bipedal-walker` are also
accepted.

## Custom Environments

Any Gymnasium environment with a `Box` observation space and either a `Discrete`
or `Box` action space can be used by passing the environment id and dimensions:

```bash
python main.py custom --env-name MountainCarContinuous-v0 --num-inputs 2 --num-outputs 1 --max-episode-steps 999 --gens 50
```

For `Discrete` actions, `num_outputs` must equal the number of discrete actions.
For `Box` actions, `num_outputs` must equal the flattened action dimension.

## Common Options

| Option | Description |
| --- | --- |
| `--gens` | Number of generations. |
| `--pop-size` | NEAT population size. |
| `--repeat-per-gen` | Number of rollout repeats per genome evaluation. |
| `--num-runs` | Number of seeds per rule. |
| `--write-back` / `--no-write-back` | Enable or disable writing plasticity-updated weights back to genomes. |
| `--chunk-parallel` / `--no-chunk-parallel` | Enable or disable parallel chunk evaluation inside each task. |
| `--chunk-workers` | Worker processes per task. Defaults to automatic selection. |
| `--chunk-size` | Genome chunk size sent to each worker. |
| `--config-base` | Base NEAT config template. Defaults to `config_base`. |
| `--config-plastic` | Plastic-rule config override template. Defaults to `config_plastic`. |

Full CLI help is available with:

```bash
python main.py --help
```

## Configuration

NEOL builds runtime NEAT configuration files from:

- `config_base`: base config for static NEAT
- `config_plastic`: override template for plasticity-based rules

At runtime, the selected benchmark dimensions and `--pop-size` are written into
generated config files in `results/_generated_configs/`. These generated files
are derived artifacts and do not need to be committed.

## Outputs

Each benchmark writes to `results/<benchmark-name>/`:

- `<rule>_seed<seed>_curve.csv`: per-generation mean and best fitness
- `<rule>_seed<seed>_best.pkl`: pickled winning genome
- `seeds.txt`: seed list used for the run

Generated runtime config files are stored in `results/_generated_configs/`.

## Reproducibility Notes

The runner seeds Python and NumPy for each task. Gymnasium environments and
third-party physics backends may still have platform-specific behavior, so exact
fitness curves can differ across operating systems, dependency versions, or CPU
architectures.

On Windows, run experiments through `python main.py ...` from the repository
root so multiprocessing workers can import the project correctly.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
