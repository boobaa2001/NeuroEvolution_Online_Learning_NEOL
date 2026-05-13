import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from neol.benchmark import BenchmarkSpec, validate_benchmark_env
from neol.paths import result_dir
from neol.settings import DEFAULT_TRAINING, TrainingSettings


def run_experiment(
    benchmark: BenchmarkSpec,
    settings: TrainingSettings = DEFAULT_TRAINING,
):
    from neol.evolution import run_task
    from neol.networks import RULE_SPECS

    mp.freeze_support()
    validate_benchmark_env(benchmark)

    out_dir = result_dir(benchmark.name)
    with open(os.path.join(out_dir, "seeds.txt"), "w") as f:
        f.write("\n".join(str(s) for s in range(settings.num_runs)))

    tasks = [
        (tag, seed, max(1, len(RULE_SPECS) * settings.num_runs), benchmark, settings)
        for tag in RULE_SPECS
        for seed in range(settings.num_runs)
    ]

    if len(tasks) == 1:
        tag, seed = run_task(tasks[0])
        print(f"[{tag}|seed={seed}] finished.", flush=True)
        return

    mp.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(tasks))) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]

        for future in as_completed(futures):
            tag, seed = future.result()
            print(f"[{tag}|seed={seed}] finished.", flush=True)
