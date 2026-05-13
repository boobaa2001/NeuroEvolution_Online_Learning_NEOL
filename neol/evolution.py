import os
import pickle
import random

import numpy as np
import pandas as pd
from neat.population import Population
from neat.reporting import ReporterSet

from neol.config import make_config
from neol.evaluator import ChunkEvaluator, auto_chunk_workers
from neol.networks import apply_connection_weights_to_genome
from neol.paths import result_dir
from neol.reporting import GenReporter


def run_task(args):
    tag, seed, total_tasks, benchmark, settings = args

    random.seed(seed)
    np.random.seed(seed)

    cfg = make_config(tag, benchmark, settings)
    pop = Population(cfg)

    reporter = GenReporter(seed, tag)
    pop.reporters = ReporterSet()
    pop.add_reporter(reporter)

    if settings.enable_chunk_parallel:
        workers = auto_chunk_workers(total_tasks, settings)
    else:
        workers = 1

    evaluator = ChunkEvaluator(benchmark, tag, seed, workers, settings)

    def eval_multi(genomes, _cfg):
        result_by_gid = evaluator.evaluate(genomes)

        for gid, genome in genomes:
            fitness, final_weights = result_by_gid[gid]
            genome.fitness = fitness

            if settings.write_back and final_weights is not None:
                apply_connection_weights_to_genome(genome, final_weights)

    try:
        winner = pop.run(eval_multi, settings.gens)
    finally:
        evaluator.close()

    out_dir = result_dir(benchmark.name)

    with open(os.path.join(out_dir, f"{tag}_seed{seed}_best.pkl"), "wb") as f:
        pickle.dump(winner, f)

    curve_df = pd.DataFrame({
        "generation": np.arange(len(reporter.means)),
        "mean": reporter.means,
        "best": reporter.bests,
    })

    curve_df.to_csv(
        os.path.join(out_dir, f"{tag}_seed{seed}_curve.csv"),
        index=False,
    )

    return tag, seed
