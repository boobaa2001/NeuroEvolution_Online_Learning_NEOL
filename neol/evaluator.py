import multiprocessing as mp
import os
import random

import numpy as np

from neol.benchmark import BenchmarkSpec, make_env
from neol.config import make_config
from neol.networks import RULE_SPECS, extract_enabled_connection_weights
from neol.rollout import rollout
from neol.settings import DEFAULT_TRAINING, TrainingSettings


_WORKER_ENV = None
_WORKER_BENCHMARK = None
_WORKER_CFG = None
_WORKER_NETCLS = None
_WORKER_REPEAT = 1
_WORKER_WRITE_BACK = True


def _init_eval_worker(
    benchmark: BenchmarkSpec,
    tag: str,
    settings: TrainingSettings,
    seed_base: int,
):
    global _WORKER_ENV, _WORKER_BENCHMARK, _WORKER_CFG, _WORKER_NETCLS
    global _WORKER_REPEAT, _WORKER_WRITE_BACK

    worker_seed = seed_base + os.getpid()
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))

    _WORKER_BENCHMARK = benchmark
    _WORKER_ENV = make_env(benchmark)
    _WORKER_CFG = make_config(tag, benchmark, settings)
    _WORKER_NETCLS = RULE_SPECS[tag]
    _WORKER_REPEAT = settings.repeat_per_gen
    _WORKER_WRITE_BACK = settings.write_back


def _score_genome(genome, env, benchmark, net_cls, cfg, repeat_times, write_back):
    scores = []

    for _ in range(repeat_times):
        net = net_cls(genome, cfg, write_back=write_back)
        scores.append(rollout(env, net, benchmark.max_episode_steps))

    fitness = float(np.mean(scores))

    if write_back and getattr(net_cls, "is_plastic", False):
        final_weights = extract_enabled_connection_weights(genome)
    else:
        final_weights = None

    return fitness, final_weights


def _evaluate_chunk(chunk):
    out = []

    for gid, genome in chunk:
        fitness, final_weights = _score_genome(
            genome,
            _WORKER_ENV,
            _WORKER_BENCHMARK,
            _WORKER_NETCLS,
            _WORKER_CFG,
            _WORKER_REPEAT,
            _WORKER_WRITE_BACK,
        )

        out.append((gid, fitness, final_weights))

    return out


class ChunkEvaluator:
    def __init__(
        self,
        benchmark: BenchmarkSpec,
        tag,
        seed,
        workers,
        settings: TrainingSettings = DEFAULT_TRAINING,
    ):
        self.benchmark = benchmark
        self.tag = tag
        self.seed = seed
        self.settings = settings
        self.net_cls = RULE_SPECS[tag]
        self.workers = max(1, workers)
        self.chunk_size = max(1, settings.chunk_size)

        self.cfg = make_config(tag, benchmark, settings)

        self.local_env = None
        self.pool = None

        if self.workers > 1:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(
                processes=self.workers,
                initializer=_init_eval_worker,
                initargs=(
                    benchmark,
                    tag,
                    settings,
                    seed * 100003,
                ),
            )
        else:
            self.local_env = make_env(benchmark)

    def _iter_chunks(self, genomes):
        for i in range(0, len(genomes), self.chunk_size):
            yield genomes[i:i + self.chunk_size]

    def evaluate(self, genomes):
        genomes = list(genomes)

        if self.pool is None:
            result_by_gid = {}

            for gid, genome in genomes:
                fitness, final_weights = _score_genome(
                    genome,
                    self.local_env,
                    self.benchmark,
                    self.net_cls,
                    self.cfg,
                    self.settings.repeat_per_gen,
                    self.settings.write_back,
                )

                result_by_gid[gid] = (fitness, final_weights)

            return result_by_gid

        result_by_gid = {}

        for chunk_result in self.pool.imap_unordered(
            _evaluate_chunk,
            self._iter_chunks(genomes),
            chunksize=1,
        ):
            for gid, fitness, final_weights in chunk_result:
                result_by_gid[gid] = (fitness, final_weights)

        return result_by_gid

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

        if self.local_env is not None:
            self.local_env.close()


def auto_chunk_workers(
    total_tasks,
    settings: TrainingSettings = DEFAULT_TRAINING,
):
    if settings.chunk_workers is not None:
        return max(1, settings.chunk_workers)

    return max(1, mp.cpu_count() // max(1, total_tasks))
