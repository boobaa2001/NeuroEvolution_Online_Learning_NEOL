from dataclasses import dataclass
from typing import Optional


GENS = 100
POP_SIZE = 100
REPEAT_PER_GEN = 1
NUM_RUNS = 1

ETA_HEBB = 0.0025
ETA_OJA = 0.0025
ETA_BCM = 0.0025
THETA_TAU = 0.02

R_SCALE = 0.001
TRACE_LAMBDA = 0.9

CLIP_PREACT = 50.0
CLIP_WEIGHT = 10.0

# True writes plasticity-updated weights back to the genome.
# False keeps plasticity changes temporary during rollout.
WRITE_BACK = True

ENABLE_CHUNK_PARALLEL = True
CHUNK_WORKERS: Optional[int] = None
CHUNK_SIZE = 8


@dataclass(frozen=True)
class TrainingSettings:
    gens: int = GENS
    pop_size: int = POP_SIZE
    repeat_per_gen: int = REPEAT_PER_GEN
    num_runs: int = NUM_RUNS
    write_back: bool = WRITE_BACK
    enable_chunk_parallel: bool = ENABLE_CHUNK_PARALLEL
    chunk_workers: Optional[int] = CHUNK_WORKERS
    chunk_size: int = CHUNK_SIZE


DEFAULT_TRAINING = TrainingSettings()
