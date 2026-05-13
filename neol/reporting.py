import numpy as np
from neat.reporting import BaseReporter


class GenReporter(BaseReporter):
    def __init__(self, seed, tag):
        self.seed = seed
        self.tag = tag
        self.gen = 0
        self.means = []
        self.bests = []

    def post_evaluate(self, cfg, pop, species, best):
        fitnesses = [g.fitness for g in pop.values()]
        mean_fit = np.mean(fitnesses)
        best_fit = np.max(fitnesses)

        self.means.append(mean_fit)
        self.bests.append(best_fit)

        print(
            f"[{self.tag}|seed={self.seed}] "
            f"Gen {self.gen} mean={mean_fit:.1f}, best={best_fit:.1f}",
            flush=True,
        )

        self.gen += 1

