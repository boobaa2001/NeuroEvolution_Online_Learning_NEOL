from typing import Dict, Tuple

import numpy as np
from neat.nn import FeedForwardNetwork

from neol.settings import (
    CLIP_PREACT,
    CLIP_WEIGHT,
    ETA_BCM,
    ETA_HEBB,
    ETA_OJA,
    R_SCALE,
    THETA_TAU,
    TRACE_LAMBDA,
    WRITE_BACK,
)


def extract_enabled_connection_weights(genome) -> Dict[Tuple[int, int], float]:
    return {
        c.key: float(c.weight)
        for c in genome.connections.values()
        if c.enabled
    }


def apply_connection_weights_to_genome(
    genome,
    weights: Dict[Tuple[int, int], float],
) -> None:
    for key, weight in weights.items():
        if key in genome.connections:
            genome.connections[key].weight = float(weight)


class BaseNet:
    is_plastic = False

    def __init__(self, genome, cfg, write_back=WRITE_BACK):
        net = FeedForwardNetwork.create(genome, cfg)

        self.node_evals = list(net.node_evals)
        self.input_keys = list(cfg.genome_config.input_keys)
        self.output_keys = list(cfg.genome_config.output_keys)
        self.values = {k: 0.0 for k in self.input_keys}
        self.write_back = write_back

    def prepare(self):
        for key in self.values:
            self.values[key] = 0.0

    def set_reward(self, reward):
        pass

    def _set_inputs(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32).reshape(-1)

        if inputs.size != len(self.input_keys):
            raise ValueError(
                f"Network received {inputs.size} inputs, "
                f"but config expects {len(self.input_keys)}"
            )

        for key, value in zip(self.input_keys, inputs):
            self.values[key] = value

        return self.values


class BasePlasticNet(BaseNet):
    is_plastic = True

    def __init__(self, genome, cfg, write_back=WRITE_BACK):
        super().__init__(genome, cfg, write_back=write_back)

        self.w_lookup = {
            c.key: c
            for c in genome.connections.values()
            if c.enabled
        }
        self.runtime_weights: Dict[Tuple[int, int], float] = {
            c.key: float(c.weight)
            for c in genome.connections.values()
            if c.enabled
        }
        self.e_trace: Dict[Tuple[int, int], float] = {
            key: 0.0
            for key in self.runtime_weights
        }
        self._reward_mod = 0.0

    def prepare(self):
        super().prepare()
        self._reward_mod = 0.0

        for key in self.e_trace:
            self.e_trace[key] = 0.0

    def set_reward(self, reward):
        self._reward_mod = reward * R_SCALE

    def _current_weight(self, src, dst, default_w):
        return self.runtime_weights.get((src, dst), default_w)

    def _weighted_sum(self, nid, links):
        return sum(
            self.values.get(src, 0.0) * self._current_weight(src, nid, default_w)
            for src, default_w in links
        )

    def _update_weight(self, src, dst, new_weight):
        new_weight = float(new_weight)
        key = (src, dst)

        self.runtime_weights[key] = new_weight

        if self.write_back and key in self.w_lookup:
            self.w_lookup[key].weight = new_weight

        for i, (nid, act, agg, bias, resp, links) in enumerate(self.node_evals):
            if nid == dst:
                self.node_evals[i] = (
                    nid,
                    act,
                    agg,
                    bias,
                    resp,
                    [(s, new_weight) if s == src else (s, w) for s, w in links],
                )
                break


class StaticNetwork(BaseNet):
    def activate(self, inputs):
        values = self._set_inputs(inputs)

        for nid, act, agg, bias, resp, links in self.node_evals:
            total = bias + resp * sum(values.get(src, 0.0) * w for src, w in links)
            values[nid] = act(np.clip(total, -CLIP_PREACT, CLIP_PREACT))

        return [np.tanh(values.get(key, 0.0)) for key in self.output_keys]


class HebbNetwork(BasePlasticNet):
    def activate(self, inputs):
        values = self._set_inputs(inputs)

        for nid, act, agg, bias, resp, links in self.node_evals:
            total = bias + resp * self._weighted_sum(nid, links)
            y = act(np.clip(total, -CLIP_PREACT, CLIP_PREACT))
            values[nid] = y

            for src, default_w in links:
                x = values.get(src, 0.0)
                key = (src, nid)

                self.e_trace[key] = TRACE_LAMBDA * self.e_trace[key] + x * y

                w = self._current_weight(src, nid, default_w)
                dw = ETA_HEBB * self._reward_mod * self.e_trace[key]
                new_weight = np.clip(w + dw, -CLIP_WEIGHT, CLIP_WEIGHT)

                self._update_weight(src, nid, new_weight)

        return [np.tanh(values.get(key, 0.0)) for key in self.output_keys]


class OjaNetwork(BasePlasticNet):
    def activate(self, inputs):
        values = self._set_inputs(inputs)

        for nid, act, agg, bias, resp, links in self.node_evals:
            total = bias + resp * self._weighted_sum(nid, links)
            y = act(np.clip(total, -CLIP_PREACT, CLIP_PREACT))
            values[nid] = y

            for src, default_w in links:
                x = values.get(src, 0.0)
                key = (src, nid)

                self.e_trace[key] = TRACE_LAMBDA * self.e_trace[key] + x * y

                w = self._current_weight(src, nid, default_w)
                dw = ETA_OJA * self._reward_mod * (
                    y * (x - y * w) + self.e_trace[key]
                )
                new_weight = np.clip(w + dw, -CLIP_WEIGHT, CLIP_WEIGHT)

                self._update_weight(src, nid, new_weight)

        return [np.tanh(values.get(key, 0.0)) for key in self.output_keys]


class BCMNetwork(BasePlasticNet):
    def __init__(self, genome, cfg, write_back=WRITE_BACK):
        super().__init__(genome, cfg, write_back=write_back)
        self.theta: Dict[int, float] = {}

    def prepare(self):
        super().prepare()
        self.theta.clear()

    def activate(self, inputs):
        values = self._set_inputs(inputs)

        for nid, act, agg, bias, resp, links in self.node_evals:
            total = bias + resp * self._weighted_sum(nid, links)
            y = act(np.clip(total, -CLIP_PREACT, CLIP_PREACT))
            values[nid] = y

            theta_j = self.theta.get(nid, 0.0)
            theta_j = (1.0 - THETA_TAU) * theta_j + THETA_TAU * y * y
            self.theta[nid] = theta_j

            for src, default_w in links:
                x = values.get(src, 0.0)

                w = self._current_weight(src, nid, default_w)
                dw = ETA_BCM * self._reward_mod * y * (y - theta_j) * x
                new_weight = np.clip(w + dw, -CLIP_WEIGHT, CLIP_WEIGHT)

                self._update_weight(src, nid, new_weight)

        return [np.tanh(values.get(key, 0.0)) for key in self.output_keys]


RULE_SPECS = {
    "NEAT": StaticNetwork,
    "Hebb": HebbNetwork,
    "Oja": OjaNetwork,
    "BCM": BCMNetwork,
}
