from typing import List, Optional, Sequence

import numpy as np


class Metric:
    """Metric to evaluate quality of algorithm"""

    def fit(self, data: Sequence, *args, **kwargs) -> None:
        """Compute result without DP as baseline"""
        pass

    def test(self, data: Sequence, alg_output: List) -> float:
        """Compare alg_output to fitted baseline"""
        pass


class QuantilesMissedPoints(Metric):
    """From  https://arxiv.org/pdf/2102.08244.pdf"""

    def __init__(self, nb_quantiles: int):
        self.nb_quantiles = nb_quantiles
        self.name = "QuantilesMissedPoints"

    def fit(
        self, data: Sequence, quantiles: Optional[Sequence] = None, *args, **kwargs
    ) -> None:
        if quantiles:
            self.outputs = np.quantile(data, quantiles)
        else:
            quantiles = [
                (m + 1) / (self.nb_quantiles + 1) for m in range(self.nb_quantiles)
            ]
            self.outputs = np.quantile(data, quantiles)

    def test(self, data: Sequence, alg_output: List) -> float:
        return sum(
            [
                np.abs(np.searchsorted(data, output) - np.searchsorted(data, dp_output))
                for output, dp_output in zip(self.outputs, alg_output)
            ]
        )


class Distortion(Metric):
    def __init__(self, nb_quantizers: int):
        self.nb_quantizers = nb_quantizers
        self.name = "Distortion"

    def test(self, data: Sequence, alg_output: List) -> float:
        return sum([np.min([np.abs(x - y) for y in alg_output]) ** 2 for x in data])
