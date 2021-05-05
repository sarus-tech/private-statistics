import operator
import os
import subprocess
from typing import List, Optional, Tuple

import numpy as np


class DPAlgorithm:
    """DP algorithm to test"""

    def fit(
        self, datafile: str, epsilon: float, delta: float
    ) -> Tuple[List[float], List[float]]:
        """Given dataset and privacy parameters, compute quantiles
        Return ([quantiles], [quantiles values])"""
        pass


class AlgFromScript(DPAlgorithm):
    """save data as csv, call script, get stdout and return it"""

    def __init__(self, args: Optional[List[str]] = None):  # add location ?
        if args == None:
            self.args = []
        else:
            self.args = args

    def fit(
        self, datafile: str, epsilon: float, delta: float
    ) -> Tuple[List[float], List[float]]:
        """Fit algorithm to data stored in datafile"""
        args = self.args + [
            "--data",
            datafile,
            "--epsilon",
            str(epsilon),
            "--delta",
            str(delta),
        ]
        completed_process = subprocess.run(args, capture_process=True)
        return completed_process.stdout


class GoogleQuantiles(DPAlgorithm):
    """run Joan's implementation of https://arxiv.org/pdf/2102.08244.pdf
    Arguments:
    - repository: path to folder containing the C++ file
    - nb_quantiles"""

    def __init__(
        self, repository: str, nb_quantiles: int, box: Tuple[float, float] = (0, 1)
    ):
        self.repository = repository
        self.nb_quantiles = nb_quantiles
        self.box = box
        self.name = "GoogleQuantiles"

    def fit(
        self, datafile: str, epsilon: float, delta: float
    ) -> Tuple[List[float], List[float]]:
        """Fit algorithm to data stored in datafile"""
        cwd = os.getcwd()
        data = np.loadtxt(datafile)
        os.chdir(self.repository)
        np.savetxt(datafile, data, delimiter="\n", fmt="%s", header="Data", comments="")

        quantiles = [
            (m + 1) / (self.nb_quantiles + 1) for m in range(self.nb_quantiles)
        ]
        np.savetxt(
            "quantile_query.csv",
            quantiles,
            delimiter="\n",
            fmt="%s",
            header="Quantile",
            comments="",
        )

        subprocess.call(
            [
                "./google_quantile",
                datafile,
                str(epsilon),
                str(self.box[0]),
                str(self.box[1]),
            ]
        )
        output = np.loadtxt("out_quantiles.csv", delimiter=",")[1]
        # print(output)
        os.chdir(cwd)
        return quantiles, output


class LaplaceHistogramQuantiles(DPAlgorithm):
    def __init__(self, nb_quantiles: int, box: Tuple[float, float] = (0, 1)):
        self.nb_quantiles = nb_quantiles
        self.box = box
        self.name = "LaplaceHistogramQuantiles"

    def fit(self, datafile, epsilon, delta) -> Tuple[List[float], List[float]]:
        nb_bins = self.nb_quantiles + 1
        noise = (self.box[1] - self.box[0]) / epsilon * nb_bins
        data = np.loadtxt(datafile)
        hist = np.histogram(data, nb_bins, self.box)[0]
        hist = hist + np.random.laplace(0, noise, hist.shape)  # clip negative counts

        cumhist = np.cumsum(hist)
        qs = cumhist / cumhist[-1]
        quantiles = {
            min(1, max(0, qs[n - 1])): self.box[0]
            + (self.box[1] - self.box[0]) * n / nb_bins
            for n in range(1, nb_bins)
        }
        # print(quantiles)
        return list(quantiles.keys()), list(quantiles.values())


class LaplaceQuantiles(DPAlgorithm):
    def __init__(self, nb_quantiles: int, box: Tuple[float, float] = (0, 1)):
        self.nb_quantiles = nb_quantiles
        self.box = box
        self.name = "LaplaceQuantiles"

    def fit(self, datafile, epsilon, delta) -> Tuple[List[float], List[float]]:
        data = np.loadtxt(datafile)
        noise = (self.box[1] - self.box[0]) / epsilon * self.nb_quantiles
        quantiles = [
            q / (self.nb_quantiles + 1) for q in range(1, self.nb_quantiles + 1)
        ]
        qs = np.array([np.quantile(data, q) for q in quantiles])
        qs += np.random.laplace(0, noise * (self.box[1] - self.box[0]), qs.shape)
        # print(qs)
        return quantiles, qs.tolist()


class LaplaceAdaptiveQuantiles(DPAlgorithm):
    def __init__(self, nb_quantiles: int, box: Tuple[float, float] = (0, 1)):
        self.nb_quantiles = nb_quantiles
        self.box = box
        self.name = "LaplaceAdaptiveQuantiles"

    def fit(self, datafile, epsilon, delta) -> Tuple[List[float], List[float]]:
        data = np.loadtxt(datafile)
        quantiles = {0: self.box[0], 1: self.box[1]}
        noise = (self.box[1] - self.box[0]) / epsilon * self.nb_quantiles
        for i in range(self.nb_quantiles):
            a, b = self._max_error_volume(quantiles)
            x0 = (quantiles[b] + quantiles[a]) / 2
            q = self._get_quantile(data, x0, noise)
            quantiles = self._add_quantile(quantiles, q, x0, a, b)
        return list(quantiles.keys()), list(quantiles.values())

    def _max_error_volume(self, qs):
        qs_items = sorted(qs.items(), key=operator.itemgetter(1))
        d = {
            (i[0], j[0]): self._error_volume(i, j)
            for i, j in zip(qs_items[:-1], qs_items[1:])
        }
        return max(d.items(), key=operator.itemgetter(1))[0]

    def _error_volume(self, i, j):
        return (j[0] - i[0]) * (j[1] - i[1])

    def _get_quantile(self, X, x0, noise):
        """Given dataset, value and noise, report noisy quantile estimate"""
        n = X.shape[0]
        s = np.searchsorted(X, x0, "right")
        below = s + np.random.laplace(0, noise)
        over = n - s + np.random.laplace(0, noise)
        q = below / (over + below)
        return q

    def _add_quantile(self, quantiles, q, x0, a, b):
        q = max(0, min(1, q))  # clip to box
        quantiles[q] = x0
        if q < a or q > b:  # reorder if needed
            quantiles = {
                i: j
                for i, j in zip(sorted(quantiles.keys()), sorted(quantiles.values()))
            }
        return quantiles
