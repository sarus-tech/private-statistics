from typing import List

import numpy as np
from algorithms import DPAlgorithm
from distributions import Distribution
from metrics import Metric


def test(
    metric: Metric,
    algorithm: DPAlgorithm,
    distribution: Distribution,
    nb_samples: int,
    nb_trials: int,
    epsilon: float,
    delta: float,
    datafile: str = "data.csv",
) -> List[float]:
    """test an algorithm on a given distribution using metric

    Arguments:
    nb_samples: number of datapoints to sample from distribution
    nb_trials: number of times the pipeline should be run
    epsilon, delta: privacy parameters
    datafile: path to file in which the sampled data is saved

    Returns:
    List of size nb_trials"""
    result = []
    for _ in range(nb_trials):
        data = distribution.sample(nb_samples)
        np.savetxt(datafile, data, delimiter=",", fmt="%s")
        quantiles, alg_output = algorithm.fit(datafile, epsilon, delta)
        metric.fit(data, quantiles)
        result.append(metric.test(data, alg_output))

    return result


def test_multiple(
    metrics: List[Metric],
    algorithms: List[DPAlgorithm],
    distributions: List[Distribution],
    nb_samples: int,
    nb_trials: int,
    epsilon: float,
    delta: float,
    datafile: str = "data.csv",
) -> np.array:
    """test algorithms on given distributions using metrics

    Arguments:
    nb_samples: number of datapoints to sample from distribution
    nb_trials: number of times the pipeline should be run
    epsilon, delta: privacy parameters
    datafile: path to file in which the sampled data is saved

    Returns:
    Array of shape (len(metrics), len(algorithms), len(distributions), nb_trials)"""
    result = np.zeros((len(metrics), len(algorithms), len(distributions), nb_trials))
    for k, distrib in enumerate(distributions):
        for l in range(nb_trials):
            data = distrib.sample(nb_samples)
            np.savetxt(datafile, data, delimiter=",", fmt="%s")
            for j, algorithm in enumerate(algorithms):
                algorithm.box = (distrib.min, distrib.max)
                quantiles, alg_output = algorithm.fit(datafile, epsilon, delta)
                for i, metric in enumerate(metrics):
                    metric.fit(data, quantiles)
                    result[i, j, k, l] = metric.test(data, alg_output)

    return result
