import subprocess
from typing import List

import algorithms
import distributions
import metrics
import numpy as np
import pandas as pd
from runner import test

distribution = distributions.Uniform()
metric = metrics.QuantilesMissedPoints(9)

repo = "algorithms/google_quantiles"
algorithm = algorithms.GoogleQuantiles(repo, 9)

print(
    test(
        metric,
        algorithm,
        distribution,
        nb_samples=1000,
        nb_trials=5,
        epsilon=1,
        delta=1e-4,
    )
)

algorithm = algorithms.LaplaceQuantiles(9)
print(
    test(
        metric,
        algorithm,
        distribution,
        nb_samples=1000,
        nb_trials=5,
        epsilon=1,
        delta=1e-4,
    )
)
