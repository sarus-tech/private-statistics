import subprocess
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from algorithms import (GoogleQuantiles, LaplaceAdaptiveQuantiles,
                        LaplaceHistogramQuantiles, LaplaceQuantiles)
from distributions import GoodreadsPages, GoodreadsRatings, Normal, Uniform
from metrics import Distortion, QuantilesMissedPoints
from runner import test, test_multiple

NB_QUANTILES = 9

goodreads_csv = "books2.csv"
distributions = [
    Uniform(),
    Normal(),
    GoodreadsPages(goodreads_csv),
    GoodreadsRatings(goodreads_csv),
]
metrics = [QuantilesMissedPoints(NB_QUANTILES), Distortion(NB_QUANTILES)]

repo = "google_quantiles"
algorithms = [
    LaplaceQuantiles(NB_QUANTILES),
    GoogleQuantiles(repo, NB_QUANTILES),
    LaplaceHistogramQuantiles(NB_QUANTILES),
    LaplaceAdaptiveQuantiles(NB_QUANTILES),
]

results = test_multiple(
    metrics,
    algorithms,
    distributions,
    nb_samples=1000,
    nb_trials=1,
    epsilon=1,
    delta=1e-4,
)

df = pd.DataFrame(columns=["Distribution", "Metric", "Algorithm", "Value"])
for row1, metric in zip(results, metrics):
    for row2, algorithm in zip(row1, algorithms):
        for row3, distribution in zip(row2, distributions):
            for x in row3:
                df.loc[len(df)] = [distribution.name, algorithm.name, metric.name, x]

print(df)
plot = sns.scatterplot(
    x="Distribution", y="Value", hue="Algorithm", style="Metric", data=df
)
plot.set(yscale="log")
plt.show()
