from typing import Sequence

import numpy as np
import pandas as pd


class Distribution:
    """Distribution to sample from"""

    def sample(self, nb_samples: int) -> Sequence:
        pass


class Uniform(Distribution):
    """Uniform distribution"""

    def __init__(self, minimum: float = 0, maximum: float = 1):
        self.min = minimum
        self.max = maximum
        self.name = "Uniform"

    def sample(self, nb_samples: int) -> np.array:
        return np.random.uniform(self.min, self.max, size=nb_samples)


class Normal(Distribution):
    def __init__(self, scale: float = 1, minimum: float = -5, maximum: float = 5):
        self.scale = scale
        self.min = minimum
        self.max = maximum
        self.name = "Normal"

    def sample(self, nb_samples: int) -> np.array:
        result = np.random.normal(scale=self.scale, size=nb_samples)
        result = np.clip(result, self.min, self.max)
        return result


class GoodreadsRatings(Distribution):
    def __init__(self, filename: str):
        self.filename = filename
        self.name = "GoodreadsRatings"
        self.min = 0
        self.max = 5

    def sample(self, nb_samples: int) -> np.array:
        data = pd.read_csv(self.filename, error_bad_lines=False)
        data = data["average_rating"]
        data = data.sample(nb_samples)
        return data


class GoodreadsPages(Distribution):
    def __init__(self, filename: str):
        self.filename = filename
        self.name = "GoodreadsPages"
        self.min = 0
        self.max = 100

    def sample(self, nb_samples: int) -> np.array:
        data = pd.read_csv(self.filename, error_bad_lines=False)
        data = data["  num_pages"] / 100
        data = data.sample(nb_samples)
        return data
