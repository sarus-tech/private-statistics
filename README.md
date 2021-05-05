# Private Statistics

This repository is used to test quantiles and quantization algorithms on different distributions and metrics.

To define new objects, simply inherit from `Distribution`, `Metric` or `DPAlgorithm`.

## Distributions
Currently there are 4 implemented distributions:
- Uniform
- Normal
- GoodReadsRatings
- GoodReadsPages

To use the GoodReads datasets, please download the dataset from https://www.kaggle.com/jealousleopard/goodreadsbooks

## Metrics
The implemented metrics are:
- Number of missed points from https://arxiv.org/pdf/2102.08244.pdf
- Quantizing distortion

## Running the tests
To get the results, use either:
- runner.test to get results for one algorithm, metric and distribution
- runner.test_multiple to get results for lists of each.

In order to use the GoogleQuantiles algorithm, please install clang and `run g++ google_quantile.cc -std=c++14 -o google_quantile`