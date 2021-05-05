#include <iostream>
#include <random>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <fstream>
#include <chrono>
#include "rapidcsv.h"

const double delta_uq = 2.0;

double** array2d(int N, int M) {
  double **ptr;

  ptr = new double*[N];
  for(int i = 0; i < N; i++)
    ptr[i] = new double[M];

  return ptr;
}

double*** array3d(int N, int M, int L) {
  double ***ptr;

  ptr = new double**[N];
  for(int i = 0; i < N; i++) {
    ptr[i] = array2d(M, L);
  }

  return ptr;
}


void sample_grid(double *logprobas, const int p, const int q, int* i, int* j,
  std::default_random_engine& gen) {
  // samples from a 2d grid of logarithmic probas[i,j]
  // uses Mironov's Racing algorithm

  std::uniform_real_distribution<double> distrib(0, 1);
  const int Nb = p*q;

  double R = std::numeric_limits<double>::infinity();
  int idx = 0;
  for(int k = 0; k < Nb; k++) {
    double U = distrib(gen);
    double v = log(log(1.0/U)) - logprobas[k];
    if(v < R) {
      R = v;
      idx = k;
    }
  }
  // (i,j) -> i*q + j
  *j = idx % q;
  *i = static_cast<int>(idx / q);
}

std::vector<double> compute_quantiles(const std::vector<double>& X, double* q, int M, double dp_epsilon,
  std::default_random_engine& gen) {
  int N = X.size() - 2;
  int index[M+2], I[N+1];
  auto logphi = array3d(N+1, N+1, M+2);
  auto logalpha = array3d(M+1, N+2, M+1);
  float n[M+2]; // nb of points in bins

  for(int i = 0; i < N+1; i++)
    I[i] = i;

  for(int j = 1; j <= M+1; j++) {
      n[j] = (q[j] - q[j-1])*N;
  }

  index[0] = 0;
  index[M+1] = N;

  for(int i = 0; i < N+2; i++) {
    for(int k = 0; k < M+1; k++) {
      for(int j = 0; j < M+1; j++) {
        logalpha[j][i][k] = -std::numeric_limits<double>::infinity();
      }
    }
  }

  for(int i : I) {
    for(int i2 = i; i2 < N+1; i2++) {
      for(int j = 1; j < M+1; j++) {
        if(fabs(X[i2+1] - X[i2]) > 0) {
          logphi[i][i2][j]= log(X[i2+1] - X[i2]) - dp_epsilon * fabs(i2 - i - n[j]) / (2*delta_uq);
        } else {
          logphi[i][i2][j] = -std::numeric_limits<double>::infinity();
        }
      }
    }
  }

  for(int i : I) {
    logphi[i][N][M+1] = -dp_epsilon*fabs(N-i-n[M+1])/(2*delta_uq);
  }

  for(int i : I) {
    logalpha[1][i][1] = logphi[0][i][1];
  }
  // logalpha(1,i,k) = -inf pour k > 1
  for(int i : I) {
    for(int k = 2; k < M+1; k++) {
      logalpha[1][i][k] = -std::numeric_limits<double>::infinity();
    }
  }

  auto logahat = array2d(M+2, N+1);
  for(int j=2; j<M+1; j++) {
    // compute log(alpha hat(j-1, i))
    for(int i : I) {
      // compute temp max
      double tmp_max = -std::numeric_limits<double>::infinity(), tmp_sum = 0;
      for(int k = 1; k < j; k++) {
        if(logalpha[j-1][i][k] > tmp_max)
          tmp_max = logalpha[j-1][i][k];
      }
      if(tmp_max != -std::numeric_limits<double>::infinity()) {
        for(int k = 1; k < j; k++) {
          tmp_sum += std::exp(logalpha[j-1][i][k] - tmp_max);
        }
      }

      logahat[j-1][i] =  std::log(tmp_sum) + tmp_max;
    }
    for(int i : I) {
      // compute log(alpha(j, i, 1))
      double tmp_max = -std::numeric_limits<double>::infinity(), tmp_sum = 0;
      for(int i2 = 0; i2 < i; i2++) {
         if(logphi[i2][i][j] + logahat[j-1][i2] > tmp_max)
          tmp_max = logphi[i2][i][j] + logahat[j-1][i2];
      }
      if(tmp_max != -std::numeric_limits<double>::infinity()) {
        for(int i2 = 0; i2 < i; i2++) {
          tmp_sum += std::exp(  logphi[i2][i][j] + logahat[j-1][i2] - tmp_max);
        }
      }

      logalpha[j][i][1] =  std::log(tmp_sum) + tmp_max ;

      // compute log(alpha(j, i, k)) for 2 < k <= j
      for(int k = 2; k < j+1; k++) {
         logalpha[j][i][k] = logphi[i][i][j] + logalpha[j-1][i][k-1] - std::log(k);
      }
    }
  }
  // sample (i, k) ∝ α(m, i, k)φ(i, n, m + 1)
  // fill proba values
  auto logprobas = new double[ (N+1)*M ];
  for(int i : I) {
    for(int k = 1; k < M+1; k++) {
      logprobas[i*M + k-1] = logalpha[M][i][k] + logphi[i][N][M+1];
    }
  }

  int sampled_i, sampled_k;
  sample_grid(logprobas, N+1, M, &sampled_i, &sampled_k, gen);
  // index  0 corresponds to k=1
  sampled_k += 1;

  for(int l = 0; l <= sampled_k-1; l++) {
    index[M-l] = sampled_i;
  }

  int j = M-sampled_k;
  delete[] logprobas;

  while(j > 0) {
    auto logprobas = new double[ index[j+1]*M];

    for(int k = 1; k < M+1; k++) {
      for(int i = 0; i < index[j+1]; i++) {
        logprobas[i*M + k-1] = logalpha[j][i][k] + logphi[i][index[j+1]][j+1];
      }
    }

    sample_grid(logprobas, index[j+1], M, &sampled_i, &sampled_k, gen);
    sampled_k += 1;

    for(int l = 0; l <= sampled_k-1; l++) {
      index[j-l] = sampled_i;
    }

    j = j-sampled_k;
    delete[] logprobas;
  }
  std::vector<double> computed_quantiles;
  computed_quantiles.resize(M);
  for(int m = 1; m < M+1; m++) {
    // sample uniformly from [X_i_j, X_(i_j + 1)]
    std::uniform_real_distribution<double> distrib(X[index[m]], X[index[m]+1]);
    computed_quantiles[m-1] = distrib(gen);
  }

  std::sort(computed_quantiles.begin(), computed_quantiles.end());

  for(int i = 0; i < N+1; i++) {
    for(int i2 = 0; i2 < N+1; i2++) {
      delete[] logphi[i][i2];
    }
  }

  for(int j = 0; j < M+1; j++) {
    for(int i = 0; i < N+2; i++) {
      delete[] logalpha[j][i];
    }
  }

  return computed_quantiles;
}

int main(int argc, char* argv[])
{
  if(argc != 5) {
    std::cout << "Usage: ./google_quantile_dp filename epsilon lower_bound upper_bound\n";
    return -1;
  }

  rapidcsv::Document csv(argv[1]);
  double dp_epsilon = atof(argv[2]);
  double lower_bound = atof(argv[3]);
  double upper_bound = atof(argv[4]); 

  std::vector<double> X = csv.GetColumn<double>("Data");
  int N = X.size();

  X.insert(X.begin(), lower_bound);
  X.insert(X.end(), upper_bound);
  std::sort(X.begin(), X.end());

  rapidcsv::Document quantile_query("quantile_query.csv");
  std::vector<double> query = quantile_query.GetColumn<double>("Quantile");

  int M = query.size();
  // std::cout << "M = " << M << " epsilon = " << dp_epsilon << "\n";

  double q[M+2];

  q[0] = 0;
  for(int j = 1; j <= M; j++)
    q[j] = query[j-1];
  q[M+1] = 1;

  std::ofstream out_results("out_quantiles.csv");
  if(!out_results.is_open()) {
    return -1;
  }
  const int num_samples = 1;

  for(int k = 1; k < M+1; k++) {
    if(k != M)
      out_results << q[k] << ",";
    else
      out_results << q[k] << "\n";
  }
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);

  for(int i = 0; i < num_samples; i++) {
    auto quantiles = compute_quantiles(X, q, M, dp_epsilon,gen);

    for(int k = 1; k < M+1; k++) {
      if(k != M)
        out_results << quantiles[k-1] << ",";
      else
        out_results << quantiles[k-1] << "\n";
    }
  }

  out_results.close();
  return 0;
}
