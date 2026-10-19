
data {
  int<lower=1> N;
  int<lower=2> K;
  matrix[N, K] X;
  real alpha_mean;
  real<lower=0> alpha_sd;
  real<lower=0> beta_sd;
}
generated quantities {
  vector[K] beta;
  real<lower=0> sigma = exponential_rng(1);
  beta[1] = normal_rng(alpha_mean, alpha_sd);
  for (k in 2:K) {
    beta[k] = normal_rng(0, beta_sd);
  }
  array[N] real y_rep = normal_rng(X * beta, sigma);
}
