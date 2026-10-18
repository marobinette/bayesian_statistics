
data {
  int<lower=1> N;
  int<lower=2> K;
  matrix[N, K] X;
  vector[N] y;
  real alpha_mean;
  real<lower=0> alpha_sd;
  real<lower=0> beta_sd;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  beta[1] ~ normal(alpha_mean, alpha_sd);
  beta[2:K] ~ normal(0, beta_sd);
  sigma ~ exponential(1);
  y ~ normal_id_glm(X, 0, beta, sigma);
}
