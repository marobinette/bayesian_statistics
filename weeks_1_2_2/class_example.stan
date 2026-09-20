
data {
  int<lower=0> N;
  int<lower=0, upper=N> K;
}
parameters {
  real<lower=0, upper=1> rho;
}
model {
  rho ~ beta(1.5, 1.5);
  K ~ binomial(N, rho);
}
