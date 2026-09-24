data {
  int<lower=1> N;                        // number of double-bongcloud games (324)
  vector[N] elo;                         // observed black_elo (Black's blitz rating)
  real mu_mean;                          // prior: mu ~ normal(mu_mean, mu_sd)
  real<lower=0> mu_sd;
  real<lower=0> sigma_sd;                // prior: sigma ~ half-normal(0, sigma_sd)
  int<lower=0, upper=1> use_likelihood;  // 0 = prior only, 1 = posterior
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(mu_mean, mu_sd);
  sigma ~ normal(0, sigma_sd);           // lower=0 bound makes this half-normal
  if (use_likelihood == 1) {
    elo ~ normal(mu, sigma);
  }
}
generated quantities {
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
  }
}