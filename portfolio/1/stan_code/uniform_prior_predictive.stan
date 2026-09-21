data {
  int<lower=1> N;          // dataset size; set to 5068 to match the observed data
  // uniform priors
  real mu_lo;              // mu ~ Uniform(mu_lo, mu_hi)   <- center prior
  real mu_hi;
  real<lower=0> sigma_hi;  // sigma ~ Uniform(0, sigma_hi) <- spread prior
}

generated quantities {
  // one draw of the two dials, per simulated dataset
  real mu = uniform_rng(mu_lo, mu_hi); // uniform_rng takes the min and max, not the mean and sd
  real<lower=0> sigma = uniform_rng(0, sigma_hi);

  // one full simulated dataset of N ratings from those dials
  array[N] real elo_sim;
  for (i in 1:N)
    elo_sim[i] = normal_rng(mu, sigma);   // normal_rng takes the sd, not the variance
}
