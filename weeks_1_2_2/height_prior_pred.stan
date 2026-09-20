
data {
  int<lower=1> N;                        // measurements per simulated dataset
}
generated quantities {
  real mu = normal_rng(160, 10);         // prior on mean height
  real<lower=0> sigma2 = uniform_rng(0, 5000);   // uniform prior on the variance
  array[N] real y_sim;
  for (i in 1:N)
    y_sim[i] = normal_rng(mu, sqrt(sigma2));      // likelihood takes the sd
}
