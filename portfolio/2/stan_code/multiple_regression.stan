data {
    int<lower=1> N;
    int<lower=1> q;
    matrix[N, q] X;
    vector[N] Y;
    real alpha_mean;
    real<lower=0> alpha_sd;
    real beta_sd;
    int<lower=0, upper=1> use_likelihood;
}
parameters {
    real alpha;
    vector[q] beta;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    for (i in 1:N) {
        mu[i] = alpha;
        for (j in 1:q) {
            mu[i] += beta[j] * X[i, j];
        }
    }
}
model {
    alpha ~ normal(alpha_mean, alpha_sd);  
    beta ~ normal(0, beta_sd);       
    sigma ~ exponential(3);
    if (use_likelihood) {
        Y ~ normal(mu, sigma);
    }
}
generated quantities {
    array[N] real Y_rep = normal_rng(mu, sigma);
}