data {
    int<lower=1> N;
    int<lower=1> q;
    matrix[N, q] X;
    vector[N] Y;
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
    alpha ~ normal(12, 0.5);  
    beta ~ normal(0, 0.5);       
    sigma ~ exponential(3);
    if (use_likelihood) {
        Y ~ normal(mu, sigma);
    }
}
generated quantities {
    array[N] real Y_rep = normal_rng(mu, sigma);
}