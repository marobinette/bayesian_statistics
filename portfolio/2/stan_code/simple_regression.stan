data {
    int<lower=1> N;
    vector[N] X;
    vector[N] Y;
    int<lower=0, upper=1> use_likelihood;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu = alpha + beta * X;
}
// model {
//     alpha ~ normal(300000, 100000);  // typical out-of-zone house: roughly $100k to $500k
//     beta ~ normal(0, 50000);         // flood-zone gap: either direction, roughly within ±$100k
//     sigma ~ exponential(1e-5);       // house-to-house spread: mean $100k
//     if (use_likelihood) {
//         Y ~ normal(mu, sigma);
//     }
// }
// swap to log scale priors
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