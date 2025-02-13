data {
    int<lower=1> N;              // Number of data points
    int<lower=1> I;              // Number of populations
    int<lower=1> X;              // Number of age groups
    int<lower=1> T;              // Number of time points
    array[N] int<lower=1> age;         // Age column
    array[N] int<lower=1> year;        // Year column
    array[N] int<lower=1> population;  // Population column
    vector[N] log_mortality;     // Log-transformed mortality rates
}

parameters {
    real mu_alpha;                 // Global mean for alpha 
    real<lower=0> sigma_alpha;     // Standard deviation of alpha 
    matrix[X, I] alpha_raw;    // Observation-level deviations for age effects 

    real mu_beta;                  // Global mean for beta 
    real<lower=0> sigma_beta;      // Standard deviation of beta 
    matrix[X, I] beta_raw;     // Observation-level deviations for beta effects 

    matrix[T, I] kappa;                 // Temporal trend for each population
    real<lower=0.0001> sigma_kappa;     // Smoothness parameter for kappa
    real<lower=0, upper=1> phi;         // AR(1) parameter (using AR(1) instead of RMD for multi-population)
                                        // inspired by Shi et al.

    real<lower=0.0001> sigma_obs;       // Observation noise
}

transformed parameters {
    matrix[X, I] alpha;  // Age-specific baseline effects 
    matrix[X, I] beta;   // Age-specific sensitivities 

    for (i in 1:I) {
        alpha[, i] = mu_alpha + sigma_alpha * alpha_raw[, i];
        beta[, i] = mu_beta + sigma_beta * beta_raw[, i];
    }
}

model {
    mu_alpha ~ normal(0, 1);
    sigma_alpha ~ normal(0.5, 0.25) T[0, ];
    to_vector(alpha_raw) ~ normal(0, 1);

    mu_beta ~ normal(0, 1);
    sigma_beta ~ normal(0.5, 0.25) T[0, ];
    to_vector(beta_raw) ~ normal(0, 1);

    phi ~ beta(2, 2);       // prior used in Shi et al. 
    sigma_kappa ~ normal(0.5, 0.25) T[0, ];
    sigma_obs ~ normal(0.5, 0.25) T[0, ];

    // Temporal trend priors
    for (i in 1:I) {
        kappa[1, i] ~ normal(-0.5, 0.5);  // Initial value
        for (t in 2:T) {
            kappa[t, i] ~ normal(phi * kappa[t - 1, i], sigma_kappa);
        }
    }

    // Likelihood
    vector[N] mu;
    for (n in 1:N) {
        mu[n] = alpha[age[n], population[n]] +
                beta[age[n], population[n]] * kappa[year[n], population[n]];
    }
    log_mortality ~ normal(mu, sigma_obs);

  }

