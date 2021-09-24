//
// This script defines a binomial likelihood GP classification model
// with an exponentiated quadratic covariance kernel function
//

//
// Author: Trent Henderson, 24 September 2021
//

data {
  int<lower=1> N;             // Sample size
  real x[N];                  // Predictor variable
  int<lower=0, upper=1> y[N]; // Response variable
}

transformed data {
  real delta = 1e-9;
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real a;
  vector[N] eta;
}

model {

  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x, alpha, rho);

    // Add small number to matrix diagonal for numerical stability
    
    for (n in 1:N)
      K[n, n] = K[n, n] + delta;

    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
  
  // Priors

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  a ~ std_normal();
  eta ~ std_normal();
  
  // Likelihood

  y ~ bernoulli_logit(a + f);
}