//
// This script defines a binomial likelihood GP classification model
// with an exponentiated quadratic covariance kernel function
//

//
// Author: Trent Henderson, 24 September 2021
//

data {
  int<lower=1> N1;
  real x1[N1];
  int<lower=0, upper=1> y1[N1];
  int<lower=1> N2;
  real x2[N2];
}

transformed data {
  real delta = 1e-9;
  int<lower=1> N = N1 + N2;
  real x[N];
  for (n1 in 1:N1) x[n1] = x1[n1];
  for (n2 in 1:N2) x[N1 + n2] = x2[n2];
}

parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real a;
  vector[N] eta;
}

transformed parameters {

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
}

model {
  
  // Priors

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  a ~ std_normal();
  eta ~ std_normal();
  
  // Likelihood

  y1 ~ bernoulli_logit(a + f[1:N1]);
}

generated quantities {
  int y2[N2];
  
  for (n2 in 1:N2)
    y2[n2] = bernoulli_logit_rng(a + f[N1 + n2]);
}