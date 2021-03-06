//
// This script defines a Gaussian likelihood GP model
// with an exponentiated quadratic covariance kernel function
//

//
// Author: Trent Henderson, 22 September 2021
//

data {
  int<lower=1> N; // Sample size
  real x[N];      // Predictor variable
  vector[N] y;    // Response variable
}
transformed data {
  real delta = 1e-9;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
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

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();
  eta ~ std_normal();

  y ~ normal(f, sigma);
}
