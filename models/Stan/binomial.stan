//
// This script defines a binomial likelihood GP classification model
// with a squared exponential (RBF) covariance kernel function
//

//
// Author: Trent Henderson, 22 September 2021
//

data {
  int<lower=0> P;             // Number of parameters
  int<lower=0> N;             // Sample size
  int<lower=0, upper=1> y[N]; // Response variable
  matrix[N, P] X;             // Model design matrix

  // Hyperparameters for GP covariance function range and scale

  real m_rho;
  real<lower=0> s_rho;
  real m_alpha;
  real<lower=0> s_alpha;
  real<lower=0> eps;
}

parameters {
  real<lower=0> rho;     // Range parameter in GP covariance fn
    real<lower=0> alpha; // Covariance scale parameter in GP covariance function
  vector[N] eta;
  real beta;
}

transformed parameters {
  vector[N] f;
  {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // Cholesky of GP covariance matrix
    row_vector[N] row_x[N];
    
    // Covariance kernel function

    for (n in 1:N) {
      row_x[n] = to_row_vector(X[n, :]);
    }

    K = cov_exp_quad(row_x, alpha, rho); 

    // Add small jitter along the diagonal for numerical stability

    for (n in 1:N) {
        K[n, n] = K[n, n] + eps;
    }

    // Cholesky of K (lower triangle)

    LK = cholesky_decompose(K); 
  
    f = LK * eta;
  }
}

model {
  
  // Priors

  alpha ~ lognormal(m_alpha, s_alpha);
  rho ~ lognormal(m_rho, s_rho);
  eta ~ std_normal();
  beta ~ std_normal();
 
  // Likelihood

  y ~ bernoulli_logit(beta + f);
}
