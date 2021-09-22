//
// This script defines a Gaussian likelihood GP regression model
// with a squared exponential (RBF) covariance kernel function
//

//
// Author: Trent Henderson, 22 September 2021
//

data {
    int<lower=0> P; // Number of parameters
    int<lower=0> N; // Sample size
    vector[N] y;    // Response variable
    matrix[N, P] X; // Model design matrix
    
    // Hyperparameters for GP covariance function range and scale

    real m_rho;
    real<lower=0> s_rho;
    real m_alpha;
    real<lower=0> s_alpha;
    real m_sigma;
    real<lower=0> s_sigma;
}

transformed data {

    // GP mean function

    vector[N] mu = rep_vector(0, N);
}

parameters {
    real<lower=0> rho;   // Range parameter in GP covariance fn
    real<lower=0> alpha; // Covariance scale parameter in GP covariance function
    real<lower=0> sigma;   // Standard deviation
}

model {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // Cholesky of GP covariance matrix

    // Piors

    rho ~ lognormal(m_rho, s_rho);  // GP covariance function range parameter
    alpha ~ lognormal(m_alpha, s_alpha);  // GP covariance function scale parameter
    sigma ~ lognormal(m_sigma, s_sigma);  // Standard deviation
   
    // Covariance kernel function

    K = cov_exp_quad(to_array_1d(X), alpha, rho); 
    
    // Add small jitter along the diagonal for numerical stability

    for (n in 1:N) {
        K[n, n] = K[n, n] + sigma^2;
    }
        
    // Cholesky of K (lower triangle)

    LK = cholesky_decompose(K);

    // Likelihood

    y ~ multi_normal_cholesky(mu, LK);
}
