#-------------------------------------------
# This script defines GP models for binomial
# data
#-------------------------------------------

#-------------------------------------------
# Author: Trent Henderson, 20 September 2021
#-------------------------------------------

using Random, Distributions, Distances, Turing
import LinearAlgebra

Random.seed!(123) # Fix seed for reproducibility

#---------------------------
# Define covariance function
#---------------------------

# Squared exponential kernel

SEkernel(α, ρ) = α^2 * transform(SEKernel(), invsqrt2/ρ)

#------------------------------
# Define kernel helper function
#------------------------------

function compute_f(kernel, X, η, β = 0, jitter::Float64 = 0.0)

    K = kernelmatrix(kernel, X, obsdim = 1) + LinearAlgebra.I * jitter

    return LinearAlgebra.cholesky(K).L * η .+ β
end

#-----------------
# Define the model
#-----------------

@model function GP(y::Array, X::Array, jitter::Float64 = 1e-6)
    
    # Specify priors

    α ~ LogNormal(0, 1)
    ρ ~ LogNormal(0, 1)
    β ~ Normal(0, 1)
    η ~ filldist(Normal(0, 1), length(y))

    # Kernel operations

    f = compute_f(SEkernel(α, ρ), X, η, β, jitter)
    
    # Draw from the sampling distribution

    y ~ arraydist(Bernoulli.(logistic.(f)))
end
