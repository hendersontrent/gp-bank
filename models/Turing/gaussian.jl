#-------------------------------------------
# This script defines GP models for Gaussian
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

SEkernel(D, ϕ, ϵ = 1e-3) = exp.(-D^2 / ϕ) + LinearAlgebra.I * ϵ

#-----------------
# Define the model
#-----------------

@model function GP(y::Array, X::Array, μₘ::Float64 = 0.0, σₛ::Float64 = 1.0, Σfunction = SEkernel)
    
    # Get row size of the input matrix

    N = size(X, 1)
    
    # Compute all pairwise distances across the matrix

    D = pairwise(Distances.Euclidean(), X, dims = 1)
    
    # Specify priors

    μ ~ Normal(μₘ, σₛ)
    σ ~ LogNormal(0, 1)
    ϕ ~ LogNormal(0, 1)
    
    # Define the covariance function

    K = Σfunction(D, ϕ)
    
    # Draw from the sampling distribution

    y ~ MvNormal(μ * ones(N), K + σ * LinearAlgebra.I(N))
end
