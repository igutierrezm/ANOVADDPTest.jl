using Revise
using Gadfly
using ANOVADDPTest
using DataFrames
using Distributions
using Random

# Here, I simulate data
rng = MersenneTwister(1); # seed
N = 1000; # number of observations
D = 2; # number of covariates
X = rand(rng, 1:3, N, D);
y = randn(rng, N); # clearly, all groups are equal
for i in 1:N
    ((X[i, 1] != 1) && (X[i, 2] == 1)) && (y[i] += 1.0)
end

# Organize the data
data = NormalData(X, y);

# Create some data for prediction (useful for density plots)
pred = expandgrid([1:3 for _ in 1:D]...,  range(-2.0, stop = 2.0, length = 50));
pred = NormalData(hcat(pred[1:D]...), pred[D+1]);

# Create the model
model = NormalDDP(rng, N, length(data.Xunique));

# Fit the model
ch = train(rng, model, data, pred);

# Compute the effects
simple_effect_probabilities(ch, data)
interaction_effect_probabilities(ch, data)
