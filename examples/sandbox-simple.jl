using Revise
using Gadfly
using ANOVADDPTest
using DataFrames
using Distributions
using Random

rng = MersenneTwister(1);
N = 1000;
D = 2
X = rand(rng, 1:2, N, D);
y = randn(rng, N);
for i in 1:N
    if X[i, 1] == 2 && X[i, 2] == 1
        y[i] += 1
    end
end
data = NormalData(X, y);
pred = expandgrid([1:2 for _ in 1:D]...,  range(-2.0, stop = 2.0, length = 50));
pred = NormalData(hcat(pred[1:D]...), pred[D+1]);
m = NormalDDP(rng, N, length(data.Xunique));
ch = train(rng, m, data, pred);
simple_effect_probabilities(ch, data)
interaction_effect_probabilities(ch, data)
