using Revise
using ANOVADDPTest
using Gadfly
using StatsModels
using Statistics
using DataFrames
using Random

# Simulate sample
function simulate_sample_normal(rng, N)
    X = rand(rng, 1:2, N, 2);
    y = randn(rng, N);
    for i in 1:N
        ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1)
    end
    return y, X
end

# Example
N = 1000;
rng = MersenneTwister(1);
y, X = simulate_sample_normal(rng, N);
fit = anova_bnp_normal(y, X);
show(fit[:group_probs]; allrows = true)
plot(
    fit[:posterior_predictive_density],
    x = :y,
    y = :f,
    color = :group,
    Geom.line,
    Scale.color_discrete()
)
