using Revise
using ANOVADDPTest
using Gadfly
using StatsModels
using Statistics
using DataFrames
using Distributions
using Random

# # Simulate sample
# function simulate_sample_normal(rng, N)
#     X = rand(rng, 1:2, N, 2);
#     y = randn(rng, N);
#     for i in 1:N
#         ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1)
#     end
#     return y, X
# end

# Simulate sample
function simulate_sample_poisson(rng, N)
    X = rand(rng, 1:2, N, 2);
    y = rand(rng, Poisson(1.0), N);
    for i in 1:N
        ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1)
    end
    return y, X
end

# Example
N = 1000;
rng = MersenneTwister(1);
y, X = simulate_sample_poisson(rng, N);
fit = anova_bnp_poisson(y, X);
show(group_probs(fit); allrows = true)
show(group_codes(fit); allrows = true)
plot(
    fpost(fit),
    x = :y,
    y = :f,
    color = :group,
    Geom.line,
    Scale.color_discrete()
)
