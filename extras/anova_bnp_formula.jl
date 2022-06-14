using Revise
using ANOVADDPTest
using StatsModels
using Statistics
using DataFrames
using Random

# Setup
rng = MersenneTwister(1);
N = 1000;

# Simulate sample
function simulate_sample(rng, N)
    X = rand(rng, 1:2, N, 2);
    y = randn(rng, N);
    for i in 1:N
        ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1.0)
    end
    return DataFrame(y = y, x1 = X[:, 1], x2 = X[:, 2])
end

data = simulate_sample(rng, N);
formula = @formula(y ~ x1 + x2)

function anova_bnp_normal(
    formula,
    data::DataFrame;
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    n::Int = 50, # taken from ggplot2::geom_density()
)
    # Extract X and y from the formula and the data
    f = apply_schema(formula, schema(formula, data))
    y0, X0 = modelcols(f, data)

    # Set data for training
    data0 = NormalData(X0, y0)

    # Set data for prediction
    ngroups = length(data0.Xunique)
    lb = minimum(y0) - 1.5 * std(y0)
    ub = maximum(y0) + 1.5 * std(y0)
    y1 = range(lb, ub, length = n) |> x -> repeat(x, inner = ngroups)
    X1 = repeat(vcat(data0.Xunique'...), n)
    data1 = NormalData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    model = NormalDDP(rng, N, ngroups)

    # Train the model
    ch = train(rng, model, data0, data1; iter, warmup);
    return ch
end
data1 = anova_bnp_normal(formula, data);
