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

# Fit the model in a more pleasant way
function anova_bnp_normal(
    y::Vector{Float64},
    X::Matrix{Int};
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    n::Int = 50, # taken from ggplot2::geom_density(),
    zeta0::Float64 = 1.0,
    a0::Float64 = 1.0,
    b0::Float64 = 1.0,
    v0::Float64 = 2.0,
    r0::Float64 = 1.0,
    u0::Float64 = 0.0,
    s0::Float64 = 1.0,
)
    # Set data for training
    data0 = NormalData(X, y)

    # Set data for prediction
    G = length(data0.Xunique)
    lb = minimum(y) - 0.5 * std(y)
    ub = maximum(y) + 0.5 * std(y)
    y1 = range(lb, ub, length = n) |> x -> repeat(x, inner = G)
    X1 = repeat(vcat(data0.Xunique'...), n)
    data1 = NormalData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    m = NormalDDP(rng, N, G; a0, b0, v0, r0, u0, s0, ζ0 = zeta0)

    # Train the model
    ch = train(rng, m, data0, data1; iter, warmup);

    # Compute p(γ | y)
    gamma_probs = gamma_posterior(ch);

    # Compute the codebook
    gamma_codes = gamma_codebook(data0);

    # Compute the effects
    effects1 = simple_effect_probabilities(ch, data0)
    effects2 = interaction_effect_probabilities(ch, data0)

    # Compute p(y0 | y)
    df = DataFrame(x = data1.x, y = data1.y, f = mean(ch.f))

    return df, gamma_probs, gamma_codes, effects1, effects2
end

# Example
N = 1000;
rng = MersenneTwister(1);
y, X = simulate_sample_normal(rng, N);
df, gamma_probs, gamma_codes, effects1, effects2 = anova_bnp_normal(y, X);
show(effects2; allrows = true)
plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())
