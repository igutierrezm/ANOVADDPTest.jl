# Fit the model in a more pleasant way
function anova_bnp_berpoi(
    y::Vector{Int},
    X::Matrix{Int};
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    rho::Float64 = 1.0,
    a::Float64 = 2.0,
    b::Float64 = 4.0,
    a1::Float64 = 2.0,
    b1::Float64 = 4.0,
    alpha0::Float64 = 1.0,
    beta0::Float64 = 1.0,
    lb::Int = minimum(y),
    ub::Int = maximum(y)
)
    # Set data for training
    data0 = BerPoiData(X, y)

    # Set data for prediction
    ngroups = length(data0.Xunique)
    y1 = lb:ub |> x -> repeat(x, inner = ngroups)
    X1 = repeat(vcat(data0.Xunique'...), ub - lb + 1)
    data1 = BerPoiData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    model = BerPoiDDP(rng, N, ngroups; a, b, a1, b1, alpha0, beta0, rho)

    # Train the model
    ch = train(rng, model, data0, data1; iter, warmup);

    # Tidy up the chain for gamma
    gamma_ch = gamma_chain(ch)

    # Compute p(gamma | y)
    group_probs = gamma_posterior(ch);

    # Compute the codebook
    group_codes = gamma_codebook(data0);

    # Compute the effects
    effects1 = simple_effect_probabilities(ch, data0)
    effects2 = interaction_effect_probabilities(ch, data0)

    # Compute p(y0 | y)
    fpost = DataFrame(group = data1.x, y = data1.y, f = mean(ch.f))
    Fpost = DataFrame(group = data1.x, y = data1.y, F = mean(ch.F))

    # Compute the shift functions
    shiftpost = shift_function(Fpost)

    return anova_bnp_fitted(
        group_codes,
        group_probs,
        gamma_ch,
        effects1,
        effects2,
        fpost,
        Fpost,
        shiftpost
    )
end

# begin
#     using Revise
#     using ANOVADDPTest
#     using StatsModels
#     using Statistics
#     using DataFrames
#     using Distributions
#     using Random
# end

# # Simulate sample
# function simulate_sample_berpoi(rng, N)
#     X = rand(rng, 1:2, N, 2);
#     λ = 1.0
#     y = rand(rng, Poisson(λ), N);
#     for i in 1:N
#         ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1)
#     end
#     return y, X
# end

# # Example
# N = 1000;
# rng = MersenneTwister(1);
# y, X = simulate_sample_berpoi(rng, N);
# data1 = anova_bnp_berpoi(y, X)
# effects1(data1)
# effects2(data1)
