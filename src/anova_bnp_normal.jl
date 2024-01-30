struct anova_bnp_fitted
    group_codes::DataFrame
    group_probs::DataFrame
    effects1::DataFrame
    effects2::DataFrame
    fpost::DataFrame
    # densitypost::DataFrame
end

# Fit the model in a more pleasant way
function anova_bnp_normal(
    y::Vector{Float64},
    X::Matrix{Int};
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    n::Int = 50, # taken from ggplot2::geom_density(),
    rho::Float64 = 1.0,
    a::Float64 = 1.0,
    b::Float64 = 1.0,
    mu0::Float64 = 0.0,
    lambda0::Float64 = 1.0,
    a0::Float64 = 2.0,
    b0::Float64 = 1.0,
    lb = minimum(y) - std(y),
    ub = maximum(y) + std(y),
    standardize_y = false
)
    # Standardize the data if requested
    if standardize_y == true
        my0, sy0 = mean_and_std(y)
        lb = (lb - my0) / sy0
        ub = (ub - my0) / sy0
        @. y = (y - my0) / sy0
    end

    # Set data for training
    data0 = NormalData(X, y)

    # Set data for prediction
    ngroups = length(data0.Xunique)
    y1 = range(lb, ub, length = n) |> x -> repeat(x, inner = ngroups)
    X1 = repeat(vcat(data0.Xunique'...), n)
    data1 = NormalData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    model = NormalDDP(rng, N, ngroups; a, b, mu0, lambda0, a0, b0, rho)

    # Train the model
    ch = train(rng, model, data0, data1; iter, warmup);

    # Unroll the standardization
    if standardize_y == true
        @. y = my0 + sy0 * y
        @. ch.f = ch.f ./ sy0
        # @. ch.d = ch.d ./ sy0
        @. data1.y = my0 + sy0 * data1.y
    end

    # Compute p(gamma | y)
    group_probs = gamma_posterior(ch);

    # Compute the codebook
    group_codes = gamma_codebook(data0);

    # Compute the effects
    effects1 = simple_effect_probabilities(ch, data0)
    effects2 = interaction_effect_probabilities(ch, data0)

    # Compute p(y0 | y)
    fpost = DataFrame(group = data1.x, y = data1.y, f = mean(ch.f))

    # # Compute the density
    # densitypost = DataFrame(group = data1.x, y = data1.y, f = mean(ch.d))
    # # Compute the target density
    # densitypost = DataFrame(hcat(ch.d...), :auto)
    # densitypost.group = data1.x
    # densitypost.y = data1.y

    # # Pivot densitypost
    # myparse = u -> parse.(Int, replace.(u, "x" => ""))
    # densitypost =
    #     densitypost |>
    #     x -> stack(x, Not(:group, :y), variable_name = "iter") |>
    #     x -> transform!(x, :iter => myparse => :iter) |>
    #     x -> sort!(x, [:iter, :group, :y])

    return anova_bnp_fitted(
        group_codes,
        group_probs,
        effects1,
        effects2,
        fpost,
        # densitypost
    )
end

group_codes(x::anova_bnp_fitted) = x.group_codes
group_probs(x::anova_bnp_fitted) = x.group_probs
effects1(x::anova_bnp_fitted) = x.effects1
effects2(x::anova_bnp_fitted) = x.effects2
fpost(x::anova_bnp_fitted) = x.fpost
density(x::anova_bnp_fitted) = x.densitypost

# using Revise
# using ANOVADDPTest
# using StatsModels
# using Statistics
# using DataFrames
# using Distributions
# using Random

# # Simulate sample
# function simulate_sample_poisson(rng, N)
#     X = rand(rng, 1:2, N, 2);
#     y = rand(rng, Poisson(1.0), N);
#     for i in 1:N
#         ((X[i, 1] == 2) && (X[i, 2] == 2)) && (y[i] += 1)
#     end
#     return y, X
# end
