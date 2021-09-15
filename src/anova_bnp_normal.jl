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
    group_probs = gamma_posterior(ch);

    # Compute the codebook
    group_codes = gamma_codebook(data0);

    # Compute the effects
    effects1 = simple_effect_probabilities(ch, data0)
    effects2 = interaction_effect_probabilities(ch, data0)

    # Compute p(y0 | y)
    df = DataFrame(
        group = data1.x,
        y = data1.y,
        f = mean(ch.f)
    )
    return Dict(
        :group_codes => group_codes,
        :group_probs => group_probs,
        :effects_1st_order => effects1,
        :effects_1st_order => effects2,
        :posterior_predictive_density => df,
    )
end
