# Fit the model in a more pleasant way
function anova_bnp_bernoulli(
    y::Vector{Bool},
    X::Matrix{Int};
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    zeta0::Float64 = 1.0,
    a0::Float64 = 2.0,
    b0::Float64 = 4.0,
    a1::Float64 = 2.0,
    b1::Float64 = 4.0,
    lb::Int = minimum(y),
    ub::Int = maximum(y)
)
    # Set data for training
    data0 = BernoulliData(X, y)

    # Set data for prediction
    G = length(data0.Xunique)
    y1 = lb:ub |> x -> repeat(x, inner = G)
    X1 = repeat(vcat(data0.Xunique'...), ub - lb + 1)
    data1 = BernoulliData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    m = BernoulliDDP(rng, N, G; αa0 = a0, αb0 = b0, a0 = a1, b0 = b1, ζ0 = zeta0)

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
    fpost = DataFrame(group = data1.x, y = data1.y, f = mean(ch.f))

    return anova_bnp_fitted(
        group_codes,
        group_probs,
        effects1,
        effects2,
        fpost
    )
end

# # Example
# N = 1000;
# rng = MersenneTwister(1);
# y, X = simulate_sample_poisson(rng, N);
# data1 = anova_bnp_poisson(y, X);
# group_probs = gamma_posterior(data1)
# show(group_probs; allrows = true)
