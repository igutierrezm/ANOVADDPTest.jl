# Fit the model in a more pleasant way
function anova_bnp_bernoulli(
    y::Vector{Bool},
    X::Matrix{Int};
    iter::Int = 4000, # taken from rstan::stan()
    warmup::Int = 2000, # taken from rstan::stan()
    seed::Int = 1, # taken from rstan::stan()
    rho::Float64 = 1.0,
    a::Float64 = 2.0,
    b::Float64 = 4.0,
    a2::Float64 = 2.0,
    b2::Float64 = 4.0,
    lb::Int = 0,
    ub::Int = 1
)
    # Set data for training
    data0 = BernoulliData(X, y)

    # Set data for prediction
    ngroups = length(data0.Xunique)
    y1 = Vector{Bool}(lb:ub |> x -> repeat(x, inner = ngroups))
    X1 = repeat(vcat(data0.Xunique'...), ub - lb + 1)
    data1 = BernoulliData(X1, y1)

    # Initialize the model
    N = length(y)
    rng = MersenneTwister(seed)
    model = BernoulliDDP(rng, N, ngroups; a, b, a2, b2, rho)

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

# # Example
# N = 1000;
# rng = MersenneTwister(1);
# y, X = simulate_sample_bernoulli(rng, N);
# data1 = anova_bnp_bernoulli(y, X);
# group_probs = gamma_posterior(data1)
# show(group_probs; allrows = true)
