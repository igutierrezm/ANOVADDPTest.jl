struct BerPoiData
    X::Matrix{Int}
    x::Vector{Int}
    y::Vector{Int}
    Xunique::Vector{Vector{Int}}
    function BerPoiData(X::Matrix{Int}, y::Vector{Int})
        x = denserank([X[i, :] for i in 1:size(X, 1)])
        Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
        new(X, x, y, Xunique)
    end
end

BerPoiData(x::Vector{Int}, y::Vector{Int}) = BerPoiData(x[:, :], y)

struct BerPoiDDP <: AbstractDPM
    parent::DPM
    a1::Float64 # λ ~ Gamma(a1, b1) (λ from the berpoi distribution)
    b1::Float64 # λ ~ Gamma(a1, b1) (λ from the berpoi distribution)
    alpha0::Float64 # α ~ Beta(α0, β0) (α from the berpoi distribution)
    beta0::Float64 # α ~ Beta(α0, β0) (α from the berpoi distribution)
    a1_post::Vector{Vector{Int}}
    b1_post::Vector{Vector{Int}}
    alpha0_post::Vector{Vector{Float64}}
    beta0_post::Vector{Vector{Float64}}
    sumlogfactystar::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    ngroups::Int
    zberpoi::Vector{Bool}
    alphaberpoi::Vector{Vector{Float64}}
    lambdaberpoi::Vector{Vector{Float64}}
    function BerPoiDDP(
        rng::AbstractRNG,
        N::Int,
        ngroups::Int;
        K0::Int = 1,
        a::Float64 = 2.0,
        b::Float64 = 4.0,
        alpha0::Float64 = 1.0,
        beta0::Float64 = 1.0,
        a1::Float64 = 2.0,
        b1::Float64 = 4.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a, b0 = b)
        a1_post = [a1 * ones(Int, ngroups)]
        b1_post = [b1 * ones(Int, ngroups)]
        alpha0_post = [alpha0 * ones(Int, ngroups)]
        beta0_post = [beta0 * ones(Int, ngroups)]
        sumlogfactystar = [zeros(ngroups)]
        gammaprior = Womack(ngroups - 1, rho)
        gamma = ones(Bool, ngroups)
        zberpoi = zeros(Bool, N)
        alphaberpoi = [0.5 * ones(Int, ngroups)]
        lambdaberpoi = [ones(Int, ngroups)]
        new(
            parent, a1, b1, alpha0, beta0, a1_post, b1_post,
            alpha0_post, beta0_post, sumlogfactystar,
            gammaprior, gamma, ngroups, zberpoi, alphaberpoi, lambdaberpoi
        )
    end
end

function parent_dpm(model::BerPoiDDP)
    model.parent
end

function add_cluster!(model::BerPoiDDP)
    @extract model : ngroups a1 b1 alpha0 beta0
    @extract model : a1_post b1_post alpha0_post beta0_post
    @extract model : sumlogfactystar alphaberpoi lambdaberpoi
    push!(a1_post, a1 * ones(ngroups))
    push!(b1_post, b1 * ones(ngroups))
    push!(alpha0_post, alpha0 * ones(ngroups))
    push!(beta0_post, beta0 * ones(ngroups))
    push!(alphaberpoi, 0.5 * ones(ngroups))
    push!(lambdaberpoi, ones(ngroups))
    push!(sumlogfactystar, zeros(ngroups))
end

function update_suffstats!(model::BerPoiDDP, data)
    @extract data : y x
    @extract model : a1 b1 alpha0 beta0
    @extract model : a1_post b1_post alpha0_post beta0_post sumlogfactystar gamma zberpoi
    d = cluster_labels(model)
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for k in active_clusters(model)
        a1_post[k] .= a1
        b1_post[k] .= b1
        alpha0_post[k] .= alpha0
        beta0_post[k] .= beta0
        sumlogfactystar[k] .= 0.0
    end
    for i in eachindex(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        sumlogfactystar[di][zi] += logfactorial(y[i] - zberpoi[i])
        a1_post[di][zi] += y[i]
        b1_post[di][zi] += 1
        alpha0_post[di][zi] += zberpoi[i]
        beta0_post[di][zi] += 1 - zberpoi[i]
    end
end

function update_suffstats!(model::BerPoiDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract model : a1_post b1_post alpha0_post beta0_post
    @extract model : sumlogfactystar gamma zberpoi
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a1_post[k2][zi] += y[i]
    b1_post[k2][zi] += 1
    alpha0_post[k2][zi] += zberpoi[i]
    beta0_post[k2][zi] += 1 - zberpoi[i]
    sumlogfactystar[k2][zi] += logfactorial(y[i] - zberpoi[i])

    # Modify cluster/group k1/zi
    a1_post[k1][zi] -= y[i]
    b1_post[k1][zi] -= 1
    alpha0_post[k1][zi] -= zberpoi[i]
    beta0_post[k1][zi] -= 1 - zberpoi[i]
    sumlogfactystar[k1][zi] -= logfactorial(y[i] - zberpoi[i])
end

function logpredlik(model::BerPoiDDP, data, i::Int, k::Int)
    d = cluster_labels(model)
    @extract model : a1_post b1_post gamma zberpoi
    @extract data : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j] - (d[i] == k) * y[i]
    b1kj = b1_post[k][j] - (d[i] == k)
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i] - zberpoi[i])
end

function logpredlik(model::BerPoiDDP, train, predict, i::Int, k::Int)
    @extract model : a1_post b1_post gamma alphaberpoi
    @extract predict : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j]
    b1kj = b1_post[k][j]
    # Compute P(znew = 1 | y, ...)
    # znew = rand(Bernoulli(alphaberpoi[k][j]))
    # return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i] - znew)
    dist = NegativeBinomial(a1kj, b1kj / (b1kj + 1))
    return log(
        (0 + alphaberpoi[k][j]) * pdf(dist, y[i] - 1) +
        (1 - alphaberpoi[k][j]) * pdf(dist, y[i])
    )
end

function logmglik(model::BerPoiDDP, j::Int, k::Int)
    @extract model : a1 b1 a1_post b1_post sumlogfactystar
    return (
        a1 * log(b1) - a1_post[k][j] * log(b1_post[k][j]) +
        loggamma(a1_post[k][j]) - loggamma(a1) -
        sumlogfactystar[k][j]
    )
end

function update_gamma!(rng::AbstractRNG, model::BerPoiDDP, data)
    @extract model : gammaprior gamma
    A = active_clusters(model)

    # Resample gamma[g], given the other gamma's
    for g = 2:length(gamma)
        # log-odds (numerator)
        gamma[g] = 1
        update_suffstats!(model, data)
        log_num = logpdf(gammaprior, gamma[2:end])
        for k ∈ A, j ∈ (1, g)
            log_num += logmglik(model, j, k)
        end

        # log-odds (denominator)
        gamma[g] = 0
        update_suffstats!(model, data)
        log_den = logpdf(gammaprior, gamma[2:end])
        for k ∈ A, j ∈ (1)
            log_den += logmglik(model, j, k)
        end

        # log-odds and new gamma[g]
        log_odds = log_num - log_den
        gamma[g] = rand(rng) <= 1 / (1 + exp(-log_odds))
    end
end

function update_alphaberpoi!(rng::AbstractRNG, model::BerPoiDDP, data)
    @extract model : ngroups a1_post b1_post alphaberpoi
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for g in 1:ngroups, k in active_clusters(model)
        alphaberpoi[k][g] = rand(rng, Beta(a1_post[k][g], b1_post[k][g]))
    end
end

function update_lambdapoi!(rng::AbstractRNG, model::BerPoiDDP, data)
    @extract model : ngroups alpha0_post beta0_post lambdaberpoi
    while length(alpha0_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for g in 1:ngroups, k in active_clusters(model)
        lambdaberpoi[k][g] = rand(rng, Gamma(alpha0_post[k][g], beta0_post[k][g]))
    end
end

function update_zberpoi!(rng::AbstractRNG, model::BerPoiDDP, data)
    @extract model : ngroups alpha0_post zberpoi lambdaberpoi alphaberpoi
    @extract data : y x
    d = cluster_labels(model)
    for i in eachindex(z)
        k = d[i]
        ak = alphaberpoi[k]
        lk = lambdaberpoi[k]
        pz = ak * y[i] / (ak * y[i] + (1 - ak) * lk)
        zberpoi[i] = rand(Bernoulli(pz))
    end
    return nothing
end

function update_hyperpars!(rng::AbstractRNG, model::BerPoiDDP, data)
    update_gamma!(rng, model, data)
    update_lambdapoi!(rng, model, data)
    update_alphaberpoi!(rng, model, data)
    update_zberpoi!(rng, model, data)
end
