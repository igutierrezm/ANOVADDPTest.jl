struct NormalData
    X::Matrix{Int}
    x::Vector{Int}
    y::Vector{Float64}
    Xunique::Vector{Vector{Int}}
    function NormalData(X::Matrix{Int}, y::Vector{Float64})
        x = denserank([X[i, :] for i in 1:size(X, 1)])
        Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
        new(X, x, y, Xunique)
    end
end

NormalData(x::Vector{Int}, y::Vector{Float64}) = NormalData(x[:, :], y)

struct NormalDDP <: AbstractDPM
    parent::DPM
    mu0::Float64
    lambda0::Float64
    a0::Float64
    b0::Float64
    mu0_post::Vector{Vector{Float64}}
    lambda0_post::Vector{Vector{Float64}}
    a0_post::Vector{Vector{Float64}}
    b0_post::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    ngroups::Int
    function NormalDDP(
        rng::AbstractRNG,
        N::Int,
        ngroups::Int;
        K0::Int = 5,
        a::Float64 = 2.0,
        b::Float64 = 4.0,
        mu0::Float64 = 0.0,
        lambda0::Float64 = 1.0,
        a0::Float64 = 2.0,
        b0::Float64 = 1.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a, b0 = b)
        mu0_post = [mu0 * ones(ngroups)]
        lambda0_post = [lambda0 * ones(ngroups)]
        a0_post = [a0 * ones(ngroups)]
        b0_post = [b0 * ones(ngroups)]
        gammaprior = Womack(ngroups - 1, rho)
        gamma = ones(Bool, ngroups)
        new(
            parent, mu0, lambda0, a0, b0, mu0_post, lambda0_post, 
            a0_post, b0_post, gammaprior, gamma, ngroups
        )
    end
end

function parent_dpm(model::NormalDDP)
    model.parent
end

function add_cluster!(model::NormalDDP)
    @extract model : ngroups mu0 lambda0 a0 b0 mu0_post lambda0_post a0_post b0_post
    push!(mu0_post, mu0 * ones(ngroups))
    push!(lambda0_post, lambda0 * ones(ngroups))
    push!(a0_post, a0 * ones(ngroups))
    push!(b0_post, b0 * ones(ngroups))
end

function update_suffstats!(model::NormalDDP, data)
    @extract model : mu0 lambda0 a0 b0 mu0_post lambda0_post a0_post b0_post gamma
    @extract data : y x
    d = cluster_labels(model)
    while length(a0_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for k in active_clusters(model)
        mu0_post[k] .= mu0
        lambda0_post[k] .= lambda0
        a0_post[k] .= a0
        b0_post[k] .= b0
    end
    for i = 1:length(y)
        zi = gamma[x[i]] ? x[i] : 1
        di = d[i]
        ld = lambda0_post[di][zi] += 1
        mu = mu0_post[di][zi] = ((ld - 1) * mu0_post[di][zi] + y[i]) / ld
        a0_post[di][zi] += 1 / 2
        b0_post[di][zi] += (ld / (ld - 1)) * (y[i] - mu)^2 / 2
    end
end

function update_suffstats!(model::NormalDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract model : mu0_post lambda0_post a0_post b0_post gamma
    while length(a0_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group di/k2
    a0_post[k2][zi] += 1 / 2
    ld = lambda0_post[k2][zi] += 1
    mu = mu0_post[k2][zi] = ((ld - 1) * mu0_post[k2][zi] + y[i]) / ld
    b0_post[k2][zi] += (ld / (ld - 1)) * (y[i] - mu)^2 / 2

    # Modify cluster/group di/k1
    ld = lambda0_post[k1][zi]
    mu = mu0_post[k1][zi]
    b0_post[k1][zi] -= (ld / (ld - 1)) * (y[i] - mu)^2 / 2
    mu0_post[k1][zi] = (ld * mu0_post[k1][zi] - y[i]) / (ld - 1)
    a0_post[k1][zi] -= 1 / 2
    lambda0_post[k1][zi] -= 1
end

function update_gamma!(rng::AbstractRNG, model::NormalDDP, data)
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

function update_hyperpars!(rng::AbstractRNG, model::NormalDDP, data)
    update_gamma!(rng, model, data)
end

function logpredlik(model::NormalDDP, data, i::Int, k::Int)
    @extract model : mu0_post lambda0_post a0_post b0_post gamma
    @extract data : y x
    d = cluster_labels(model)
    yi = y[i]
    di = d[i]
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    if di == k
        v̄1 = a0_post[k][zi]
        r̄1 = lambda0_post[k][zi]
        ū1 = mu0_post[k][zi]
        s̄1 = b0_post[k][zi]
        v̄0 = v̄1 - 1 / 2
        r̄0 = r̄1 - 1
        ū0 = (r̄1 * ū1 - yi) / r̄0
        s̄0 = s̄1 - (r̄1 / r̄0) * (yi - ū1)^2 / 2
    else
        v̄0 = a0_post[k][zi]
        r̄0 = lambda0_post[k][zi]
        ū0 = mu0_post[k][zi]
        s̄0 = b0_post[k][zi]
        v̄1 = v̄0 + 1 / 2
        r̄1 = r̄0 + 1
        ū1 = (r̄0 * ū0 + yi) / r̄1
        s̄1 = s̄0 + (r̄1 / r̄0) * (yi - ū1)^2 / 2
    end

    # if iszero(r̄0)
    #     return - v̄1 * log(s̄1) + loggamma(v̄1) - 0.5 * log(2π)
    # else
        return (
            v̄0 * log(s̄0) -
            v̄1 * log(s̄1) +
            loggamma(v̄1) -
            loggamma(v̄0) +
            0.5 * log(r̄0 / r̄1) -
            0.5 * log(2π)
        )
    # end
end

function logpredlik(model::NormalDDP, train, predict, i::Int, k::Int)
    @extract model : mu0_post lambda0_post a0_post b0_post gamma
    @extract predict : y x
    yi = y[i]
    zi = iszero(gamma[x[i]]) ? 1 : x[i]
    v̄0 = a0_post[k][zi]
    r̄0 = lambda0_post[k][zi]
    ū0 = mu0_post[k][zi]
    s̄0 = b0_post[k][zi]
    v̄1 = v̄0 + 1 / 2
    r̄1 = r̄0 + 1
    ū1 = (r̄0 * ū0 + yi) / r̄1
    s̄1 = s̄0 + (r̄1 / r̄0) * (yi - ū1)^2 / 2

    # if iszero(r̄0)
    #     return - v̄1 * log(s̄1) + loggamma(v̄1) - 0.5 * log(2π)
    # else
        return (
            v̄0 * log(s̄0) -
            v̄1 * log(s̄1) +
            loggamma(v̄1) -
            loggamma(v̄0) +
            0.5 * log(r̄0 / r̄1) -
            0.5 * log(2π)
        )
    # end
end

function logmglik(model::NormalDDP, j::Int, k::Int)
    @extract model : lambda0 a0 b0 lambda0_post a0_post b0_post
    return(
        a0 * log(b0) -
        a0_post[k][j] * log(b0_post[k][j]) +
        loggamma(a0_post[k][j]) -
        loggamma(a0) +
        0.5 * log(lambda0 / lambda0_post[k][j]) -
        0.5 * (lambda0_post[k][j] - lambda0) * log(2π)
    )
end
