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
length(data::NormalData) = length(data.y)

struct NormalDDP <: AbstractDPM
    parent::DPM
    v0::Float64
    r0::Float64
    u0::Float64
    s0::Float64
    v1::Vector{Vector{Float64}}
    r1::Vector{Vector{Float64}}
    u1::Vector{Vector{Float64}}
    s1::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    G::Int
    function NormalDDP(
        rng::AbstractRNG,
        N::Int,
        G::Int;
        K0::Int = 5,
        a0::Float64 = 2.0,
        b0::Float64 = 4.0,
        v0::Float64 = 2.0,
        r0::Float64 = 1.0,
        u0::Float64 = 0.0,
        s0::Float64 = 1.0,
        ζ0::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0, b0)
        v1 = [v0 * ones(G)]
        r1 = [r0 * ones(G)]
        u1 = [u0 * ones(G)]
        s1 = [s0 * ones(G)]
        gammaprior = Womack(G - 1, ζ0)
        gamma = ones(Bool, G)
        new(parent, v0, r0, u0, s0, v1, r1, u1, s1, gammaprior, gamma, G)
    end
end

function parent_dpm(m::NormalDDP)
    m.parent
end

function add_cluster!(m::NormalDDP)
    @extract m : G v0 r0 u0 s0 v1 r1 u1 s1
    push!(v1, v0 * ones(G))
    push!(r1, r0 * ones(G))
    push!(u1, u0 * ones(G))
    push!(s1, s0 * ones(G))
end

function update_suffstats!(m::NormalDDP, data)
    @extract m : v0 r0 u0 s0 v1 r1 u1 s1 gamma
    @extract data : y x
    d = cluster_labels(m)
    while length(v1) < cluster_capacity(m)
        add_cluster!(m)
    end
    for k in active_clusters(m)
        v1[k] .= v0
        r1[k] .= r0
        u1[k] .= u0
        s1[k] .= s0
    end
    for i = 1:length(y)
        zi = gamma[x[i]] ? x[i] : 1
        di = d[i]
        v1[di][zi] += 1
        rm = r1[di][zi] += 1
        um = u1[di][zi] = ((rm - 1) * u1[di][zi] + y[i]) / rm
        s1[di][zi] += (rm / (rm - 1)) * (y[i] - um)^2
    end
end

function update_suffstats!(m::NormalDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract m : v1 r1 u1 s1 gamma
    while length(v1) < cluster_capacity(m)
        add_cluster!(m)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group di/k2
    v1[k2][zi] += 1
    rm = r1[k2][zi] += 1
    um = u1[k2][zi] = ((rm - 1) * u1[k2][zi] + y[i]) / rm
    s1[k2][zi] += (rm / (rm - 1)) * (y[i] - um)^2

    # Modify cluster/group di/k1
    rm = r1[k1][zi]
    um = u1[k1][zi]
    s1[k1][zi] -= (rm / (rm - 1)) * (y[i] - um)^2
    u1[k1][zi]  = (rm * u1[k1][zi] - y[i]) / (rm - 1)
    v1[k1][zi] -= 1
    r1[k1][zi] -= 1
end

function update_gamma!(rng::AbstractRNG, m::NormalDDP, data)
    @extract m : gammaprior gamma
    A = active_clusters(m)

    # Resample gamma[g], given the other gamma's
    for g = 2:length(gamma)
        # log-odds (numerator)
        gamma[g] = 1
        update_suffstats!(m, data)
        log_num = logpdf(gammaprior, gamma[2:end])
        for k ∈ A, j ∈ (1, g)
            log_num += logmglik(m, j, k)
        end

        # log-odds (denominator)
        gamma[g] = 0
        update_suffstats!(m, data)
        log_den = logpdf(gammaprior, gamma[2:end])
        for k ∈ A, j ∈ (1)
            log_den += logmglik(m, j, k)
        end

        # log-odds and new gamma[g]
        log_odds = log_num - log_den
        gamma[g] = rand(rng) <= 1 / (1 + exp(-log_odds))
    end
end

function update_hyperpars!(rng::AbstractRNG, m::NormalDDP, data)
    update_gamma!(rng, m, data)
end

function logpredlik(m::NormalDDP, data, i::Int, k::Int)
    @extract m : v1 r1 u1 s1 gamma
    @extract data : y x
    d = cluster_labels(m)
    yi = y[i]
    di = d[i]
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    if di == k
        v̄1 = v1[k][zi]
        r̄1 = r1[k][zi]
        ū1 = u1[k][zi]
        s̄1 = s1[k][zi]
        v̄0 = v̄1 - 1
        r̄0 = r̄1 - 1
        ū0 = (r̄1 * ū1 - yi) / r̄0
        s̄0 = s̄1 - (r̄1 / r̄0) * (yi - ū1)^2
    else
        v̄0 = v1[k][zi]
        r̄0 = r1[k][zi]
        ū0 = u1[k][zi]
        s̄0 = s1[k][zi]
        v̄1 = v̄0 + 1
        r̄1 = r̄0 + 1
        ū1 = (r̄0 * ū0 + yi) / r̄1
        s̄1 = s̄0 + (r̄1 / r̄0) * (yi - ū1)^2
    end

    if iszero(r̄0)
        return - 0.5v̄1 * log(s̄1) + loggamma(v̄1 / 2) - 0.5 * log(π)
    else
        return (
            0.5v̄0 * log(s̄0) -
            0.5v̄1 * log(s̄1) +
            loggamma(v̄1 / 2) -
            loggamma(v̄0 / 2) +
            0.5 * log(r̄0 / r̄1) -
            0.5 * log(π)
        )
    end
end

function logpredlik(m::NormalDDP, train, predict, i::Int, k::Int)
    @extract m : v1 r1 u1 s1 gamma
    @extract predict : y x
    yi = y[i]
    zi = iszero(gamma[x[i]]) ? 1 : x[i]
    v̄0 = v1[k][zi]
    r̄0 = r1[k][zi]
    ū0 = u1[k][zi]
    s̄0 = s1[k][zi]
    v̄1 = v̄0 + 1
    r̄1 = r̄0 + 1
    ū1 = (r̄0 * ū0 + yi) / r̄1
    s̄1 = s̄0 + (r̄1 / r̄0) * (yi - ū1)^2

    if iszero(r̄0)
        return - 0.5v̄1 * log(s̄1) + loggamma(v̄1 / 2) - 0.5 * log(π)
    else
        return (
            0.5v̄0 * log(s̄0) -
            0.5v̄1 * log(s̄1) +
            loggamma(v̄1 / 2) -
            loggamma(v̄0 / 2) +
            0.5 * log(r̄0 / r̄1) -
            0.5 * log(π)
        )
    end
end

function logmglik(m::NormalDDP, j::Int, k::Int)
    @extract m : v0 v1 r0 r1 s0 s1
    return(
        0.5v0 * log(s0) -
        0.5v1[k][j] * log(s1[k][j]) +
        loggamma(v1[k][j] / 2) -
        loggamma(v0 / 2) +
        0.5 * log(r0 / r1[k][j]) -
        0.5 * log(π) * (r1[k][j] - r0)
    )
end
