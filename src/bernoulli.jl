struct BernoulliData
    X::Matrix{Int}
    x::Vector{Int}
    y::Vector{Bool}
    Xunique::Vector{Vector{Int}}
end

function BernoulliData(x::Vector{Int}, y::Vector{Bool})
    X = x[:, :]
    Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
    BernoulliData(X, x, y, Xunique)
end

function BernoulliData(X::Matrix{Int}, y::Vector{Bool})
    x = denserank([X[i, :] for i in 1:size(X, 1)])
    Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
    BernoulliData(X, x, y, Xunique)
end

function length(data::BernoulliData)
    length(data.y)
end

struct BernoulliDDP <: AbstractDPM
    parent::DPM
    a0::Float64
    b0::Float64
    a1::Vector{Vector{Float64}}
    b1::Vector{Vector{Float64}}
    πgamma::Vector{Float64}
    gamma::Vector{Bool}
    G::Int
    function BernoulliDDP(
            rng::AbstractRNG,
            N::Int,
            G::Int;
            K0::Int = 1,
            αa0::Float64 = 2.0,
            αb0::Float64 = 4.0,
            a0::Float64 = 2.0,
            b0::Float64 = 4.0
        )
        parent = DPM(rng, N; K0, a0 = αa0, b0 = αb0)
        a1 = [a0 * ones(G)]
        b1 = [b0 * ones(G)]
        πgamma = ones(G) / G
        gamma = ones(Bool, G)
        new(parent, a0, b0, a1, b1, πgamma, gamma, G)
    end
end

function parent_dpm(m::BernoulliDDP)
    m.parent
end

function add_cluster!(m::BernoulliDDP)
    @extract m : a0 b0 a1 b1 G
    push!(a1, a0 * ones(Int, G))
    push!(b1, b0 * ones(Int, G))
end

function update_suffstats!(m::BernoulliDDP, data)
    @extract data : y x
    @extract m : a0 b0 a1 b1 gamma
    d = cluster_labels(m)
    while length(a1) < cluster_capacity(m)
        add_cluster!(m)
    end
    for k in active_clusters(m)
        a1[k] .= a0
        b1[k] .= b0
    end
    for i = 1:length(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        a1[di][zi] += y[i]
        b1[di][zi] += 1 - y[i]
    end
end

function update_suffstats!(m::BernoulliDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract m : a0 b0 a1 b1 gamma
    while length(a1) < cluster_capacity(m)
        add_cluster!(m)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a1[k2][zi] += y[i]
    b1[k2][zi] += 1 - y[i]

    # Modify cluster/group k1/zi
    a1[k1][zi] -= y[i]
    b1[k1][zi] -= 1 - y[i]
end

function logpredlik(m::BernoulliDDP, data, i::Int, k::Int)
    @extract data : y x
    @extract m : a1 b1 gamma
    d = cluster_labels(m)
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1[k][j] - (d[i] == k) * y[i]
    b1kj = b1[k][j] - (d[i] == k) * (1 - y[i])
    if y[i]
        return log(a1kj / (a1kj + b1kj))
    else
        return log(b1kj / (a1kj + b1kj))
    end
end

function logpredlik(m::BernoulliDDP, train, predict, i::Int, k::Int)
    @extract predict : y x
    @extract m : a1 b1 gamma
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1[k][j]
    b1kj = b1[k][j]
    if y[i] == 1
        return log(a1kj / (a1kj + b1kj))
    else
        return log(b1kj / (a1kj + b1kj))
    end
end

function logmglik(m::BernoulliDDP, j::Int, k::Int)
    @extract m : a0 b0 a1 b1
    return logbeta(a1[k][j], b1[k][j]) - logbeta(a0, b0)
end

function update_gamma!(rng::AbstractRNG, m::BernoulliDDP, data)
    @extract m : πgamma gamma
    A = active_clusters(m)

    # Resample gamma[g], given the other gamma's
    for g = 2:length(gamma)
        # log-odds (numerator)
        gamma[g] = 1
        update_suffstats!(m, data)
        log_num = log(πgamma[sum(gamma)])
        for k ∈ A, j ∈ (1, g)
            log_num += logmglik(m, j, k)
        end

        # log-odds (denominator)
        gamma[g] = 0
        update_suffstats!(m, data)
        log_den = log(πgamma[sum(gamma)])
        for k ∈ A, j ∈ (1)
            log_den += logmglik(m, j, k)
        end

        # log-odds and new gamma[g]
        log_odds = log_num - log_den
        gamma[g] = rand(rng) <= 1 / (1 + exp(-log_odds))
    end
end

function update_hyperpars!(rng::AbstractRNG, m::BernoulliDDP, data)
    update_gamma!(rng, m, data)
end
