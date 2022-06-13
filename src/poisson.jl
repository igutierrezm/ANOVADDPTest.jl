struct PoissonData
    X::Matrix{Int}
    x::Vector{Int}
    y::Vector{Int}
    Xunique::Vector{Vector{Int}}
    function PoissonData(X::Matrix{Int}, y::Vector{Int})
        x = denserank([X[i, :] for i in 1:size(X, 1)])
        Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
        new(X, x, y, Xunique)
    end
end

PoissonData(x::Vector{Int}, y::Vector{Int}) = PoissonData(x[:, :], y)

struct PoissonDDP <: AbstractDPM
    parent::DPM
    a1::Float64
    b1::Float64
    a1_post::Vector{Vector{Int}}
    b1_post::Vector{Vector{Int}}
    sumlogu::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    G::Int
    function PoissonDDP(
        rng::AbstractRNG,
        N::Int,
        G::Int;
        K0::Int = 1,
        a0::Float64 = 2.0,
        b0::Float64 = 4.0,
        a1::Float64 = 2.0,
        b1::Float64 = 4.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a0, b0 = b0)
        a1_post = [a1 * ones(Int, G)]
        b1_post = [b1 * ones(Int, G)]
        sumlogu = [zeros(G)]
        gammaprior = Womack(G - 1, rho)
        gamma = ones(Bool, G)
        new(parent, a1, b1, a1_post, b1_post, sumlogu, gammaprior, gamma, G)
    end
end

function parent_dpm(m::PoissonDDP)
    m.parent
end

function add_cluster!(m::PoissonDDP)
    @extract m : G a1 b1 a1_post b1_post sumlogu
    push!(a1_post, a1 * ones(G))
    push!(b1_post, b1 * ones(G))
    push!(sumlogu, zeros(G))
end

function update_suffstats!(m::PoissonDDP, data)
    @extract data : y x
    @extract m : a1 b1 a1_post b1_post sumlogu gamma
    d = cluster_labels(m)
    while length(a1_post) < cluster_capacity(m)
        add_cluster!(m)
    end
    for k in active_clusters(m)
        a1_post[k] .= a1
        b1_post[k] .= b1
        sumlogu[k] .= 0.0
    end
    for i = 1:length(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        sumlogu[di][zi] += logfactorial(y[i])
        a1_post[di][zi] += y[i]
        b1_post[di][zi] += 1
    end
end

function update_suffstats!(m::PoissonDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract m : a1 b1 a1_post b1_post sumlogu gamma
    while length(a1_post) < cluster_capacity(m)
        add_cluster!(m)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a1_post[k2][zi] += y[i]
    b1_post[k2][zi] += 1
    sumlogu[k2][zi] += logfactorial(y[i])

    # Modify cluster/group k1/zi
    a1_post[k1][zi] -= y[i]
    b1_post[k1][zi] -= 1
    sumlogu[k1][zi] -= logfactorial(y[i])
end

function logpredlik(m::PoissonDDP, data, i::Int, k::Int)
    d = cluster_labels(m)
    @extract m : a1_post b1_post gamma
    @extract data : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j] - (d[i] == k) * y[i]
    b1kj = b1_post[k][j] - (d[i] == k)
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i])
end

function logpredlik(m::PoissonDDP, train, predict, i::Int, k::Int)
    @extract m : a1_post b1_post gamma
    @extract predict : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j]
    b1kj = b1_post[k][j]
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i])
end

function logmglik(m::PoissonDDP, j::Int, k::Int)
    @extract m : a1 b1 a1_post b1_post sumlogu
    return (
        a1 * log(b1) - a1_post[k][j] * log(b1_post[k][j]) +
        loggamma(a1_post[k][j]) - loggamma(a1) -
        sumlogu[k][j]
    )
end

function update_gamma!(rng::AbstractRNG, m::PoissonDDP, data)
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

function update_hyperpars!(rng::AbstractRNG, m::PoissonDDP, data)
    update_gamma!(rng, m, data)
end

# Tõnu Kollo (tonu.kollo@ut.ee) University of Tartu, Tartu, Estonia
