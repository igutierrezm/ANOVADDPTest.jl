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
    sumlogfacty::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    G::Int
    function PoissonDDP(
        rng::AbstractRNG,
        N::Int,
        G::Int;
        K0::Int = 1,
        a::Float64 = 2.0,
        b::Float64 = 4.0,
        a1::Float64 = 2.0,
        b1::Float64 = 4.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a, b0 = b)
        a1_post = [a1 * ones(Int, G)]
        b1_post = [b1 * ones(Int, G)]
        sumlogfacty = [zeros(G)]
        gammaprior = Womack(G - 1, rho)
        gamma = ones(Bool, G)
        new(parent, a1, b1, a1_post, b1_post, sumlogfacty, gammaprior, gamma, G)
    end
end

function parent_dpm(model::PoissonDDP)
    model.parent
end

function add_cluster!(model::PoissonDDP)
    @extract model : G a1 b1 a1_post b1_post sumlogfacty
    push!(a1_post, a1 * ones(G))
    push!(b1_post, b1 * ones(G))
    push!(sumlogfacty, zeros(G))
end

function update_suffstats!(model::PoissonDDP, data)
    @extract data : y x
    @extract model : a1 b1 a1_post b1_post sumlogfacty gamma
    d = cluster_labels(model)
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for k in active_clusters(model)
        a1_post[k] .= a1
        b1_post[k] .= b1
        sumlogfacty[k] .= 0.0
    end
    for i = 1:length(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        sumlogfacty[di][zi] += logfactorial(y[i])
        a1_post[di][zi] += y[i]
        b1_post[di][zi] += 1
    end
end

function update_suffstats!(model::PoissonDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract model : a1_post b1_post sumlogfacty gamma
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a1_post[k2][zi] += y[i]
    b1_post[k2][zi] += 1
    sumlogfacty[k2][zi] += logfactorial(y[i])

    # Modify cluster/group k1/zi
    a1_post[k1][zi] -= y[i]
    b1_post[k1][zi] -= 1
    sumlogfacty[k1][zi] -= logfactorial(y[i])
end

function logpredlik(model::PoissonDDP, data, i::Int, k::Int)
    d = cluster_labels(model)
    @extract model : a1_post b1_post gamma
    @extract data : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j] - (d[i] == k) * y[i]
    b1kj = b1_post[k][j] - (d[i] == k)
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i])
end

function logpredlik(model::PoissonDDP, train, predict, i::Int, k::Int)
    @extract model : a1_post b1_post gamma
    @extract predict : y x
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a1_post[k][j]
    b1kj = b1_post[k][j]
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i])
end

function logmglik(model::PoissonDDP, j::Int, k::Int)
    @extract model : a1 b1 a1_post b1_post sumlogfacty
    return (
        a1 * log(b1) - a1_post[k][j] * log(b1_post[k][j]) +
        loggamma(a1_post[k][j]) - loggamma(a1) -
        sumlogfacty[k][j]
    )
end

function update_gamma!(rng::AbstractRNG, model::PoissonDDP, data)
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

function update_hyperpars!(rng::AbstractRNG, model::PoissonDDP, data)
    update_gamma!(rng, model, data)
end

# Tõnu Kollo (tonu.kollo@ut.ee) University of Tartu, Tartu, Estonia
