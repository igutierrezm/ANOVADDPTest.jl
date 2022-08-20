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
    a0ϵ::Float64
    b0ϵ::Float64
    a0λ::Float64
    b0λ::Float64
    a1ϵ::Vector{Vector{Float64}}
    b1ϵ::Vector{Vector{Float64}}
    a1λ::Vector{Vector{Float64}}
    b1λ::Vector{Vector{Float64}}
    sumlogfactyu::Vector{Vector{Float64}}
    u::Vector{Bool}
    gammaprior::Womack
    gamma::Vector{Bool}
    ngroups::Int
    function PoissonDDP(
        rng::AbstractRNG,
        N::Int,
        ngroups::Int;
        K0::Int = 1,
        a0α::Float64 = 2.0,
        b0α::Float64 = 4.0,
        a0ϵ::Float64 = 2.0,
        b0ϵ::Float64 = 4.0,
        a0λ::Float64 = 2.0,
        b0λ::Float64 = 4.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a0α, b0 = b0α)
        a1ϵ = [a0ϵ * ones(Int, ngroups)]
        b1ϵ = [b0ϵ * ones(Int, ngroups)]
        a1λ = [a0λ * ones(Int, ngroups)]
        b1λ = [b0λ * ones(Int, ngroups)]
        sumlogfactyu = [zeros(ngroups)]
        u = zeros(N)
        gammaprior = Womack(ngroups - 1, rho)
        gamma = ones(Bool, ngroups)
        new(parent, a0ϵ, b0ϵ, a0λ, b0λ, a1ϵ, b1ϵ, a1λ, b1λ, sumlogfactyu, u, gammaprior, gamma, ngroups)
    end
end

function parent_dpm(model::BerPoiDDP)
    model.parent
end

function add_cluster!(model::BerPoiDDP)
    @extract model : ngroups a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu
    push!(a1ϵ, a0ϵ * ones(ngroups))
    push!(b1ϵ, b0ϵ * ones(ngroups))
    push!(a1λ, a0λ * ones(ngroups))
    push!(b1λ, b0λ * ones(ngroups))
    push!(sumlogfactyu, zeros(ngroups))
end

function update_suffstats!(model::BerPoiDDP, data)
    @extract data : y x
    @extract model : a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu u gamma
    d = cluster_labels(model)
    while length(a1_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for k in active_clusters(model)
        a1ϵ[k] .= a0ϵ
        b1ϵ[k] .= b0ϵ
        a1λ[k] .= a0λ
        b1λ[k] .= b0λ
        sumlogfactyu[k] .= 0.0
    end
    for i = 1:length(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        a1λ[di][zi] += y[i] - u[i]
        b1λ[di][zi] += 1
        a1ϵ[di][zi] += u[i]
        b1ϵ[di][zi] += 1 - u[i]
        sumlogfactyu[di][zi] += logfactorial(y[i] - u[i])
    end
end

function update_suffstats!(model::BerPoiDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract model : a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu u gamma
    while length(a1λ) < cluster_capacity(model)
        add_cluster!(model)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a1λ[k2][zi] += y[i] - u[i]
    b1λ[k2][zi] += 1
    a1ϵ[k2][zi] += u[i]
    b1ϵ[k2][zi] += 1 - u[i]
    sumlogfactyu[k2][zi] += logfactorial(y[i] - u[i])

    # Modify cluster/group k1/zi
    a1λ[k1][zi] -= y[i] - u[i]
    b1λ[k1][zi] -= 1
    a1ϵ[k1][zi] -= u[i]
    b1ϵ[k1][zi] -= 1 - u[i]
    sumlogfactyu[k1][zi] -= logfactorial(y[i] - u[i])
end

function logpredlik(model::BerPoiDDP, data, i::Int, k::Int)
    d = cluster_labels(model)
    @extract data : y x
    @extract model : a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu u gamma
    zi = iszero(gamma[x[i]]) ? 1 : x[i]
    log_num = log_den = 0.0
    if d[i] == k
        log_num = logmglik(model, j, k)
        a1λ[k1][zi] -= y[i] - u[i]
        b1λ[k1][zi] -= 1
        a1ϵ[k1][zi] -= u[i]
        b1ϵ[k1][zi] -= 1 - u[i]
        log_den = logmglik(model, j, k)
        a1λ[k1][zi] += y[i] - u[i]
        b1λ[k1][zi] += 1
        a1ϵ[k1][zi] += u[i]
        b1ϵ[k1][zi] += 1 - u[i]
    else
        log_den = logmglik(model, j, k)
        a1λ[k1][zi] += y[i] - u[i]
        b1λ[k1][zi] += 1
        a1ϵ[k1][zi] += u[i]
        b1ϵ[k1][zi] += 1 - u[i]
        log_num = logmglik(model, j, k)
        a1λ[k1][zi] -= y[i] - u[i]
        b1λ[k1][zi] -= 1
        a1ϵ[k1][zi] -= u[i]
        b1ϵ[k1][zi] -= 1 - u[i]
    end
    return log_num - log_den
end

# function logpredlik(model::BerPoiDDP, train, predict, i::Int, k::Int)
#     d = cluster_labels(model)
#     @extract predict : y x
#     @extract model : a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu u gamma
#     zi = iszero(gamma[x[i]]) ? 1 : x[i]
#     log_num = log_den = 0.0
#     if d[i] == k
#         log_num = logmglik(model, j, k)
#         a1λ[k1][zi] -= y[i] - u[i]
#         b1λ[k1][zi] -= 1
#         a1ϵ[k1][zi] -= u[i]
#         b1ϵ[k1][zi] -= 1 - u[i]
#         log_den = logmglik(model, j, k)
#         a1λ[k1][zi] += y[i] - u[i]
#         b1λ[k1][zi] += 1
#         a1ϵ[k1][zi] += u[i]
#         b1ϵ[k1][zi] += 1 - u[i]
#     else
#         log_den = logmglik(model, j, k)
#         a1λ[k1][zi] += y[i] - u[i]
#         b1λ[k1][zi] += 1
#         a1ϵ[k1][zi] += u[i]
#         b1ϵ[k1][zi] += 1 - u[i]
#         log_num = logmglik(model, j, k)
#         a1λ[k1][zi] -= y[i] - u[i]
#         b1λ[k1][zi] -= 1
#         a1ϵ[k1][zi] -= u[i]
#         b1ϵ[k1][zi] -= 1 - u[i]
#     end
#     return log_num - log_den
# end

function logmglik(model::BerPoiDDP, j::Int, k::Int)
    @extract model : a0ϵ b0ϵ a1ϵ b1ϵ a0λ b0λ a1λ b1λ sumlogfactyu u gamma
    return (
        + logbeta(a1ϵ[k][j], b1ϵ[k][j])
        - logbeta(a0ϵ, b0ϵ)
        - a1λ[k][j] * log(b1λ[k][j])
        + a0λ * log(b0λ)
        + loggamma(a1λ[k][j])
        - loggamma(a0λ)
        - sumlogfactyu[k][j]
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

# function update_u!()

function update_hyperpars!(rng::AbstractRNG, model::BerPoiDDP, data)
    update_gamma!(rng, model, data)
end

# Tõnu Kollo (tonu.kollo@ut.ee) University of Tartu, Tartu, Estonia
