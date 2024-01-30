struct BernoulliData
    X::Matrix{Int}
    x::Vector{Int}
    y::Vector{Bool}
    Xunique::Vector{Vector{Int}}
    function BernoulliData(X::Matrix{Int}, y::Vector{Bool})
        x = denserank([X[i, :] for i in 1:size(X, 1)])
        Xunique = sort(unique([X[i, :] for i in 1:size(X, 1)]))
        new(X, x, y, Xunique)
    end
end

BernoulliData(x::Vector{Int}, y::Vector{Bool}) = BernoulliData(x[:, :], y)

struct BernoulliDDP <: AbstractDPM
    parent::DPM
    a2::Float64
    b2::Float64
    a2_post::Vector{Vector{Float64}}
    b2_post::Vector{Vector{Float64}}
    gammaprior::Womack
    gamma::Vector{Bool}
    ngroups::Int
    function BernoulliDDP(
        rng::AbstractRNG,
        N::Int,
        ngroups::Int;
        K0::Int = 1,
        a::Float64 = 2.0,
        b::Float64 = 4.0,
        a2::Float64 = 2.0,
        b2::Float64 = 4.0,
        rho::Float64 = 1.0,
    )
        parent = DPM(rng, N; K0, a0 = a, b0 = b)
        a2_post = [a2 * ones(ngroups)]
        b2_post = [b2 * ones(ngroups)]
        gammaprior = Womack(ngroups - 1, rho)
        gamma = ones(Bool, ngroups)
        new(parent, a2, b2, a2_post, b2_post, gammaprior, gamma, ngroups)
    end
end

function parent_dpm(model::BernoulliDDP)
    model.parent
end

function add_cluster!(model::BernoulliDDP)
    @extract model : a2 b2 a2_post b2_post ngroups
    push!(a2_post, a2 * ones(Int, ngroups))
    push!(b2_post, b2 * ones(Int, ngroups))
end

function update_suffstats!(model::BernoulliDDP, data)
    @extract data : y x
    @extract model : a2 b2 a2_post b2_post gamma
    d = cluster_labels(model)
    while length(a2_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    for k in active_clusters(model)
        a2_post[k] .= a2
        b2_post[k] .= b2
    end
    for i = 1:length(y)
        di = d[i]
        zi = iszero(gamma[x[i]]) ? 1 : x[i]
        a2_post[di][zi] += y[i]
        b2_post[di][zi] += 1 - y[i]
    end
end

function update_suffstats!(model::BernoulliDDP, data, i::Int, k1::Int, k2::Int)
    @extract data : y x
    @extract model : a2_post b2_post gamma
    while length(a2_post) < cluster_capacity(model)
        add_cluster!(model)
    end
    zi = iszero(gamma[x[i]]) ? 1 : x[i]

    # Modify cluster/group k2/zi
    a2_post[k2][zi] += y[i]
    b2_post[k2][zi] += 1 - y[i]

    # Modify cluster/group k1/zi
    a2_post[k1][zi] -= y[i]
    b2_post[k1][zi] -= 1 - y[i]
end

function logpredlik(model::BernoulliDDP, data, i::Int, k::Int)
    @extract data : y x
    @extract model : a2_post b2_post gamma
    d = cluster_labels(model)
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a2_post[k][j] - (d[i] == k) * y[i]
    b1kj = b2_post[k][j] - (d[i] == k) * (1 - y[i])
    if y[i]
        return log(a1kj / (a1kj + b1kj))
    else
        return log(b1kj / (a1kj + b1kj))
    end
end

function logpredlik(model::BernoulliDDP, train, predict, i::Int, k::Int)
    @extract predict : y x
    @extract model : a2_post b2_post gamma
    j = iszero(gamma[x[i]]) ? 1 : x[i]
    a1kj = a2_post[k][j]
    b1kj = b2_post[k][j]
    if y[i] == 1
        return log(a1kj / (a1kj + b1kj))
    else
        return log(b1kj / (a1kj + b1kj))
    end
end

function logmglik(model::BernoulliDDP, j::Int, k::Int)
    @extract model : a2 b2 a2_post b2_post
    return logbeta(a2_post[k][j], b2_post[k][j]) - logbeta(a2, b2)
end

function update_gamma!(rng::AbstractRNG, model::BernoulliDDP, data)
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

function update_hyperpars!(rng::AbstractRNG, model::BernoulliDDP, data)
    update_gamma!(rng, model, data)
end

function density(model::BernoulliDDP, predict; v = 0.01, ϵ = 0.01)
    wnew, snew = polya_completion(m; v, ϵ)
    atoms = draw_atoms(model, maximum(snew))
    @extract predict : y x
    npoints = length(y)
    fgrid = zeros(npoints)
    nclusters = length(snew)
    for i in 1:npoints
        for k in 1:nclusters
            x0 = x[i]
            y0 = y[i]
            w0 = wnew[k]
            s0 = snew[k]
            θsj = atoms[s0][x0]
            dist0 = Bernoulli(θsj)
            fgrid[i] += w0 * pdf(dist0, y0)
        end
    end
    return fgrid
end

function draw_atoms(model::BernoulliDDP, max_snew::Int)
    out = map(1:max_snew) do k
        map(1:model.ngroups) do j
            draw_atom(model, k, j)
        end
    end
    return out
end

function draw_atom(model::BernoulliDDP, k::Int, j::Int)
    @extract model : a2_post b2_post a2 b2
    if k in active_clusters(model)
        thetakj = rand(Beta(a2_post[k][j], b2_post[k][j]))
    else
        thetakj = rand(Beta(a2, b2))
    end
    return thetakj
end
