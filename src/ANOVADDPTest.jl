module ANOVADDPTest

using Random: AbstractRNG, MersenneTwister
using Distributions: logpdf
using SpecialFunctions
using Parameters: @unpack
using DPMNeal3
import DPMNeal3: parent_dpm, logpredlik, update_hyperpars!, update_suffstats!
# export GenericBlock, SpecificBlock, Data, update!, fit
export NormalDDP, Data, dp_mass, n_clusters, cluster_sizes, cluster_capacity, cluster_labels, active_clusters, passive_clusters, update!

struct Data
    x::Vector{Int}
    y::Vector{Float64}
end

"""
    ANOVADDP_SS_Normal_NIG(rng::AbstractRNG, N::Int, G::Int; K0 = 5, a0 = 2.0, b0 = 4, v0 = 2.0, r0 = 1.0, u0 = 0.0, s0 = 1.0)

TODO
"""
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
    πγ::Vector{Float64}
    γ::Vector{Bool}
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
            s0::Float64 = 1.0
        )
        parent = DPM(rng, N; K0, a0, b0)
        v1 = [v0 * ones(G)]
        r1 = [r0 * ones(G)]
        u1 = [u0 * ones(G)]
        s1 = [s0 * ones(G)]
        πγ = ones(G) / G
        γ = ones(Bool, G)
        new(parent, v0, r0, u0, s0, v1, r1, u1, s1, πγ, γ, G)
    end
end

function parent_dpm(m::NormalDDP)
    m.parent
end

function add_cluster!(m::NormalDDP)
    @unpack G, v0, r0, u0, s0, v1, r1, u1, s1 = m
    push!(v1, v0 * ones(G))
    push!(r1, r0 * ones(G))
    push!(u1, u0 * ones(G))
    push!(s1, s0 * ones(G))
end

function update_suffstats!(m::NormalDDP, data)
    @unpack v0, r0, u0, s0, v1, r1, u1, s1, γ = m
    @unpack y, x = data
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
        zi = iszero(γ[x[i]]) ? 1 : x[i]
        di = d[i]
        v1[di][zi] += 1
        rm = r1[di][zi] += 1
        um = u1[di][zi] = ((rm - 1) * u1[di][zi] + y[i]) / rm
        s1[di][zi] += (rm / (rm - 1)) * (y[i] - um)^2
    end
end

function update_suffstats!(m::NormalDDP, data, i::Int, k1::Int, k2::Int)
    @unpack y, x = data
    @unpack v1, r1, u1, s1, γ = m
    while length(v1) < cluster_capacity(m)
        add_cluster!(m)
    end
    zi = iszero(γ[x[i]]) ? 1 : x[i]

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

function update_γ!(rng::AbstractRNG, m::NormalDDP, data)
    @unpack πγ, γ = m
    A = active_clusters(m)

    # Resample γ[g], given the other γ's
    for g = 2:length(γ)
        # log-odds (numerator)
        γ[g] = 1
        update_suffstats!(m, data)
        log_num = log(πγ[sum(γ)])
        for k ∈ A, j ∈ (1, g)
            log_num += logmglik(m, j, k)
        end

        # log-odds (denominator)
        γ[g] = 0
        update_suffstats!(m, data)
        log_den = log(πγ[sum(γ)])
        for k ∈ A, j ∈ (1)
            log_den += logmglik(m, j, k)
            # println(logmglik(m, j, k))
        end

        # log-odds and new γ[g]
        log_odds = log_num - log_den
        γ[g] = rand(rng) <= 1 / (1 + exp(-log_odds))
    end
end

function update_hyperpars!(rng::AbstractRNG, m::NormalDDP, data)
    update_γ!(rng, m, data)
end

function logpredlik(m::NormalDDP, data, i::Int, k::Int)
    @unpack v1, r1, u1, s1, γ = m
    @unpack y, x = data
    d = cluster_labels(m)
    yi = y[i]
    di = d[i]
    zi = iszero(γ[x[i]]) ? 1 : x[i]

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

function logmglik(m::NormalDDP, j::Int, k::Int)
    @unpack v0, v1, r0, r1, s0, s1 = m
    return(
        0.5v0 * log(s0) -
        0.5v1[k][j] * log(s1[k][j]) +
        loggamma(v1[k][j] / 2) -
        loggamma(v0 / 2) +
        0.5 * log(r0 / r1[k][j]) -
        0.5 * log(π) * (r1[k][j] - r0)
    )
end

# function fit(y, x; seed = 0, iter = 2000, warmup = iter ÷ 2, thin = 1)
#     rng = seed == 0 ? MersenneTwister() : MersenneTwister()
#     N = length(y)
#     G = length(unique(x))
#     data = Data(x, y)
#     sb = SpecificBlock(G)
#     gb = GenericBlock(rng, N)
#     γs = [zeros(Bool, G) for _ = 1:(iter - warmup) ÷ thin]
#     for t in 1:iter
#         update!(rng, m, data)
#         if t > warmup
#             γs[(t - warmup) ÷ thin] = m.γ[:]
#         end
#     end
#     return γs
# end

end # module
