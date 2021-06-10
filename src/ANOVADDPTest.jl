module ANOVADDPTest

using Random: MersenneTwister
using SpecialFunctions
using Parameters: @unpack
using DPMNeal3: GenericBlock, update!
import DPMNeal3: update!, update_sb!, logpredlik
export GenericBlock, SpecificBlock, Data, update!, fit

struct Data
    x::Vector{Int}
    y::Vector{Float64}
end

"""
    SpecificBlock(G; v0 = 2.0, r0 = 1.0, u0 = 0.0, s0 = 1.0)

TODO
"""
struct SpecificBlock
    G::Int
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
    function SpecificBlock(G; v0 = 2.0, r0 = 1.0, u0 = 0.0, s0 = 1.0)
        γ = ones(Bool, G)
        v1 = [v0 * ones(G)]
        r1 = [r0 * ones(G)]
        u1 = [u0 * ones(G)]
        s1 = [s0 * ones(G)]
        πγ = ones(G) / G
        new(G, v0, r0, u0, s0, v1, r1, u1, s1, πγ, γ)
    end
end

function resize!(sb::SpecificBlock, n::Integer)
    @unpack G, v1, r1, u1, s1, v0, r0, u0, s0 = sb
    while length(v1) < n
        push!(v1, v0 * ones(G))
        push!(r1, r0 * ones(G))
        push!(u1, u0 * ones(G))
        push!(s1, s0 * ones(G))
    end    
end

function update_γ!(rng, sb::SpecificBlock, gb::GenericBlock, data)
    @unpack πγ, γ = sb
    @unpack A = gb

    # Resample γ[g], given the other γ's
    for g = 2:length(γ)
        # log-odds (numerator)
        γ[g] = 1
        update_suffstats!(sb, gb, data)
        log_num = log(πγ[sum(γ)])
        for k ∈ A, j ∈ (1, g)
            log_num += logmglik(sb, j, k)
        end

        # log-odds (denominator)
        γ[g] = 0
        update_suffstats!(sb, gb, data)
        log_den = log(πγ[sum(γ)])
        for k ∈ A, j ∈ (1)
            log_den += logmglik(sb, j, k)
            # println(logmglik(sb, j, k))
        end

        # log-odds and new γ[g]
        log_odds = log_num - log_den
        γ[g] = rand(rng) <= 1 / (1 + exp(-log_odds))
    end
end

function update_suffstats!(sb::SpecificBlock, gb::GenericBlock, data)
    @unpack y, x = data
    @unpack N, A, d, n = gb
    @unpack v1, r1, u1, s1, v0, r0, u0, s0, γ = sb
    length(v1) < length(n) && resize!(sb, length(n))
    for k in A
        v1[k] .= v0
        r1[k] .= r0
        u1[k] .= u0
        s1[k] .= s0
    end
    for i = 1:N
        zi = iszero(γ[x[i]]) ? 1 : x[i]
        di = d[i]
        v1[di][zi] += 1
        rm = r1[di][zi] += 1
        um = u1[di][zi] = ((rm - 1) * u1[di][zi] + y[i]) / rm
        s1[di][zi] += (rm / (rm - 1)) * (y[i] - um)^2
    end
end

function update_sb!(rng, sb::SpecificBlock, gb::GenericBlock, data)
    update_γ!(rng, sb, gb, data)
    update_suffstats!(sb, gb, data)
end

function update_sb!(rng, sb::SpecificBlock, gb::GenericBlock, data, i, k1, k2)
    @unpack n = gb
    @unpack y, x = data
    @unpack v1, r1, u1, s1, γ = sb
    length(v1) < length(n) && resize!(sb, length(n))
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

function logpredlik(sb::SpecificBlock, gb::GenericBlock, data, i, k)
    @unpack v1, r1, u1, s1, γ = sb
    @unpack y, x = data
    @unpack d = gb
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

function logmglik(sb::SpecificBlock, j, k)
    @unpack v0, v1, r0, r1, s0, s1 = sb
    return(
        0.5v0 * log(s0) -
        0.5v1[k][j] * log(s1[k][j]) +
        loggamma(v1[k][j] / 2) -
        loggamma(v0 / 2) +
        0.5 * log(r0 / r1[k][j]) -
        0.5 * log(π) * (r1[k][j] - r0)
    )
end

function fit(y, x; seed = 0, iter = 100, warmup = iter ÷ 2, thin = 1)
    rng = seed == 0 ? MersenneTwister() : MersenneTwister()
    N = length(y)
    G = length(unique(x))
    data = Data(x, y)
    sb = SpecificBlock(G)
    gb = GenericBlock(rng, N)
    γs = [zeros(Bool, G) for _ = 1:(iter - warmup) ÷ thin]
    for t in 1:iter
        update!(rng, sb, gb, data)
        if t > warmup
            γs[(t - warmup) ÷ thin] = sb.γ[:]
        end
    end
    return γs
end

end # module
