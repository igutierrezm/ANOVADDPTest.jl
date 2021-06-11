struct BernoulliBetaSB
    G::Int
    a0::Float64
    b0::Float64
    a1::Vector{Vector{Int}}
    b1::Vector{Vector{Int}}
    πγ::Vector{Float64}
    γ::Vector{Bool}
    function PoissonGammaSB(G; a0 = 2.0, b0 = 1.0)
        γ = ones(Bool, G)
        πγ = ones(G) / G
        a1 = [a0 * ones(Int, G)]
        b1 = [b0 * ones(Int, G)]
        new(G, a0, b0, a1, b1, πγ, γ)
    end
end

function resize!(sb::BernoulliBetaSB, n::Integer)
    @unpack a1, b1 = sb
    while length(a1) < n
        push!(a1, zeros(Int, G))
        push!(b1, zeros(Int, G))
    end    
end

function update_suffstats!(sb::BernoulliBetaSB, gb::GenericBlock, data)
    @unpack y, x = data
    @unpack a1, b1, γ = sb
    @unpack N, A, d, n = gb
    length(a1) < length(n) && resize!(sb, length(n))
    for k in A
        a1[k] .= a0
        b1[k] .= b0
    end
    for i = 1:N
        di = d[i]
        zi = iszero(γ[x[i]]) ? 1 : x[i]
        a1[di][zi] += y[i]
        b1[di][zi] += 1 - y[i]
    end
end

function logpredlik(sb::BernoulliBetaSB, gb::GenericBlock, data, i, k)
    @unpack a1, b1, γ = sb
    @unpack y, x = data
    @unpack d = gb
    j = iszero(γ[x[i]]) ? 1 : x[i]
    a1kj = a1[k][j] - (d[i] == k) * y[i]
    b1kj = b1[k][j] - (d[i] == k)
    if y[i] == 1
        return log(a1kj / (a1kj + b1kj))
    else
        return log(b1kj / (a1kj + b1kj))
    end
end

function logmglik(sb::BernoulliBetaSB, j, k)
    @unpack a0, b0, a1, b1, sumlogu = sb
    return (
        a0 * log(b0) - a1[k][j] * log(b1[k][j]) +
        loggamma(a1[k][j]) - loggamma(a0) -
        sumlogu[k][j]
    )
end
