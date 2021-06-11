struct PoissonGammaSB
    G::Int
    a0::Float64
    b0::Float64
    a1::Vector{Vector{Int}}
    b1::Vector{Vector{Int}}
    sumlogu::Vector{Vector{Float64}}
    πγ::Vector{Float64}
    γ::Vector{Bool}
    function PoissonGammaSB(G; a0 = 2.0, b0 = 1.0)
        γ = ones(Bool, G)
        πγ = ones(G) / G
        a1 = [a0 * ones(Int, G)]
        b1 = [b0 * ones(Int, G)]
        sumlogu = [zeros(Float64, G)]
        new(G, a0, b0, a1, b1, sumlogu, πγ, γ)
    end
end

function resize!(sb::PoissonGammaSB, n::Integer)
    @unpack a1, b1 = sb
    while length(a1) < n
        push!(a1, zeros(Int, G))
        push!(b1, zeros(Int, G))
    end    
end

function update_suffstats!(sb::PoissonGammaSB, gb::GenericBlock, data)
    @unpack y, x = data
    @unpack a1, b1, γ = sb
    @unpack N, A, d, n = gb
    length(a1) < length(n) && resize!(sb, length(n))
    for k in A
        a1[k] .= a0
        b1[k] .= b0
        sumlogu .= 0.0
    end
    for i = 1:N
        di = d[i]
        zi = iszero(γ[x[i]]) ? 1 : x[i]
        sumlogu[di][zi] += logfactorial(y[i])
        a1[di][zi] += y[i]
        b1[di][zi] += 1
    end
end

function logpredlik(sb::PoissonGammaSB, gb::GenericBlock, data, i, k)
    @unpack a1, b1, γ = sb
    @unpack y, x = data
    @unpack d = gb
    j = iszero(γ[x[i]]) ? 1 : x[i]
    a1kj = a1[k][j] - (d[i] == k) * y[i]
    b1kj = b1[k][j] - (d[i] == k)
    return logpdf(NegativeBinomial(a1kj, b1kj / (b1kj + 1)), y[i])
end

function logmglik(sb::PoissonGammaSB, j, k)
    @unpack a0, b0, a1, b1, sumlogu = sb
    return (
        a0 * log(b0) - a1[k][j] * log(b1[k][j]) +
        loggamma(a1[k][j]) - loggamma(a0) -
        sumlogu[k][j]
    )
end
