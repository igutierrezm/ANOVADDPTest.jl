struct PoissonGammaSB
    G::Int
    a0::Float64
    b0::Float64
    s1::Vector{Vector{Int}}
    m1::Vector{Vector{Int}}
    πγ::Vector{Float64}
    γ::Vector{Bool}
    function PoissonGammaSB(G; a0 = 2.0, b0 = 1.0)
        γ = ones(Bool, G)
        πγ = ones(G) / G
        s1 = [zeros(Int, G)]
        m1 = [zeros(Int, G)]
        new(G, a0, b0, s1, m1, πγ, γ)
    end
end

function resize!(sb::PoissonGammaSB, n::Integer)
    @unpack s1, m1 = sb
    while length(s1) < n
        push!(s1, zeros(Int, G))
        push!(m1, zeros(Int, G))
    end    
end

function update_suffstats!(sb::PoissonGammaSB, gb::GenericBlock, data)
    @unpack y, x = data
    @unpack N, A, d, n = gb
    @unpack s1, m1, γ = sb
    length(s1) < length(n) && resize!(sb, length(n))
    for k in A
        s1[k] .= 0
        m1[k] .= 0
    end
    for i = 1:N
        di = d[i]
        zi = iszero(γ[x[i]]) ? 1 : x[i]
        s1[di][zi] += y[i]
        m1[di][zi] += 1
    end
end

function logpredlik(sb::SpecificBlock, gb::GenericBlock, data, i, k)
    @unpack s1, m1, γ = sb
    @unpack y, x = data
    @unpack d = gb
    yi = y[i]
    di = d[i]
    zi = iszero(γ[x[i]]) ? 1 : x[i]
    a1 = a0 + s1[k][zi] - (di == k) * yi
    b1 = b0 + m1[k][zi] - (di == k)
    return logpdf(NegativeBinomial(a1, b1 / (b1 + 1)), yi)
end

