struct Womack <: DiscreteMultivariateDistribution
    D::Int
    ζ::Float64
    p::Vector{Float64}
    punnormalized::Vector{Float64}
    function Womack(D::Int, ζ::Float64)
        p = big.([zeros(D); 1.0])
        for d1 in (D - 1):-1:0
            for d2 in 1:(D - d1)
                p[1 + d1] += ζ * p[1 + d1 + d2] * binomial(big(d1 + d2), big(d1))
            end
        end
        p /= sum(p)
        punnormalized = copy(p)
        for d1 in 1:D
            p[d1] /= binomial(big(D), big(d1 - 1))
        end
        return new(D, ζ, p, punnormalized)
    end
end

function pdf(d::Womack, γ::Vector{Bool})
    return d.p[sum(γ) + 1]
end

function logpdf(d::Womack, γ::Vector{Bool})
    return log(pdf(d, γ))
end
