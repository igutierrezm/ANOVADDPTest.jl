module ANOVADDPTest

using Random: AbstractRNG, MersenneTwister
using Distributions: logpdf, NegativeBinomial
using SpecialFunctions: loggamma, logbeta, logfactorial
using ExtractMacro: @extract
using DPMNeal3
import Base.length
import DPMNeal3: parent_dpm, logpredlik, update_hyperpars!, update_suffstats!
# export GenericBlock, SpecificBlock, Data, update!, train
export NormalDDP, PoissonDDP, NormalData, PoissonData, BernoulliDDP, BernoulliData, dp_mass, n_clusters, cluster_sizes, cluster_capacity, cluster_labels, active_clusters, passive_clusters, update!, train
using StatsBase: counts, denserank

include("normal.jl")
include("poisson.jl")
include("bernoulli.jl")

struct DPM_MCMC_Chain
    gamma::Vector{Vector{Bool}}
    f::Vector{Vector{Float64}}
end

function predlik(m::AbstractDPM, train, predict, i::Int)
    k̄ = first(passive_clusters(m))
    A = active_clusters(m)
    n = cluster_sizes(m)
    N = length(train)
    α = dp_mass(m)
    ans = 0.0
    for k in A
        ans += exp(logpredlik(m, train, predict, i, k)) * n[k] / (N + α)
    end
    ans += exp(logpredlik(m, train, predict, i, k̄)) * α / (N + α)
    return(ans)
end

function train(rng, m::AbstractDPM, train, predict; iter = 2000, warmup = iter ÷ 2, thin = 1)
    gammachain = [zeros(Bool, m.G) for _ = 1:(iter - warmup) ÷ thin]
    fchain = [zeros(length(predict)) for _ = 1:(iter - warmup) ÷ thin]
    for t in 1:iter
        update!(rng, m, train)
        if t > warmup
            t0 = (t - warmup) ÷ thin
            gammachain[t0] = m.gamma[:]
            for i = 1:length(predict)
                fchain[t0][i] = predlik(m, train, predict, i)
            end
        end
    end
    return DPM_MCMC_Chain(gammachain, fchain)
end

end # module
