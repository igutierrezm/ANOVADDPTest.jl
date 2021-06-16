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

include("normal.jl")
include("poisson.jl")
include("bernoulli.jl")

function train(rng, m::AbstractDPM, train, predict; iter = 2000, warmup = iter ÷ 2, thin = 1)
    γs = [zeros(Bool, m.G) for _ = 1:(iter - warmup) ÷ thin]
    fy = [zeros(length(predict)) for _ = 1:(iter - warmup) ÷ thin]
    for t in 1:iter
        update!(rng, m, train)
        if t > warmup
            γs[(t - warmup) ÷ thin] = m.γ[:]
        end
    end
    return γs
end

end # module
