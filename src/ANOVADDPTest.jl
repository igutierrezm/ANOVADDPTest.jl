module ANOVADDPTest

using Random: AbstractRNG, MersenneTwister
using Distributions: logpdf, NegativeBinomial
using SpecialFunctions: loggamma, logbeta, logfactorial
using ExtractMacro: @extract
using DPMNeal3
import DPMNeal3: parent_dpm, logpredlik, update_hyperpars!, update_suffstats!
# export GenericBlock, SpecificBlock, Data, update!, fit
export NormalDDP, PoissonDDP, NormalData, PoissonData, BernoulliDDP, BernoulliData, dp_mass, n_clusters, cluster_sizes, cluster_capacity, cluster_labels, active_clusters, passive_clusters, update!

include("normal.jl")
include("poisson.jl")
include("bernoulli.jl")


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
