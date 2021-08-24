module ANOVADDPTest

using Random: AbstractRNG, MersenneTwister
using Distributions: logpdf, NegativeBinomial
using Distributions: DiscreteMultivariateDistribution, ContinuousUnivariateDistribution
using SpecialFunctions: loggamma, logbeta, logfactorial
using ExtractMacro: @extract
using DPMNeal3
using DataFrames
import Base.length
import Distributions: pdf, logpdf, shape, scale, rate, rand
import DPMNeal3: parent_dpm, logpredlik, update_hyperpars!, update_suffstats!
# export GenericBlock, SpecificBlock, Data, update!, train
export NormalDDP, PoissonDDP, NormalData, PoissonData, BernoulliDDP, BernoulliData, dp_mass, n_clusters, cluster_sizes, cluster_capacity, cluster_labels, active_clusters, passive_clusters, update!, train, simple_effect_probabilities, interaction_effect_probabilities
using StatsBase: counts, denserank

include("womack.jl")
include("normalinversegamma.jl")
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
        if t > warmup && ((t - warmup) % thin == 0)
            t0 = (t - warmup) ÷ thin
            gammachain[t0] = m.gamma[:]
            for i = 1:length(predict)
                fchain[t0][i] = predlik(m, train, predict, i)
            end
        end
    end
    return DPM_MCMC_Chain(gammachain, fchain)
end

function simple_effect_probabilities(chain, predict)
    Ngroups = length(predict.Xunique)
    Nvars = size(predict.X, 2)
    gamma = hcat(expandgrid([1], [[0, 1] for _ in 1:Ngroups-1]...)...)
    df = DataFrame(:gamma => [gamma[i, :] for i in 1:size(gamma, 1)])
    df1 = deepcopy(df)
    for var in 1:Nvars
        df1[!, "var_$var"] = zeros(Bool, size(df1, 1))
        for h in 1:size(df1, 1)
            for row in 1:Ngroups
                predict.Xunique[row][var] == 1 && continue
                df1[h, "gamma"][row] == 0 && continue
                df1[h, "var_$var"] = true
            end
        end
    end

    unique_gamma = unique(chain.gamma)
    Ngammas = length(unique_gamma)
    Niter = length(chain.gamma)
    gamma_prob = zeros(Ngammas)
    for j in 1:Ngammas
        gamma_prob[j] = sum([chain.gamma[i] == unique_gamma[j] for i in 1:Niter])
    end
    gamma_prob /= sum(gamma_prob)
    probs = DataFrame(:gamma => unique_gamma, :prob => gamma_prob)
    df2 = leftjoin(df1, probs, on = :gamma)
    df2[!, :prob] = coalesce.(df2[!, :prob], 0)

    df3 = df2[!, Not([:gamma, :prob])]
    df4 = combine(df3, names(df3) .=> x -> sum(x .* df2.prob), renamecols = false)
    df5 = stack(df4)
    return df5
end

function interaction_effect_probabilities(chain, predict)
    Ngroups = length(predict.Xunique)
    Nvars = size(predict.X, 2)
    gamma = hcat(expandgrid([1], [[0, 1] for _ in 1:Ngroups-1]...)...)
    df = DataFrame(:gamma => [gamma[i, :] for i in 1:size(gamma, 1)])
    df1 = deepcopy(df)
    for var1 in 1:Nvars, var2 in 1:Nvars
        var1 ≥ var2 && continue
        df1[!, "var_$(var1)_$(var2)"] = zeros(Bool, size(df1, 1))
        for h in 1:size(df1, 1)
            for row in 1:Ngroups
                predict.Xunique[row][var1] == 1 && continue
                predict.Xunique[row][var2] == 1 && continue
                df1[h, "gamma"][row] == 0 && continue
                df1[h, "var_$(var1)_$(var2)"] = true
            end
        end
    end

    unique_gamma = unique(chain.gamma)
    Ngammas = length(unique_gamma)
    Niter = length(chain.gamma)
    gamma_prob = zeros(Ngammas)
    for j in 1:Ngammas
        gamma_prob[j] = sum([chain.gamma[i] == unique_gamma[j] for i in 1:Niter])
    end
    gamma_prob /= sum(gamma_prob)
    probs = DataFrame(:gamma => unique_gamma, :prob => gamma_prob)
    df2 = leftjoin(df1, probs, on = :gamma)
    df2[!, :prob] = coalesce.(df2[!, :prob], 0)

    df3 = df2[!, Not([:gamma, :prob])]
    df4 = combine(df3, names(df3) .=> x -> sum(x .* df2.prob), renamecols = false)
    df5 = stack(df4)
    return df5
end

function expandgrid(levels...)
    lengths = length.(levels)
    inner = 1
    outer = prod(lengths)
    grid = []
    for i in 1:length(levels)
        outer = div(outer, lengths[i])
        push!(grid, repeat(levels[i], inner=inner, outer=outer))
        inner *= lengths[i]
    end
    Tuple(grid)
end

end # module
