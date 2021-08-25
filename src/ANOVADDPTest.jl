module ANOVADDPTest

using DataFrames
using Distributions
using DPMNeal3
using ExtractMacro
using Random
using SpecialFunctions
using StatsBase

import Distributions: pdf, logpdf, shape, scale, rate, rand
import DPMNeal3: parent_dpm, logpredlik, update_hyperpars!, update_suffstats!

export BernoulliDDP, NormalDDP, PoissonDDP
export BernoulliData, NormalData, PoissonData
export train, simple_effect_probabilities, interaction_effect_probabilities, expandgrid

include("expandgrid.jl")
include("womack.jl")
include("normalinversegamma.jl")
include("normal.jl")
include("poisson.jl")
include("bernoulli.jl")
include("dpm_mcmc_chain.jl")
include("train.jl")
include("simple_effect_probabilities.jl")
include("interaction_effect_probabilities.jl")

end # module
