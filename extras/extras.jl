using Revise
using ANOVADDPTest
using DataFrames
using Distributions
using Statistics
using Random
using TidierData
using TidierPlots
using Statistics
using StatsBase

N = 1000
Random.seed!(1)
X = rand(0:1, N, 1)
y = 1.2 * (X[:, 1] .== 1) .* (2 * (rand(N) .<= 0.7) .- 1) .+ randn(N) / 2

fm = anova_bnp_normal(y, X; standardize_y = true, iter = 10000, warmup = 5000);
tbl_shift = shiftpost(fm)

ggplot(tbl_shift, aes(y = "shift", x = "y")) +
    geom_line()
