begin
    using Revise
    using ANOVADDPTest
    using DataFrames
    using Distributions
    using Statistics
    using RCall
    using Random
    using Statistics
    using StatsBase
end

begin
    N = 1000
    Random.seed!(1)
    X = rand(0:2, N, 1)
    y = rand(Poisson(1), N) + rand(Bernoulli(0.5), N)
    fm = anova_bnp_berpoi(y, X);
    # tbl_shiftpost = ANOVADDPTest.shiftpost(fm)
end
