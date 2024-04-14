begin
    using Revise
    using ANOVADDPTest
    using DataFrames
    using Distributions
    using Statistics
    using RCall
    using Random
    using TidierData
    using TidierPlots
    using Statistics
    using StatsBase
end

N = 1000
Random.seed!(1)
X = rand(0:1, N, 1)
y = 1.2 * (X[:, 1] .== 1) .* (2 * (rand(N) .<= 0.7) .- 1) .+ randn(N) / 2

fm = anova_bnp_normal(y, X; standardize_y = true, iter = 10000, warmup = 5000);
tbl_Fpost = ANOVADDPTest.Fpost(fm) #shiftpost(fm)

R"""
$tbl_Fpost |>
    dplyr::mutate(group = as.character(group)) |>
    ggplot2::ggplot(ggplot2::aes(x = y, y = F, color = group)) +
    ggplot2::geom_line()
"""
