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
    x = rand(0:2, N)
    d = @. 2 * (rand() <= 0.7) - 1
    y = @. 1.2 * (x == 1) * d + randn() / 2
    fm = anova_bnp_normal(y, x[:, :])
    tbl_shiftpost = ANOVADDPTest.shiftpost(fm)
end

R"""
$tbl_shiftpost |>
    dplyr::mutate(group1 = as.character(group1)) |>
    dplyr::mutate(group2 = as.character(group2)) |>
    ggplot2::ggplot(ggplot2::aes(x = y, y = shift)) +
    ggplot2::geom_line() +
    ggplot2::facet_grid(
        cols = ggplot2::vars(group1),
        rows = ggplot2::vars(group2)
    )
"""
