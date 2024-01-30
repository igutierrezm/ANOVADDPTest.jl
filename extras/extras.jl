using Revise
using ANOVADDPTest
using DataFrames
using Distributions
using Statistics
using Random
# using AlgebraOfGraphics, CairoMakie
using TidierData
using TidierPlots

Random.seed!(1)
X = rand(0:1, 1000, 1)
y = 4 * X[:, 1] .+ 2 * randn(1000)

fitted_model =
    anova_bnp_normal(y, X; standardize_y = true, iter = 10000, warmup = 5000);
f = fpost(fitted_model)
f.group .= string.(f.group)
# f.ftrue = pdf.(Normal(1, 2), f.y)
# f = stack(f, [:f, :ftrue])

# p = data(f) * mapping(:y, :value) * visual(Lines) * mapping(color = :variable)
# draw(p)

mean_density =
    fitted_model.densitypost |>
    # x -> filter!(:group => x -> x == 2, x) |>
    # x -> groupby(x, [:group, :y]) |>
    # x -> combine(x, :value => mean => :value) |>
    x -> transform(x, :group => (x -> string.(x)) => :group)

ggplot(mean_density, aes(y = "f", x = "y", color = "group")) +
    geom_line()
    #facet_wrap(cols = "group")

