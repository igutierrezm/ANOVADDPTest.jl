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
tblf =
    fm |>
    x -> fpost(x) |>
    x -> transform(x, :group => (x -> string.(x)) => :group)
tblf0 = @filter(tblf, group == "1")
ygrid0 = tblf0.y
fgrid0 = aweights(tblf0.f / sum(tblf0.f))
myfun(x) = quantile(ygrid0, fgrid0, x)

tbl_shift = @chain tblf begin
    @filter(group != "1")
    @arrange(group, y)
    @group_by(group)
    @mutate(F = cumsum(f))
    @mutate(F = F / maximum(F))
    @mutate(shift = myfun.(F))
    @ungroup
end

ggplot(tbl_shift, aes(y = "shift", x = "y", color = "group")) +
    geom_line()



# # p = data(f) * mapping(:y, :value) * visual(Lines) * mapping(color = :variable)
# # draw(p)

# mean_density =
#     fitted_model.densitypost |>
#     # x -> filter!(:group => x -> x == 2, x) |>
#     # x -> groupby(x, [:group, :y]) |>
#     # x -> combine(x, :value => mean => :value) |>
#     x -> transform(x, :group => (x -> string.(x)) => :group)

# ggplot(mean_density, aes(y = "f", x = "y", color = "group")) +
#     geom_line()
#     #facet_wrap(cols = "group")

