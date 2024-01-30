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


function shift_functions(fpost)
    # Extract f for the treatment and control groups
    tbl_f0 = filter(:group => (x -> x == 1), fpost) # control group
    tbl_f1 = filter(:group => (x -> x != 1), fpost) # treatment group

    # Create a quantile function
    v = tbl_f0.y
    w = aweights(tbl_f0.f / sum(tbl_f0.f))
    custom_quantile(x) = Statistics.quantile(v, w, x)

    # Compute the shift function for each treatment group
    tbl_shift =
        tbl_f1 |>
        x -> sort(x, [:group, :y]) |>
        x -> groupby(x, :group) |>
        x -> transform(x, :f => (x -> cumsum(x)) => :F; ungroup = false) |>
        x -> transform(x, :F => (x -> x / maximum(x)) => :F; ungroup = false) |>
        x -> transform(x, :F => (x -> custom_quantile(x)) => :shift) |>
        x -> select(x, [:group, :y, :shift])
    return tbl_shift
end

tbl_shift = fpost(fm) |> shift_functions

ggplot(tbl_shift, aes(y = "shift", x = "y")) +
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

