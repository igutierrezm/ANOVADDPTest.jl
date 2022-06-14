using Revise
using Gadfly
using ANOVADDPTest
using DataFrames
using Distributions
using Random

function generate_plot_normal(N, ngroups, distinct_groups)
    rng = MersenneTwister(1)
    model = NormalDDP(rng, N, ngroups)
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    for i in 1:N
        x[i] in distinct_groups && (y[i] += 1)
    end
    data = NormalData(x, y)
    pred = NormalData(expandgrid(1:ngroups,  range(-2.0, stop = 2.0, length = 50))...)
    ch = train(rng, model, data, pred)
    df = DataFrame(x = pred.x, y = pred.y, f = mean(ch.f))
    return plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())
end

generate_plot_normal(1000, 3, [])
generate_plot_normal(1000, 3, [2])
generate_plot_normal(1000, 3, [3])
generate_plot_normal(1000, 3, [2, 3])
