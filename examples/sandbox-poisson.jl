using Gadfly
using ANOVADDPTest
using DataFrames
using Distributions
using Random

function generate_plot_poisson(N, G, distinct_groups)
    rng = MersenneTwister(1)
    m = PoissonDDP(rng, N, G)
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(2), N)
    for i in 1:N
        x[i] in distinct_groups && (y[i] = rand(rng, Poisson(4)))
    end
    data = PoissonData(x, y)
    pred = PoissonData(expandgrid(1:G, 0:8)...)
    ch = train(rng, m, data, pred)
    df = DataFrame(x = pred.x, y = pred.y, f = mean(ch.f))
    plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())
end

generate_plot_poisson(1000, 3, [])
generate_plot_poisson(1000, 3, [2])
generate_plot_poisson(1000, 3, [3])
generate_plot_poisson(1000, 3, [2, 3])
