using Gadfly
using ANOVADDPTest
using DataFrames
using Distributions
using Random

function generate_plot_bernoulli(N, G, signigicant_groups)
    rng = MersenneTwister(1)
    m = BernoulliDDP(rng, N, G)
    x = rand(rng, 1:G, N)
    y = rand(rng, Bernoulli(0.25), N)
    for i = 1:N
        x[i] in signigicant_groups && (y[i] = rand(rng, Bernoulli(0.75)))
    end
    data = BernoulliData(x, y)
    pred = BernoulliData(expandgrid(1:G, false:true)...)
    ch = train(rng, m, data, pred)
    df = DataFrame(x = pred.x, y = pred.y, f = mean(ch.f))
    plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())
end

generate_plot_bernoulli(1000, 3, [])
generate_plot_bernoulli(1000, 3, [2])
generate_plot_bernoulli(1000, 3, [3])
generate_plot_bernoulli(1000, 3, [2, 3])
