begin
    using Gadfly
    using ANOVADDPTest
    using DataFrames
    using Distributions
    using Random
end

function rberpoi(rng, alpha, lambda)
    rand(rng, Bernoulli(alpha)) + rand(rng, Poisson(lambda))
end

function dberpoi(alpha, lambda, x)
    (0 + alpha) * pdf(Poisson(lambda), x - 1) +
    (1 - alpha) * pdf(Poisson(lambda), x - 0)
end


function generate_plot_berpoi(N, ngroups, distinct_groups)
    rng = MersenneTwister(1)
    model = BerPoiDDP(rng, N, ngroups)
    x = rand(rng, 1:ngroups, N)
    y = [rberpoi(rng, 0.5, 0.5) for i in 1:N]
    for i in 1:N
        x[i] in distinct_groups && (y[i] = rberpoi(rng, 0.5, 1.0))
    end
    data = BerPoiData(x, y)
    pred = BerPoiData(expandgrid(1:ngroups, 0:8)...)
    ch = train(rng, model, data, pred)
    f_true = [
        dberpoi(0.5, 0.5 + 0.5 * (pred.x[i] in distinct_groups), pred.y[i])
        for i in eachindex(pred.x)
    ]
    df = DataFrame(x = pred.x, y = pred.y, f_fit = mean(ch.f), f_true = f_true)
    df = stack(df, 3:4)
    plot(df, x = :y, ygroup = :x, y = :value, color = :variable, Geom.subplot_grid(Geom.line), Scale.color_discrete())
end

generate_plot_berpoi(10000, 3, [])
generate_plot_berpoi(1000, 3, [2])
generate_plot_berpoi(1000, 3, [3])
generate_plot_berpoi(10000, 3, [2, 3])

