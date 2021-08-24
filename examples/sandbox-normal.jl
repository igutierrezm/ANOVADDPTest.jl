using Revise
using Gadfly
using ANOVADDPTest
using DataFrames
using Statistics
using Random
using StatsBase
include("examples/utils.jl")

#####
##### Example 1
#####

rng = MersenneTwister(1)
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = randn(rng, N)
m = NormalDDP(rng, N, G; K0)
data = NormalData(x, y)
predict =
    expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(x...)
chain = train(rng, m, data, predict)

df = DataFrame(x = predict.x, y = predict.y, f = mean(chain.f))
plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())

#####
##### Example 2
#####

rng = MersenneTwister(1)
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = randn(rng, N)
for i = 1:N
    if x[i] == 2
        y[i] += 1
    end
end
data = NormalData(x, y)
predict =
    expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(x...)
m = NormalDDP(rng, N, G; K0)
chain = train(rng, m, data, predict)

df = DataFrame(x = predict.x, y = predict.y, f = mean(chain.f))
plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())

#####
##### Example 3
#####

rng = MersenneTwister(1)
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = randn(rng, N)
for i = 1:N
    if x[i] == 3
        y[i] += 1
    end
end
data = NormalData(x, y)
predict =
    expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(x...)
m = NormalDDP(rng, N, G; K0)
chain = train(rng, m, data, predict)

df = DataFrame(x = predict.x, y = predict.y, f = mean(chain.f))
plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())

#####
##### Example 4
#####

rng = MersenneTwister(1)
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = randn(rng, N)
for i = 1:N
    if x[i] != 1
        y[i] += 1
    end
end
data = NormalData(x, y)
predict =
    expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(x...)
m = NormalDDP(rng, N, G; K0)
chain = train(rng, m, data, predict)

df = DataFrame(x = predict.x, y = predict.y, f = mean(chain.f))
plot(df, x = :y, y = :f, color = :x, Geom.line, Scale.color_discrete())
