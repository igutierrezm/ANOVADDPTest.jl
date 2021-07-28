using Revise
using StatsPlots
using ANOVADDPTest
using DataFrames
using Statistics
using Random
using StatsBase
include("examples/utils.jl")
gr()

#####
##### Example 1
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 3, 1
x = rand(rng, 1:G, N)
y = randn(rng, N)
m = NormalDDP(rng, N, G; K0)
data = NormalData(x, y)
predict =
    expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(x...)
chain = train(rng, m, data, predict)

df = DataFrame(
    x = predict.x,
    y = predict.y,
    f = mean(chain.f)
)

@df df scatter(
    :y,
    :f,
    group = :x,
    m = (0.6, [:+ :xcross :diamond], 8),
    markerstrokealpha = 0.9,
    bg = RGB(0.2, 0.2, 0.2)
)

#####
##### Example 2
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 3, 1
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

df = DataFrame(
    x = predict.x,
    y = predict.y,
    f = mean(chain.f)
)

@df df scatter(
    :y,
    :f,
    group = :x,
    m = (0.6, [:+ :xcross :diamond], 8),
    markerstrokealpha = 0.9,
    bg = RGB(0.2, 0.2, 0.2)
)

#####
##### Example 3
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 3, 1
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

df = DataFrame(
    x = predict.x,
    y = predict.y,
    f = mean(chain.f)
)

@df df scatter(
    :y,
    :f,
    group = :x,
    m = (0.6, [:+ :xcross :diamond], 8),
    markerstrokealpha = 0.9,
    bg = RGB(0.2, 0.2, 0.2)
)

#####
##### Example 4
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 3, 1
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

df = DataFrame(
    x = predict.x,
    y = predict.y,
    f = mean(chain.f)
)

@df df scatter(
    :y,
    :f,
    group = :x,
    m = (0.6, [:+ :xcross :diamond], 8),
    markerstrokealpha = 0.9,
    bg = RGB(0.2, 0.2, 0.2)
)

#####
##### Example 5
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 2, 1
X = rand(rng, 1:G, N, 2)
y = randn(rng, N)
m = NormalDDP(rng, N, 2^G; K0)
data = NormalData(X = X, y = y)

predict =
    expandgrid(1:G, 1:G, range(-2.0, stop = 2.0, length = 50)) |>
    x -> NormalData(X = [x[1] x[2]], y = x[3])
chain = train(rng, m, data, predict)

simple_effect_probabilities(chain, predict)
interaction_effect_probabilities(chain, predict)


df = DataFrame(
    x = predict.x,
    y = predict.y,
    f = mean(chain.f)
)

@df df scatter(
    :y,
    :f,
    group = :x,
    m = (0.6, [:+ :xcross :diamond], 8),
    markerstrokealpha = 0.9,
    bg = RGB(0.2, 0.2, 0.2)
)
