# using CategoricalArrays
using StatsPlots
using ANOVADDPTest
using DataFrames
using Statistics
using Random
gr()
include("examples/utils.jl")

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
