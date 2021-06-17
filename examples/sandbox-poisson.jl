using StatsPlots
using ANOVADDPTest
using DataFrames
using Statistics
using Random
using Distributions
gr()
include("examples/utils.jl")

#####
##### Example 1
#####

rng = MersenneTwister(1)
N, G, K0 = 2000, 3, 1
x = rand(rng, 1:G, N)
y = rand(rng, Poisson(4), N)
data = PoissonData(x, y)
predict = 
    expandgrid(1:G, 0:8) |> 
    x -> PoissonData(x...)
m = PoissonDDP(rng, N, G; K0)
chain = train(rng, m, data, predict)

histogram(y)

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
y = rand(rng, Poisson(4), N)
for i = 1:N
    if x[i] == 2
        y[i] = rand(rng, Poisson(1))
    end
end
data = PoissonData(x, y)
predict = 
    expandgrid(1:G, 0:8) |> 
    x -> PoissonData(x...)
m = PoissonDDP(rng, N, G; K0)
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
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = rand(rng, Poisson(4), N)
for i = 1:N
    if x[i] == 3
        y[i] = rand(rng, Poisson(1))
    end
end
data = PoissonData(x, y)
predict = 
    expandgrid(1:G, 0:8) |> 
    x -> PoissonData(x...)
m = PoissonDDP(rng, N, G; K0)
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
N, G, K0 = 1000, 3, 1
x = rand(rng, 1:G, N)
y = rand(rng, Poisson(4), N)
for i = 1:N
    if x[i] != 1
        y[i] = rand(rng, Poisson(1))
    end
end
data = PoissonData(x, y)
predict = 
    expandgrid(1:G, 0:8) |> 
    x -> PoissonData(x...)
m = PoissonDDP(rng, N, G; K0)
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