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

function foo(chain, predict)
    Ngroups = length(predict.Xunique)
    Nvars = size(predict.X, 2)
    gamma = hcat(expandgrid([1], [[0, 1] for _ in 1:Ngroups-1]...)...)
    df = DataFrame(:gamma => [gamma[i, :] for i in 1:size(gamma, 1)])
    df1 = deepcopy(df)
    for var in 1:Nvars
        df1[!, "var_$var"] = zeros(Bool, size(df1, 1))
        for h in 1:size(df1, 1)
            for row in 1:Ngroups
                predict.Xunique[row][var] == 1 && continue
                df1[h, "gamma"][row] == 0 && continue
                df1[h, "var_$var"] = true
            end
        end
    end

    for var1 in 1:Nvars, var2 in 1:Nvars
        df1[!, "var_$(var1)_$(var2)"] = zeros(Bool, size(df1, 1))
        for h in 1:size(df1, 1)
            for row in 1:Ngroups
                predict.Xunique[row][var1] == 1 && continue
                predict.Xunique[row][var2] == 1 && continue
                df1[h, "gamma"][row] == 0 && continue
                df1[h, "var_$(var1)_$(var2)"] = true
            end
        end
    end

    unique_gamma = unique(chain.gamma)
    Ngammas = length(unique_gamma)
    Niter = length(chain.gamma)
    gamma_prob = zeros(Ngammas)
    for j in 1:Ngammas
        gamma_prob[j] = sum([chain.gamma[i] == unique_gamma[j] for i in 1:Niter])
    end
    gamma_prob /= sum(gamma_prob)
    probs = DataFrame(:gamma => unique_gamma, :prob => gamma_prob)
    df2 = leftjoin(df1, probs, on = :gamma)
    df2[!, :prob] = coalesce.(df2[!, :prob], 0)
    return df2
end
probs = foo(chain, predict)

# Ejemplo de uso,
sum(probs[!, :prob] .* probs[!, :var_1])
sum(probs[!, :prob] .* probs[!, :var_2])
sum(probs[!, :prob] .* probs[!, :var_1_1])
