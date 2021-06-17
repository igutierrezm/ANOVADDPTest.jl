using DPMNeal3
using Random

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
@test all(mode(chain.γ) .== [true, false, false])
