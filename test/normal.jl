@testset "NormalDDP" begin
    rng = MersenneTwister(1)
    N, G, a0, lambda0, mu0, b0 = 10, 4, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; mu0, lambda0, a0, b0)
    @test m.G  == 4
    @test m.a0 == 2
    @test m.lambda0 == 3
    @test m.mu0 == 3.0
    @test m.b0 == 9.0
    @test m.a0_post == [2 * ones(G)]
    @test m.lambda0_post == [3 * ones(G)]
    @test m.mu0_post == [3 * ones(G)]
    @test m.b0_post == [9 * ones(G)]
    @test m.gamma  == ones(Bool, G)
end

@testset "NormalDDP inherited accessors" begin
    rng = MersenneTwister(1)
    N, G, K0, a, b, a0, lambda0, mu0, b0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; K0, a, b, mu0, lambda0, a0, b0)
    @test dp_mass(m) > 0.0
    @test dp_mass(m) < Inf
    @test n_clusters(m) == K0
    @test sum(cluster_sizes(m) .== 0) == 1
    @test sum(cluster_sizes(m)) == N
    @test active_clusters(m) == Set(1:K0)
    @test passive_clusters(m) == Set(K0 + 1)
    @test cluster_capacity(m) == K0 + 1
end

@testset "add_cluster!" begin
    rng = MersenneTwister(1)
    N, G, K0, a, b, a0, lambda0, mu0, b0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; K0, a, b, mu0, lambda0, a0, b0)
    ANOVADDPTest.add_cluster!(m)
    @test length(m.a0_post) == 2
    @test length(m.lambda0_post) == 2
    @test length(m.mu0_post) == 2
    @test length(m.b0_post) == 2
end

@testset "update_suffstats! (1)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, G, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test m.a0_post[1][1] ≈ 1.5
    @test m.a0_post[2][1] ≈ 1.0
    @test m.lambda0_post[1][1] ≈ 2.0
    @test m.lambda0_post[2][1] ≈ 1.0
    @test m.mu0_post[1][1] ≈ 0.5
    @test m.mu0_post[2][1] ≈ 0.0
    @test m.b0_post[1][1] ≈ 1.25
    @test m.b0_post[2][1] ≈ 1.0
end

@testset "update_suffstats! (2)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, G, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.update_suffstats!(m, data, 1, 1, 2)
    @test m.a0_post[1][1] ≈ 1.0
    @test m.a0_post[2][1] ≈ 1.5
    @test m.lambda0_post[1][1] ≈ 1.0
    @test m.lambda0_post[2][1] ≈ 2.0
    @test m.mu0_post[1][1] ≈ 0.0
    @test m.mu0_post[2][1] ≈ 0.5
    @test m.b0_post[1][1] ≈ 1.0
    @test m.b0_post[2][1] ≈ 1.25
end

@testset "logpredlik (empty clusters)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, G, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test ANOVADDPTest.logpredlik(m, data, 1, first(passive_clusters(m))) ≈ (
        1.0 * log(1.0) -
        1.5 * log(1.25) +
        loggamma(1.5) -
        loggamma(1.0) +
        0.5 * log(1.0 / 2.0) -
        0.5 * log(2π)
    )
end

@testset "logpredlik (non-empty clusters)" begin
    rng = MersenneTwister(1)
    data = NormalData([1, 1], [1.0, 0.0])
    N, G, K0, a0, lambda0, mu0, b0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test m.a0_post[1][1] ≈ 2.0
    @test m.lambda0_post[1][1] ≈ 3.0
    @test m.mu0_post[1][1] ≈ 1/3
    @test m.b0_post[1][1] ≈ 4/3
    @test ANOVADDPTest.logpredlik(m, data, 2, 1) ≈ (
        1.5 * log(1.25) -
        2.0 * log(4/3) +
        loggamma(2.0) -
        loggamma(1.5) +
        0.5 * log(2/3) -
        0.5 * log(2π)
    )
end

@testset "update! (1)" begin
    rng = MersenneTwister(1)
    data = NormalData([1, 1], [1.0, 0.0])
    N, G, K0, a0, lambda0, mu0, b0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, mu0, lambda0, a0, b0)
    update!(rng, m, data)
end

@testset "update! (2)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = randn(rng, N)
    data = NormalData(x, y)
    m = NormalDDP(rng, N, G; K0)
    update!(rng, m, data)
end

@testset "final_example" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = randn(rng, N)
    data = NormalData(x, y)
    m = NormalDDP(rng, N, G; K0)
    for t in 1:10
        update!(rng, m, data)
    end
end

@testset "train (1)" begin
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
    @test all(mode(chain.gamma) .== [true, false, false])
end

@testset "train (2)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] == 2
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    m = NormalDDP(rng, N, G; K0)
    chain = train(rng, m, data, predict)
    @test all(mode(chain.gamma) .== [true, true, false])
end

@testset "train (3)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] == 3
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    m = NormalDDP(rng, N, G; K0)
    chain = train(rng, m, data, predict)
    @test all(mode(chain.gamma) .== [true, false, true])
end

@testset "train (4)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] != 1
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:G, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    m = NormalDDP(rng, N, G; K0)
    chain = train(rng, m, data, predict)
    @test all(mode(chain.gamma) .== [true, true, true])
end

## Note:
## Remember that you can use denserank(x) for converting
## vector codes to numeric codes
