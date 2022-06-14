@testset "NormalDDP" begin
    rng = MersenneTwister(1)
    N, ngroups, a0, lambda0, mu0, b0 = 10, 4, 2.0, 3.0, 3.0, 9.0
    model = NormalDDP(rng, N, ngroups; mu0, lambda0, a0, b0)
    @test model.ngroups  == 4
    @test model.a0 == 2
    @test model.lambda0 == 3
    @test model.mu0 == 3.0
    @test model.b0 == 9.0
    @test model.a0_post == [2 * ones(ngroups)]
    @test model.lambda0_post == [3 * ones(ngroups)]
    @test model.mu0_post == [3 * ones(ngroups)]
    @test model.b0_post == [9 * ones(ngroups)]
    @test model.gamma  == ones(Bool, ngroups)
end

@testset "NormalDDP inherited accessors" begin
    rng = MersenneTwister(1)
    N, ngroups, K0, a, b, a0, lambda0, mu0, b0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    model = NormalDDP(rng, N, ngroups; K0, a, b, mu0, lambda0, a0, b0)
    @test dp_mass(model) > 0.0
    @test dp_mass(model) < Inf
    @test n_clusters(model) == K0
    @test sum(cluster_sizes(model) .== 0) == 1
    @test sum(cluster_sizes(model)) == N
    @test active_clusters(model) == Set(1:K0)
    @test passive_clusters(model) == Set(K0 + 1)
    @test cluster_capacity(model) == K0 + 1
end

@testset "add_cluster!" begin
    rng = MersenneTwister(1)
    N, ngroups, K0, a, b, a0, lambda0, mu0, b0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    model = NormalDDP(rng, N, ngroups; K0, a, b, mu0, lambda0, a0, b0)
    ANOVADDPTest.add_cluster!(model)
    @test length(model.a0_post) == 2
    @test length(model.lambda0_post) == 2
    @test length(model.mu0_post) == 2
    @test length(model.b0_post) == 2
end

@testset "update_suffstats! (1)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, ngroups, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    model = NormalDDP(rng, N, ngroups; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(model, data)
    @test model.a0_post[1][1] ≈ 1.5
    @test model.a0_post[2][1] ≈ 1.0
    @test model.lambda0_post[1][1] ≈ 2.0
    @test model.lambda0_post[2][1] ≈ 1.0
    @test model.mu0_post[1][1] ≈ 0.5
    @test model.mu0_post[2][1] ≈ 0.0
    @test model.b0_post[1][1] ≈ 1.25
    @test model.b0_post[2][1] ≈ 1.0
end

@testset "update_suffstats! (2)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, ngroups, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    model = NormalDDP(rng, N, ngroups; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(model, data)
    ANOVADDPTest.update_suffstats!(model, data, 1, 1, 2)
    @test model.a0_post[1][1] ≈ 1.0
    @test model.a0_post[2][1] ≈ 1.5
    @test model.lambda0_post[1][1] ≈ 1.0
    @test model.lambda0_post[2][1] ≈ 2.0
    @test model.mu0_post[1][1] ≈ 0.0
    @test model.mu0_post[2][1] ≈ 0.5
    @test model.b0_post[1][1] ≈ 1.0
    @test model.b0_post[2][1] ≈ 1.25
end

@testset "logpredlik (empty clusters)" begin
    rng = MersenneTwister(1)
    data = NormalData([1], [1.0])
    N, ngroups, K0, a0, lambda0, mu0, b0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    model = NormalDDP(rng, N, ngroups; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(model, data)
    @test ANOVADDPTest.logpredlik(model, data, 1, first(passive_clusters(model))) ≈ (
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
    N, ngroups, K0, a0, lambda0, mu0, b0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    model = NormalDDP(rng, N, ngroups; K0, mu0, lambda0, a0, b0)
    ANOVADDPTest.update_suffstats!(model, data)
    @test model.a0_post[1][1] ≈ 2.0
    @test model.lambda0_post[1][1] ≈ 3.0
    @test model.mu0_post[1][1] ≈ 1/3
    @test model.b0_post[1][1] ≈ 4/3
    @test ANOVADDPTest.logpredlik(model, data, 2, 1) ≈ (
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
    N, ngroups, K0, a0, lambda0, mu0, b0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    model = NormalDDP(rng, N, ngroups; K0, mu0, lambda0, a0, b0)
    update!(rng, model, data)
end

@testset "update! (2)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    data = NormalData(x, y)
    model = NormalDDP(rng, N, ngroups; K0)
    update!(rng, model, data)
end

@testset "final_example" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    data = NormalData(x, y)
    model = NormalDDP(rng, N, ngroups; K0)
    for t in 1:10
        update!(rng, model, data)
    end
end

@testset "train (1)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    model = NormalDDP(rng, N, ngroups; K0)
    data = NormalData(x, y)
    predict =
        expandgrid(1:ngroups, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, false, false])
end

@testset "train (2)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] == 2
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:ngroups, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    model = NormalDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, true, false])
end

@testset "train (3)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] == 3
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:ngroups, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    model = NormalDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, false, true])
end

@testset "train (4)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = randn(rng, N)
    for i = 1:N
        if x[i] != 1
            y[i] += 0.5
        end
    end
    data = NormalData(x, y)
    predict =
        expandgrid(1:ngroups, range(-2.0, stop = 2.0, length = 50)) |>
        x -> NormalData(x...)
    model = NormalDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, true, true])
end

## Note:
## Remember that you can use denserank(x) for converting
## vector codes to numeric codes
