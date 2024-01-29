@testset "BerPoiDDP" begin
    N, ngroups, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    model = PoissonDDP(rng, N, ngroups; K0)
    @test model.ngroups  == 4
    @test model.a1 == 2
    @test model.b1 == 4
    @test model.a1_post == [2 * ones(ngroups)]
    @test model.b1_post == [4 * ones(ngroups)]
    @test model.gamma  == ones(Bool, ngroups)
end

@testset "PoissonDDP inherited accessors" begin
    N, ngroups, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    model = PoissonDDP(rng, N, ngroups; K0)
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
    N, ngroups, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    model = PoissonDDP(rng, N, ngroups; K0)
    ANOVADDPTest.add_cluster!(model)
    @test length(model.a1_post) == 2
    @test length(model.b1_post) == 2
    @test length(model.sumlogfacty) == 2
end

@testset "update_suffstats! (1)" begin
    N, ngroups, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    model = PoissonDDP(rng, N, ngroups; K0)
    ANOVADDPTest.update_suffstats!(model, data)
    # TODO: Add tests
end

@testset "update_suffstats! (2)" begin
    N, ngroups, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    model = PoissonDDP(rng, N, ngroups; K0)
    ANOVADDPTest.update_suffstats!(model, data)
    ANOVADDPTest.update_suffstats!(model, data, 1, 1, 2)
    # TODO: Add tests
end

@testset "logpredlik (empty clusters)" begin
    N, ngroups, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    model = PoissonDDP(rng, N, ngroups; K0)
    ANOVADDPTest.update_suffstats!(model, data)
    ANOVADDPTest.logpredlik(model, data, 1, first(passive_clusters(model)))
    # TODO: Add tests
end

@testset "logpredlik (non-empty clusters)" begin
    N, ngroups, K0 = 2, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1, 1], [1, 0])
    model = PoissonDDP(rng, N, ngroups; K0)
    ANOVADDPTest.update_suffstats!(model, data)
    ANOVADDPTest.logpredlik(model, data, 2, 1)
    # TODO: Add some tests
end

@testset "update! (1)" begin
    N, ngroups, K0 = 2, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1, 1], [1, 0])
    model = PoissonDDP(rng, N, ngroups; K0)
    update!(rng, model, data)
end

@testset "update! (2)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = rand(rng, Poisson(2), N)
    data = PoissonData(x, y)
    model = PoissonDDP(rng, N, ngroups; K0)
    update!(rng, model, data)
end

@testset "train (1)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = rand(rng, Poisson(2), N)
    data = PoissonData(x, y)
    predict =
        expandgrid(1:ngroups, 1:8) |>
        x -> PoissonData(x...)
    model = PoissonDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, false, false])
end

@testset "train (2)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = rand(rng, Poisson(2), N)
    for i = 1:N
        if x[i] == 2
            y[i] += 2
        end
    end
    data = PoissonData(x, y)
    predict =
        expandgrid(1:ngroups, 1:8) |>
        x -> PoissonData(x...)
    model = PoissonDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, true, false])
end

@testset "train (3)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = rand(rng, Poisson(2), N)
    for i = 1:N
        if x[i] == 3
            y[i] += 2
        end
    end
    data = PoissonData(x, y)
    predict =
        expandgrid(1:ngroups, 1:8) |>
        x -> PoissonData(x...)
    model = PoissonDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, false, true])
end

@testset "train (4)" begin
    rng = MersenneTwister(1)
    N, ngroups, K0 = 1000, 3, 1
    x = rand(rng, 1:ngroups, N)
    y = rand(rng, Poisson(3), N)
    for i = 1:N
        if x[i] != 1
            y[i] += 2
        end
    end
    data = PoissonData(x, y)
    predict =
        expandgrid(1:ngroups, 1:8) |>
        x -> PoissonData(x...)
    model = PoissonDDP(rng, N, ngroups; K0)
    chain = train(rng, model, data, predict)
    @test all(mode(chain.gamma) .== [true, true, true])
end
