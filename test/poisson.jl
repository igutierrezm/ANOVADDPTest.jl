@testset "PoissonDDP" begin
    N, G, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    m = PoissonDDP(rng, N, G; K0)
    @test m.G  == 4
    @test m.a0 == 2
    @test m.b0 == 4
    @test m.a1 == [2 * ones(G)]
    @test m.b1 == [4 * ones(G)]
    @test m.γ  == ones(Bool, G)
end

@testset "PoissonDDP inherited accessors" begin
    N, G, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    m = PoissonDDP(rng, N, G; K0)
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
    N, G, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    m = PoissonDDP(rng, N, G; K0)
    ANOVADDPTest.add_cluster!(m)
    @test length(m.a1) == 2
    @test length(m.b1) == 2
    @test length(m.sumlogu) == 2
end

@testset "update_suffstats! (1)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    m = PoissonDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    # TODO: Add tests
end

@testset "update_suffstats! (2)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    m = PoissonDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.update_suffstats!(m, data, 1, 1, 2)
    # TODO: Add tests
end

@testset "logpredlik (empty clusters)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1], [1])
    m = PoissonDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.logpredlik(m, data, 1, first(passive_clusters(m)))
    # TODO: Add tests
end

@testset "logpredlik (non-empty clusters)" begin
    N, G, K0 = 2, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1, 1], [1, 0])
    m = PoissonDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.logpredlik(m, data, 2, 1)
    # TODO: Add some tests
end

@testset "update! (1)" begin
    N, G, K0 = 2, 1, 1
    rng = MersenneTwister(1)
    data = PoissonData([1, 1], [1.0, 0.0])
    m = PoissonDDP(rng, N, G; K0)
    update!(rng, m, data)
end

@testset "update! (2)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(3), N)
    data = PoissonData(x, y)
    m = PoissonDDP(rng, N, G; K0)
    update!(rng, m, data)
end

@testset "train (1)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(3), N)
    data = PoissonData(x, y)
    m = PoissonDDP(rng, N, G; K0)
    γb = train(rng, m, data)
    @test all(mode(γb) .== [true, false, false])
end

@testset "train (2)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(3), N)
    for i = 1:N
        if x[i] == 2
            y[i] += 10
        end
    end
    data = PoissonData(x, y)
    m = PoissonDDP(rng, N, G; K0)
    γb = train(rng, m, data)
    @test all(mode(γb) .== [true, true, false])
end

@testset "train (3)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(3), N)
    for i = 1:N
        if x[i] == 3
            y[i] += 10
        end
    end
    data = PoissonData(x, y)
    m = PoissonDDP(rng, N, G; K0)
    γb = train(rng, m, data)
    @test all(mode(γb) .== [true, false, true])
end

@testset "train (4)" begin
    rng = MersenneTwister(1)
    N, G, K0 = 1000, 3, 1
    x = rand(rng, 1:G, N)
    y = rand(rng, Poisson(3), N)
    for i = 1:N
        if x[i] != 1
            y[i] += 10
        end
    end
    data = PoissonData(x, y)
    m = PoissonDDP(rng, N, G; K0)
    γb = train(rng, m, data)
    @test all(mode(γb) .== [true, true, true])
end