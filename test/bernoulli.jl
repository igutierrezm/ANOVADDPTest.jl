@testset "BernoulliDDP" begin
    N, G, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    m = BernoulliDDP(rng, N, G; K0)
    @test m.G  == 4
    @test m.a0 == 2.0
    @test m.b0 == 4.0
    @test m.a1 == [2 * ones(G)]
    @test m.b1 == [4 * ones(G)]
    @test m.γ  == ones(Bool, G)
end

@testset "Bernoulli inherited accessors" begin
    N, G, K0 = 10, 4, 1
    rng = MersenneTwister(1)
    m = BernoulliDDP(rng, N, G; K0)
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
    m = BernoulliDDP(rng, N, G; K0)
    ANOVADDPTest.add_cluster!(m)
    @test length(m.a1) == 2
    @test length(m.b1) == 2
end

@testset "update_suffstats! (1)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = BernoulliData([1], [1])
    m = BernoulliDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    # TODO: Add tests
end

@testset "update_suffstats! (2)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = BernoulliData([1], [1])
    m = BernoulliDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.update_suffstats!(m, data, 1, 1, 2)
    # TODO: Add tests
end

@testset "logpredlik (empty clusters)" begin
    N, G, K0 = 1, 1, 1
    rng = MersenneTwister(1)
    data = BernoulliData([1], [1])
    m = BernoulliDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.logpredlik(m, data, 1, first(passive_clusters(m)))
    # TODO: Add tests
end

@testset "logpredlik (non-empty clusters)" begin
    N, G, K0 = 2, 1, 1
    rng = MersenneTwister(1)
    data = BernoulliData([1, 1], [1, 0])
    m = BernoulliDDP(rng, N, G; K0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.logpredlik(m, data, 2, 1)
    # TODO: Add some tests
end

# @testset "update! (1)" begin
#     rng = MersenneTwister(1)
#     data = NormalData([1, 1], [1.0, 0.0])
#     N, G, K0, v0, r0, u0, s0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
#     m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
#     update!(rng, m, data)
# end

# @testset "update! (2)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 3
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     G = length(unique(x))
#     data = NormalData(x, y)
#     sb = SpecificBlock(G)
#     gb = GenericBlock(rng, N)
#     update!(rng, sb, gb, data)
# end

# @testset "final_example" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 1
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     for i = 1:N
#         if x[i] == 2
#             y[i] += 10.0
#         end
#     end
#     y .= (y .- mean(y)) ./ √var(y)
#     G = length(unique(x))
#     data = NormalData(x, y)
#     sb = SpecificBlock(G)
#     gb = GenericBlock(rng, N; K0 = 1)
#     for t in 1:10
#         update!(rng, sb, gb, data)
#         println(sb.γ[:])
#     end
# end

# @testset "fit (1)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 1
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     for i = 1:N
#         if x[i] == 2
#             y[i] += 10.0
#         end
#     end
#     y .= (y .- mean(y)) ./ √var(y)
#     γb = mean(fit(y, x; seed = 1))
#     @test γb[1] ≈ 1.0
#     @test γb[2] ≥ 0.85
#     @test γb[3] ≤ 0.15
# end

# @testset "fit (2)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 1
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     for i = 1:N
#         if x[i] == 3
#             y[i] += 10.0
#         end
#     end
#     y .= (y .- mean(y)) ./ √var(y)
#     γb = mean(fit(y, x; seed = 1))
#     @test γb[1] ≈ 1.0
#     @test γb[2] ≤ 0.15
#     @test γb[3] ≥ 0.85
# end

# @testset "fit (3)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 1
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     for i = 1:N
#         if x[i] != 1
#             y[i] += 10.0
#         end
#     end
#     y .= (y .- mean(y)) ./ √var(y)
#     γb = mean(fit(y, x; seed = 1))
#     @test γb[1] ≈ 1.0
#     @test γb[2] ≥ 0.85
#     @test γb[3] ≥ 0.85
# end

# @testset "fit (4)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 1
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     y .= (y .- mean(y)) ./ √var(y)
#     γb = mean(fit(y, x; seed = 1))
#     @test γb[1] ≈ 1.0
#     @test γb[2] ≤ 0.15
#     @test γb[3] ≤ 0.15
# end