using ANOVADDPTest
using SpecialFunctions
using StatsBase: denserank
using Statistics: mean, var
using Random
using Test

@testset "NormalDDP" begin
    rng = MersenneTwister(1)
    N, G, v0, r0, u0, s0 = 10, 4, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; v0, r0, u0, s0)
    @test m.G  == 4
    @test m.v0 == 2
    @test m.r0 == 3
    @test m.u0 == 3.0
    @test m.s0 == 9.0
    @test m.v1 == [2 * ones(G)]
    @test m.r1 == [3 * ones(G)]
    @test m.u1 == [3 * ones(G)]
    @test m.s1 == [9 * ones(G)]
    @test m.γ  == ones(Bool, G)
end

@testset "NormalDDP inherited accessors" begin
    rng = MersenneTwister(1)
    N, G, K0, a0, b0, v0, r0, u0, s0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; K0, a0, b0, v0, r0, u0, s0)
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
    N, G, K0, a0, b0, v0, r0, u0, s0 = 10, 4, 5, 2.0, 4.0, 2.0, 3.0, 3.0, 9.0
    m = NormalDDP(rng, N, G; K0, a0, b0, v0, r0, u0, s0)
    ANOVADDPTest.add_cluster!(m)
    @test length(m.v1) == 2
    @test length(m.r1) == 2
    @test length(m.u1) == 2
    @test length(m.s1) == 2
end

@testset "update_suffstats! (1)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    N, G, K0, v0, r0, u0, s0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test m.v1[1][1] ≈ 2.0
    @test m.v1[2][1] ≈ 1.0
    @test m.r1[1][1] ≈ 2.0
    @test m.r1[2][1] ≈ 1.0
    @test m.u1[1][1] ≈ 0.5
    @test m.u1[2][1] ≈ 0.0
    @test m.s1[1][1] ≈ 1.5
    @test m.s1[2][1] ≈ 1.0
end

@testset "update_suffstats! (2)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    N, G, K0, v0, r0, u0, s0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
    ANOVADDPTest.update_suffstats!(m, data)
    ANOVADDPTest.update_suffstats!(m, data, 1, 1, 2)
    @test m.v1[1][1] ≈ 1.0
    @test m.v1[2][1] ≈ 2.0
    @test m.r1[1][1] ≈ 1.0
    @test m.r1[2][1] ≈ 2.0
    @test m.u1[1][1] ≈ 0.0
    @test m.u1[2][1] ≈ 0.5
    @test m.s1[1][1] ≈ 1.0
    @test m.s1[2][1] ≈ 1.5
end

@testset "logpredlik (empty clusters)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    N, G, K0, v0, r0, u0, s0 = 1, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test ANOVADDPTest.logpredlik(m, data, 1, first(passive_clusters(m))) ≈ (
        0.5 * 1.0 * log(1.0) -
        0.5 * 2.0 * log(1.5) +
        loggamma(2.0 / 2) -
        loggamma(1.0 / 2) +
        0.5 * log(1.0 / 2.0) -
        0.5 * log(π)
    )
end

@testset "logpredlik (non-empty clusters)" begin
    rng = MersenneTwister(1)
    data = Data([1, 1], [1.0, 0.0])
    N, G, K0, v0, r0, u0, s0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
    ANOVADDPTest.update_suffstats!(m, data)
    @test m.v1[1][1] ≈ 3.0
    @test m.r1[1][1] ≈ 3.0
    @test m.u1[1][1] ≈ 1/3
    @test m.s1[1][1] ≈ 5/3

    @test ANOVADDPTest.logpredlik(m, data, 2, 1) ≈ (
        0.5 * 2 * log(1.5) -
        0.5 * 3 * log(5/3) +
        loggamma(3/2) -
        loggamma(2/2) +
        0.5 * log(2/3) -
        0.5 * log(π)
    )
end

@testset "update! (1)" begin
    rng = MersenneTwister(1)
    data = Data([1, 1], [1.0, 0.0])
    N, G, K0, v0, r0, u0, s0 = 2, 1, 1, 1.0, 1.0, 0.0, 1.0
    m = NormalDDP(rng, N, G; K0, v0, r0, u0, s0)
    update!(rng, m, data)
end

# @testset "update! (2)" begin
#     rng = MersenneTwister(1)
#     N, F = 1000, 3
#     y = randn(rng, N)
#     x = [rand(rng, 1:3, F) for _ in 1:N]
#     x = denserank(x)
#     G = length(unique(x))
#     data = Data(x, y)
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
#     data = Data(x, y)
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