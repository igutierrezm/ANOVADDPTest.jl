using ANOVADDPTest
using SpecialFunctions
using StatsBase: denserank
using Statistics: mean, var
using Random
using Test

@testset "SpecificBlock" begin
    G = 4
    sb = SpecificBlock(G; v0 = 2, r0 = 3, u0 = 3.0, s0 = 9.0)
    @test sb.G  == 4
    @test sb.v0 == 2
    @test sb.r0 == 3
    @test sb.u0 == 3.0
    @test sb.s0 == 9.0
    @test sb.v1 == [2 * ones(G)]
    @test sb.r1 == [3 * ones(G)]
    @test sb.u1 == [3 * ones(G)]
    @test sb.s1 == [9 * ones(G)]
    @test sb.γ  == ones(Bool, G)
end

@testset "resize!" begin
    G = 4
    n = 2
    sb = SpecificBlock(G; v0 = 2, r0 = 3, u0 = 3.0, s0 = 9.0)
    ANOVADDPTest.resize!(sb, n)
    @test length(sb.v1) == n
    @test length(sb.r1) == n
    @test length(sb.u1) == n
    @test length(sb.s1) == n
end

@testset "update_suffstats! (1)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    v0, r0, u0, s0 = 1.0, 1.0, 0.0, 1.0
    sb = SpecificBlock(1; v0, r0, u0, s0)
    gb = GenericBlock(rng, 1)
    ANOVADDPTest.update_suffstats!(sb, gb, data)
    @test sb.v1[1][1] ≈ 2.0
    @test sb.v1[2][1] ≈ 1.0
    @test sb.r1[1][1] ≈ 2.0
    @test sb.r1[2][1] ≈ 1.0
    @test sb.u1[1][1] ≈ 0.5
    @test sb.u1[2][1] ≈ 0.0
    @test sb.s1[1][1] ≈ 1.5
    @test sb.s1[2][1] ≈ 1.0
end

@testset "update_suffstats! (2)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    v0, r0, u0, s0 = 1.0, 1.0, 0.0, 1.0
    sb = SpecificBlock(1; v0, r0, u0, s0)
    gb = GenericBlock(rng, 1)
    ANOVADDPTest.update_suffstats!(sb, gb, data)
    ANOVADDPTest.update_suffstats!(sb, gb, data, 1, 1, 2)
    @test sb.v1[1][1] ≈ 1.0
    @test sb.v1[2][1] ≈ 2.0
    @test sb.r1[1][1] ≈ 1.0
    @test sb.r1[2][1] ≈ 2.0
    @test sb.u1[1][1] ≈ 0.0
    @test sb.u1[2][1] ≈ 0.5
    @test sb.s1[1][1] ≈ 1.0
    @test sb.s1[2][1] ≈ 1.5
end

@testset "logpredlik (empty clusters)" begin
    rng = MersenneTwister(1)
    data = Data([1], [1.0])
    v0, r0, u0, s0 = 1.0, 1.0, 0.0, 1.0
    sb = SpecificBlock(1; v0, r0, u0, s0)
    gb = GenericBlock(rng, 1)
    ANOVADDPTest.update_suffstats!(sb, gb, data)
    @test ANOVADDPTest.logpredlik(sb, gb, data, 1, first(gb.P)) ≈ (
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
    v0, r0, u0, s0 = 1.0, 1.0, 0.0, 1.0
    sb = SpecificBlock(1; v0, r0, u0, s0)
    gb = GenericBlock(rng, 2)
    ANOVADDPTest.update_suffstats!(sb, gb, data)
    @test sb.v1[1][1] ≈ 3.0
    @test sb.r1[1][1] ≈ 3.0
    @test sb.u1[1][1] ≈ 1/3
    @test sb.s1[1][1] ≈ 5/3

    @test ANOVADDPTest.logpredlik(sb, gb, data, 2, 1) ≈ (
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
    v0, r0, u0, s0 = 1.0, 1.0, 0.0, 1.0
    sb = SpecificBlock(1; v0, r0, u0, s0)
    gb = GenericBlock(rng, 2)
    update!(rng, sb, gb, data)
end

@testset "update! (2)" begin
    rng = MersenneTwister(1)
    N, F = 1000, 3
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    G = length(unique(x))
    data = Data(x, y)
    sb = SpecificBlock(G)
    gb = GenericBlock(rng, N)
    update!(rng, sb, gb, data)
end

@testset "final_example" begin
    rng = MersenneTwister(1)
    N, F = 1000, 1
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    for i = 1:N
        if x[i] == 2
            y[i] += 10.0
        end
    end
    y .= (y .- mean(y)) ./ √var(y)
    G = length(unique(x))
    data = Data(x, y)
    sb = SpecificBlock(G)
    gb = GenericBlock(rng, N; K0 = 1)
    for t in 1:10
        update!(rng, sb, gb, data)
        println(sb.γ[:])
    end
end

@testset "fit (1)" begin
    rng = MersenneTwister(1)
    N, F = 1000, 1
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    for i = 1:N
        if x[i] == 2
            y[i] += 10.0
        end
    end
    y .= (y .- mean(y)) ./ √var(y)
    γb = mean(fit(y, x; seed = 1))
    @test γb[1] ≈ 1.0
    @test γb[2] ≥ 0.85
    @test γb[3] ≤ 0.15
end

@testset "fit (2)" begin
    rng = MersenneTwister(1)
    N, F = 1000, 1
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    for i = 1:N
        if x[i] == 3
            y[i] += 10.0
        end
    end
    y .= (y .- mean(y)) ./ √var(y)
    γb = mean(fit(y, x; seed = 1))
    @test γb[1] ≈ 1.0
    @test γb[2] ≤ 0.15
    @test γb[3] ≥ 0.85
end

@testset "fit (3)" begin
    rng = MersenneTwister(1)
    N, F = 1000, 1
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    for i = 1:N
        if x[i] != 1
            y[i] += 10.0
        end
    end
    y .= (y .- mean(y)) ./ √var(y)
    γb = mean(fit(y, x; seed = 1))
    @test γb[1] ≈ 1.0
    @test γb[2] ≥ 0.85
    @test γb[3] ≥ 0.85
end

@testset "fit (4)" begin
    rng = MersenneTwister(1)
    N, F = 1000, 1
    y = randn(rng, N)
    x = [rand(rng, 1:3, F) for _ in 1:N]
    x = denserank(x)
    y .= (y .- mean(y)) ./ √var(y)
    γb = mean(fit(y, x; seed = 1))
    @test γb[1] ≈ 1.0
    @test γb[2] ≤ 0.15
    @test γb[3] ≤ 0.15
end