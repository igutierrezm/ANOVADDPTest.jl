using Revise
using ANOVADDPTest
using DataFrames
using Distributions
using Statistics
using Random 
using AlgebraOfGraphics, CairoMakie

Random.seed!(1)
X = ones(Int, 1000, 1)
y = 1 .+ 2 * randn(1000)

fitted_model = anova_bnp_normal(y, X; standardize_y = true);
f = fpost(fitted_model)
f.ftrue = pdf.(Normal(1, 2), f.y)
f = stack(f, [:f, :ftrue])

p = data(f) * mapping(:y, :value) * visual(Lines) * mapping(color = :variable)
draw(p)