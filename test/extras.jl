using Revise

using ANOVADDPTest
using ANOVADDPTest: dp_mass, n_clusters,  cluster_sizes
using ANOVADDPTest: active_clusters, passive_clusters, cluster_capacity, update!
using Distributions
using SpecialFunctions
using StatsBase: mode
using Statistics: mean, var
using Random
using Test
using CSV
using Tables

X = CSV.File("Xmat.csv") |> Tables.matrix;
y = CSV.File("yvec.csv") |> Tables.matrix;
y = y[:];
ch, data0 = anova_bnp_normal(y, X);
simple_effect_probabilities(ch, data0)