function interaction_effect_probabilities(ch, data)
    γpost_x = collect(hcat(data.Xunique...)')
    ch_gamma_mat = collect(hcat(ch.gamma...)')
    nvar = size(γpost_x, 2)
    p_effects = zeros(nvar * (nvar - 1) ÷ 2)
    var1 = zeros(nvar * (nvar - 1) ÷ 2)
    var2 = zeros(nvar * (nvar - 1) ÷ 2)
    row = 1
    for i in 1:(nvar - 1), j = (i + 1):nvar
        subset = (γpost_x[:, i] .!= γpost_x[1, i]) .* (γpost_x[:, j] .!= γpost_x[1, j])
        p_effects[row] = mean(minimum(ch_gamma_mat[:, subset], dims = 2))
        var1[row] = i
        var2[row] = j
        row += 1
    end
    return DataFrame(var1 = var1, var2 = var2, prob = p_effects)
end

# function interaction_effect_probabilities(chain, predict)
#     Ngroups = length(predict.Xunique)
#     Nvars = size(predict.X, 2)
#     gamma = hcat(expandgrid([1], [[0, 1] for _ in 1:Ngroups-1]...)...)
#     df = DataFrame(:gamma => [gamma[i, :] for i in 1:size(gamma, 1)])
#     df1 = deepcopy(df)
#     for var1 in 1:Nvars, var2 in 1:Nvars
#         var1 ≥ var2 && continue
#         df1[!, "var_$(var1)_$(var2)"] = zeros(Bool, size(df1, 1))
#         for h in 1:size(df1, 1)
#             for row in 1:Ngroups
#                 predict.Xunique[row][var1] == 1 && continue
#                 predict.Xunique[row][var2] == 1 && continue
#                 df1[h, "gamma"][row] == 0 && continue
#                 df1[h, "var_$(var1)_$(var2)"] = true
#             end
#         end
#     end

#     unique_gamma = unique(chain.gamma)
#     Ngammas = length(unique_gamma)
#     Niter = length(chain.gamma)
#     gamma_prob = zeros(Ngammas)
#     for j in 1:Ngammas
#         gamma_prob[j] = sum([chain.gamma[i] == unique_gamma[j] for i in 1:Niter])
#     end
#     gamma_prob /= sum(gamma_prob)
#     probs = DataFrame(:gamma => unique_gamma, :prob => gamma_prob)
#     df2 = leftjoin(df1, probs, on = :gamma)
#     df2[!, :prob] = coalesce.(df2[!, :prob], 0)

#     df3 = df2[!, Not([:gamma, :prob])]
#     df4 = combine(df3, names(df3) .=> x -> sum(x .* df2.prob), renamecols = false)
#     df5 = stack(df4)
#     return df5
# end
