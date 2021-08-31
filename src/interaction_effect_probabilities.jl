function interaction_effect_probabilities(ch, data)
    γpost_x = collect(hcat(data.Xunique...)')
    ch_gamma_mat = collect(hcat(ch.gamma...)')
    nvar = size(γpost_x, 2)
    p_effects = zeros(nvar * (nvar - 1) ÷ 2)
    var1 = zeros(nvar * (nvar - 1) ÷ 2)
    var2 = zeros(nvar * (nvar - 1) ÷ 2)
    row = 1
    for i in 1:(nvar - 1), j = (i + 1):nvar
        subset = deepcopy(γpost_x)
        subset[:, i] = 2 .- (subset[:, i] .!= 1)
        subset[:, j] = 2 .- (subset[:, j] .!= 1)
        subset = maximum(subset, dims = 2)[:, 1] .== 1
        p_effects[row] = mean(maximum(ch_gamma_mat[:, subset], dims = 2))
        var1[row] = i
        var2[row] = j
        row += 1
    end
    return DataFrame(var1 = var1, var2 = var2, prob = p_effects)
end
