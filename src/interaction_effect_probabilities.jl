function interaction_effect_probabilities(ch, data)
    γpost_x = collect(hcat(data.Xunique...)')
    γpost_mins = [minimum(γpost_x[:, i]) for i in 1:size(γpost_x, 2)]
    γpost_x = γpost_x - ones(size(γpost_x, 1)) * (γpost_mins .- 1)'

    ch_gamma_mat = collect(hcat(ch.gamma...)')
    nvar = size(γpost_x, 2)
    p_effects = zeros(nvar * (nvar - 1) ÷ 2)
    var1 = zeros(Int, nvar * (nvar - 1) ÷ 2)
    var2 = zeros(Int, nvar * (nvar - 1) ÷ 2)
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
    return DataFrame(
        var1 = "x" .* string.(var1),
        var2 = "x" .* string.(var2),
        prob = p_effects
    )
end
