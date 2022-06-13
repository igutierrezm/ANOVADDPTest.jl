function interaction_effect_probabilities(ch, data)
    gammapost_x = collect(hcat(data.Xunique...)')
    gammapost_mins = [minimum(gammapost_x[:, i]) for i in 1:size(gammapost_x, 2)]
    gammapost_x = gammapost_x - ones(size(gammapost_x, 1)) * (gammapost_mins .- 1)'

    ch_gamma_mat = collect(hcat(ch.gamma...)')
    nvar = size(gammapost_x, 2)
    p_effects = zeros(nvar * (nvar - 1) ÷ 2)
    var1 = zeros(Int, nvar * (nvar - 1) ÷ 2)
    var2 = zeros(Int, nvar * (nvar - 1) ÷ 2)
    row = 1
    for i in 1:(nvar - 1), j = (i + 1):nvar
        subset = deepcopy(gammapost_x)
        subset[:, i] = 2 .- (subset[:, i] .!= 1)
        subset[:, j] = 2 .- (subset[:, j] .!= 1)
        subset = maximum(subset, dims = 2)[:, 1] .== 1
        if sum(subset) > 0
            p_effects[row] = mean(maximum(ch_gamma_mat[:, subset], dims = 2))
            var1[row] = i
            var2[row] = j
            row += 1
        end
    end
    return DataFrame(
        var1 = "x" .* string.(var1),
        var2 = "x" .* string.(var2),
        prob = p_effects
    )[1:(row - 1), :]
end
