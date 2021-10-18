function simple_effect_probabilities(chain, data)
    γpost_x = collect(hcat(data.Xunique...)')
    γpost_mins = [minimum(γpost_x[:, i]) for i in 1:size(γpost_x, 2)]
    γpost_x = γpost_x - ones(size(γpost_x, 1)) * (γpost_mins .- 1)'

    ch_gamma_mat = collect(hcat(chain.gamma...)')
    nvar = size(γpost_x, 2)
    p_effects = zeros(nvar)
    for i in 1:nvar
        subset = deepcopy(γpost_x)
        subset[:, i] = 2 .- (subset[:, i] .!= 1)
        subset = maximum(subset, dims = 2)[:, 1] .== 1
        if sum(subset) > 0
            p_effects[i] = mean(maximum(ch_gamma_mat[:, subset], dims = 2))
        end
    end
    return DataFrame(var1 = ["x$i" for i in 1:nvar], prob = p_effects)
end
