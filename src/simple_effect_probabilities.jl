function simple_effect_probabilities(chain, data)
    γpost_x = collect(hcat(data.Xunique...)')
    ch_gamma_mat = collect(hcat(chain.gamma...)')
    nvar = size(γpost_x, 2)
    p_effects = zeros(nvar)
    for i in 1:nvar
        subset = γpost_x[:, i] .!= γpost_x[1, i]
        p_effects[i] = mean(minimum(ch_gamma_mat[:, subset], dims = 2))
    end
    return DataFrame(var1 = 1:nvar, prob = p_effects)
end
