function gamma_posterior(chain)
    gamma_rank = denserank(chain.gamma)
    gamma_unique = sort(unique(chain.gamma))
    gamma_card = length(unique(gamma_unique))
    gamma_freq = counts(gamma_rank, gamma_card)
    pgamma = gamma_freq ./ length(chain.gamma)
    N = length(gamma_unique)
    G = length(gamma_unique[1])
    df = DataFrame(
        [zeros(Bool, N) for k in 2:G],
        [Symbol("group$i") for i in 2:G]
    )
    for i in 1:N, j in 2:G
        df[i, j - 1] = gamma_unique[i][j]
    end
    df[!, :prob] = pgamma
    return df
end
