function gamma_chain(chain)
    nsims = length(chain.gamma)
    ngroups = length(chain.gamma[1])
    df = DataFrame(
        [zeros(Bool, nsims) for k in 2:ngroups],
        [Symbol("gamma$i") for i in 2:ngroups]
    )
    for i in 1:nsims, j in 2:ngroups
        df[i, j - 1] = chain.gamma[i][j]
    end
    df.iter = 1:nsims
    out = DataFrames.stack(df, DataFrames.Not(:iter))
    return out
end
