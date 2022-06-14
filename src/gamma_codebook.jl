function gamma_codebook(data0)
    ngroups = length(data0.Xunique);
    D = length(data0.Xunique[1]);
    df = DataFrame(
        [zeros(Int, ngroups) for k in 1:D],
        [Symbol("x$i") for i in 1:D]
    )
    for i in 1:ngroups, j in 1:D
        df[i, j] = data0.Xunique[i][j]
    end
    insertcols!(df, 1, :group => 1:ngroups)
    return df
end
