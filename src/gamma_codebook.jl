function gamma_codebook(data0)
    G = length(data0.Xunique);
    D = length(data0.Xunique[1]);
    df = DataFrame(
        [zeros(Int, G) for k in 1:D],
        [Symbol("x$i") for i in 1:D]
    )
    for i in 1:G, j in 1:D
        df[i, j] = data0.Xunique[i][j]
    end
    insertcols!(df, D + 1, :g => 1:G)
    return df
end
