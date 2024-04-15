function shift_function(Fpost)
    tbl_Fpost1 =
        Fpost |>
        x -> deepcopy(x) |>
        x -> DataFrames.rename!(x, :group => :group1) |>
        x -> DataFrames.rename!(x, :y => :y1) |>
        x -> DataFrames.rename!(x, :F => :F1)

    tbl_Fpost2 =
        Fpost |>
        x -> deepcopy(x) |>
        x -> DataFrames.rename!(x, :group => :group2) |>
        x -> DataFrames.rename!(x, :y => :y2) |>
        x -> DataFrames.rename!(x, :F => :F2)

    tbl_shift =
        tbl_Fpost1 |>
        x -> DataFrames.crossjoin(x, tbl_Fpost2) |>
        x -> DataFrames.sort(x, [:group1, :group2, :F1, :F2]) |>
        x -> DataFrames.filter([:F1, :F2] => (F1, F2) -> F2 >= F1, x) |>
        x -> DataFrames.groupby(x, [:group1, :group2, :y1]) |>
        x -> DataFrames.combine(first, x) |>
        x -> DataFrames.transform(
            x,
            [:y1, :y2] => ((y1, y2) -> y2 - y1) => :shift
        ) |>
        x -> DataFrames.rename(x, :y1 => :y) |>
        x -> DataFrames.filter(
            [:group1, :group2] => (x, y) -> x != y, x
        ) |>
        # x -> DataFrames.filter(:F1 => F1 -> 1e-4 < F1 < 1 - 1e-4, x) |>
        # x -> DataFrames.filter(:F2 => F2 -> 1e-4 < F2 < 1 - 1e-4, x) |>
        x -> select(x, [:group1, :group2, :y, :shift])

    return tbl_shift
end
