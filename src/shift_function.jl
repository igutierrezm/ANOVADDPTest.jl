function shift_function(fpost)
    # Extract f for the treatment and control groups
    tbl_f0 = filter(:group => (x -> x == 1), fpost) # control group
    tbl_f1 = filter(:group => (x -> x != 1), fpost) # treatment group

    # Create a quantile function
    v = tbl_f0.y
    f = tbl_f0.f
    w = aweights(f / sum(f))
    custom_quantile(x) = quantile(v, w, x)

    # Compute the shift function for each treatment group
    tbl_shift =
        tbl_f1 |>
        x -> sort(x, [:group, :y]) |>
        x -> groupby(x, :group) |>
        x -> transform(x, :f => (x -> cumsum(x)) => :F; ungroup = false) |>
        x -> transform(x, :F => (x -> x / maximum(x)) => :F; ungroup = false) |>
        x -> transform(x, :F => (x -> custom_quantile(x)) => :shift) |>
        # x -> transform(x, [:shift, :y] => ((x, y) -> x - y) => :shift) |>
        x -> select(x, [:group, :y, :shift])
    return tbl_shift
end
