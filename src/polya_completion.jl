function polya_completion(m::AbstractDPM; v = 0.01, ϵ = 0.01)
    # Retrieve some info about the clusters
    new_cluster_id = n_clusters(m)
    s = cluster_labels(m)
    N = length(s)
    α = dp_mass(m)

    # Initialize the weights
    λ = (α + N) * log(1 / ϵ)
    M = 1 + quantile(Poisson(λ), 1 - v)
    wvec = zeros(M)
    stick = 1.0
    for m in 1:(M - 1)
        vm = Beta(1, α + N) |> rand
        wm = vm * stick
        wvec[m] = wm
        stick -= wm
    end
    wvec[M] = stick

    # Simulate new cluster labels
    snew = zeros(Int, M)
    for m in 1:M
        if rand() <= alpha / (alpha + N)
            snew[m] = new_cluster_id
            new_cluster_id += 1
        else
            snew[m] = rand(s)
        end
        if snew[m] != m
            wvec[snew[m]] += wvec[m]
            wvec[m] = 0
        end
    end

    # Extract the unique cluster labels and their weights
    snew_eff = unique(snew)
    wvec_eff = wvec[snew_eff]
    wvec_eff ./= sum(wvec_eff)
    return wvec_eff, snew_eff
end

using Distributions;
rng = MersenneTwister(1);
alpha = 1.0;
m = DPM(rng, 1000);
@time polya_completion(m);