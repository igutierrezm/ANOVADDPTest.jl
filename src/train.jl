function train(
    rng, 
    m::AbstractDPM, 
    train, 
    predict; 
    iter = 2000, 
    warmup = iter ÷ 2, 
    thin = 1
)
    gammachain = [zeros(Bool, m.G) for _ = 1:(iter - warmup) ÷ thin]
    fchain = [zeros(length(predict.y)) for _ = 1:(iter - warmup) ÷ thin]
    for t in 1:iter
        update!(rng, m, train)
        if t > warmup && ((t - warmup) % thin == 0)
            t0 = (t - warmup) ÷ thin
            gammachain[t0] = m.gamma[:]
            for i = 1:length(predict.y)
                fchain[t0][i] = predlik(m, train, predict, i)
            end
        end
    end
    return DPM_MCMC_Chain(gammachain, fchain)
end

function predlik(m::AbstractDPM, train, predict, i::Int)
    k̄ = first(passive_clusters(m))
    A = active_clusters(m)
    n = cluster_sizes(m)
    N = length(train.y)
    alpha = dp_mass(m)
    ans = 0.0
    for k in A
        ans += exp(logpredlik(m, train, predict, i, k)) * n[k] / (N + alpha)
    end
    ans += exp(logpredlik(m, train, predict, i, k̄)) * alpha / (N + alpha)
    return(ans)
end
