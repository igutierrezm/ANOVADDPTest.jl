function train(
    rng,
    model::AbstractDPM,
    train,
    predict;
    iter = 2000,
    warmup = iter ÷ 2,
    thin = 1
)
    gammachain = [zeros(Bool, model.ngroups) for _ = 1:(iter - warmup) ÷ thin]
    # denchain = [zeros(length(predict.y)) for _ = 1:(iter - warmup) ÷ thin]
    fchain = [zeros(length(predict.y)) for _ = 1:(iter - warmup) ÷ thin]
    Fchain = [zeros(length(predict.y)) for _ = 1:(iter - warmup) ÷ thin]
    for t in 1:iter
        update!(rng, model, train)
        if t > warmup && ((t - warmup) % thin == 0)
            t0 = (t - warmup) ÷ thin
            gammachain[t0] = model.gamma[:]
            for i = 1:length(predict.y)
                fchain[t0][i] = predlik(model, train, predict, i)
                Fchain[t0][i] = predcdf(model, train, predict, i)
            end
            # denchain[t0] .= density(model, predict)
        end
    end
    # return DPM_MCMC_Chain(gammachain, fchain, denchain)
    return DPM_MCMC_Chain(gammachain, fchain, Fchain)
end

function predlik(model::AbstractDPM, train, predict, i::Int)
    k̄ = first(passive_clusters(model))
    A = active_clusters(model)
    n = cluster_sizes(model)
    N = length(train.y)
    alpha = dp_mass(model)
    ans = 0.0
    for k in A
        ans += exp(logpredlik(model, train, predict, i, k)) * n[k] / (N + alpha)
    end
    ans += exp(logpredlik(model, train, predict, i, k̄)) * alpha / (N + alpha)
    return(ans)
end


function predcdf(model::AbstractDPM, train, predict, i::Int)
    k̄ = first(passive_clusters(model))
    A = active_clusters(model)
    n = cluster_sizes(model)
    N = length(train.y)
    alpha = dp_mass(model)
    ans = 0.0
    for k in A
        ans += exp(logpredcdf(model, train, predict, i, k)) * n[k] / (N + alpha)
    end
    ans += exp(logpredcdf(model, train, predict, i, k̄)) * alpha / (N + alpha)
    return(ans)
end
