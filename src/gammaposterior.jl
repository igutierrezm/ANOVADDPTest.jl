ch
data

iter = length(ch.gamma)
ncat = length(data.Xunique)
nvar = length(data.Xunique[1])
df = zeros(ncat, nvar + 1)

function gammaposterior(chain, data)
    return [hcat(data.Xunique...)' mean(ch.gamma)]
end
