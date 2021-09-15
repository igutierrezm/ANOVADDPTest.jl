install.packages("dplyr")
install.packages("ggplot2")
install.packages("JuliaConnectoR")
library(dplyr)
library(ggplot2)
library(JuliaConnectoR)
juliaSetupOk()

# Install ANOVADDPTest.jl
juliaEval('
    import Pkg;
    Pkg.add(url = "https://github.com/igutierrezm/ANOVADDPTest.jl");
')

# Load ANOVADDPTest
ANOVADDPTest <- juliaImport("ANOVADDPTest")

# Simulate a sample
simulate_sample_normal <- function(rseed, N) {
    set.seed(rseed)
    X <- sample(1:2, N * 2, replace = TRUE) %>% matrix(ncol = 2)
    y <- rnorm(N)
    for (i in 1:N) {
        if ((X[i, 1] == 2) & (X[i, 2] == 2)) {
            y[i] = y[i] + 1
        }
    }
    list(y = y, X = X)
}
data <- simulate_sample_normal(1, 1000)

# Fit the model
fit <- ANOVADDPTest$anova_bnp_normal(data$y, data$X)

# Retrieve the posterior probability of each group
group_probs <- ANOVADDPTest$group_probs(fit) %>% as.data.frame()
group_probs

# Retrieve the meaning of each group
group_codes <- ANOVADDPTest$group_codes(fit) %>% as.data.frame()
group_codes

# Retrieve the posterior predictive density
fpost <- ANOVADDPTest$fpost(fit) %>% as.data.frame()
fpost

# Plot the posterior predictive density
p <- ggplot(fpost, aes(x = y, y = f, color = factor(group))) + geom_line()
ggsave("fig1.png", p)
