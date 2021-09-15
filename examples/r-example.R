# install.packages("JuliaConnectoR")
# install.packages("dplyr")
library(JuliaConnectoR)
library(dplyr)
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
%>% juliaGet();

# Compute the group probabilities
group_probs <- fit$group_probs

fit$values[[1]]
