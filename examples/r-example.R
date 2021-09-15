# Setup =======================================================================

# Install some useful R packages
install.packages("dplyr")
install.packages("ggplot2")
install.packages("JuliaConnectoR")
library(dplyr)
library(ggplot2)
library(JuliaConnectoR)
juliaSetupOk()

# Install ANOVADDPTest.jl
juliaEval('import Pkg')
juliaEval('Pkg.add(url = "https://github.com/igutierrezm/ANOVADDPTest.jl")')

# Load ANOVADDPTest.jl
ANOVADDPTest <- juliaImport("ANOVADDPTest")

# An utility function: Simulate a test sample
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

# Body ========================================================================

# Create a test sample
data <- simulate_sample_normal(1, 1000)
y <- data$y # a response vector
X <- data$X # a design matrix

# Fit the model (much easier than before, I think)
fit <- ANOVADDPTest$anova_bnp_normal(y, X)

# Retrieve the posterior probability of each group
(group_probs <- ANOVADDPTest$group_probs(fit) %>% as.data.frame())

# Retrieve the meaning of each group
# (e.g. what does group2 means in terms of the factors?)
(group_codes <- ANOVADDPTest$group_codes(fit) %>% as.data.frame())

# Retrieve the posterior predictive density
(fpost <- ANOVADDPTest$fpost(fit) %>% as.data.frame())

# Plot the posterior predictive density
p <- ggplot(fpost, aes(x = y, y = f, color = factor(group))) + geom_line()
ggsave("fig1.png", p)
# Should we make this a function? I don't know.
# It is too simple, but once the the plot is done, is very difficult to modify.
