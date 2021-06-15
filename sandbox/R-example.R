# El siguiente código funciona en julia 1.5.3 + R 4.0.4

# Primero, instalemos y carguemos el paquete de R "JuliaConnectoR" 
install.packages("JuliaConnectoR")
library(JuliaConnectoR)

# Verifiquemos si JuliaConnectoR detecta a julia
JuliaConnectoR::juliaSetupOk() # debiese ser igual a TRUE

# Ahora, instalemos los dos paquetes que hacen el trabajo en Julia
juliaEval('Pkg.add(url = "https://github.com/igutierrezm/DPMNeal3.jl")')
juliaEval('Pkg.add(url = "https://github.com/igutierrezm/ANOVADDPTest.jl")')

# Una vez instalados, podemos crear wrappers usando la función juliaImport:
ANOVADDPTest <- juliaImport("ANOVADDPTest")

# La función es train(), que retorna las cadenas para gamma.
# Veamos un ejemplo:

N <- 1000
G <- 3
y <- rnorm(N) # respuestas
x <- as.integer(1 + 0:(N-1) %% G) # grupos
res <- ANOVADDPTest$train(y, x, iter = 2000L)

# Una vez las cadenas, podemos traerlas a R y trabajarlas como es usual
res %>%
  juliaGet() %>%
  purrr::reduce(rbind) %>%
  colMeans()
