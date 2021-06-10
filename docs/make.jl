using ANOVADDPTest
using Documenter

DocMeta.setdocmeta!(ANOVADDPTest, :DocTestSetup, :(using ANOVADDPTest); recursive=true)

makedocs(;
    modules=[ANOVADDPTest],
    authors="Iván Gutiérrez <ivangutierrez1988@gmail.com> and contributors",
    repo="https://github.com/igutierrezm/ANOVADDPTest.jl/blob/{commit}{path}#{line}",
    sitename="ANOVADDPTest.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://igutierrezm.github.io/ANOVADDPTest.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/igutierrezm/ANOVADDPTest.jl",
)
