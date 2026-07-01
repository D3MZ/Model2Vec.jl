using Documenter
using Model2Vec

makedocs(
    sitename = "Model2Vec.jl",
    modules = [Model2Vec],
    authors = "Demetrius Michael",
    repo = Documenter.Remotes.GitHub("D3MZ", "Model2Vec.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://D3MZ.github.io/Model2Vec.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Scope" => "scope.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/D3MZ/Model2Vec.jl",
    devbranch = "main",
    push_preview = false,
)
