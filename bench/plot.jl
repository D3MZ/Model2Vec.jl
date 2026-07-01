# Dependency-free SVG plot of bench/results.csv: Julia vs Rust throughput, one panel per model
# (WordPiece / Unigram), mirroring AhoCorasickILP.jl's bench/plot.jl style.
rows = readlines(joinpath(@__DIR__, "results.csv"))[2:end]
models = String[]; tokenizers = String[]; labels = String[]; rps = Float64[]
for r in rows
    isempty(r) && continue
    f = split(r, ',')
    push!(models, f[1]); push!(tokenizers, f[2]); push!(labels, f[3]); push!(rps, parse(Float64, f[5]))
end

uniquemodels = unique(models)
n = length(uniquemodels)

W, H = 900, 420
padL, padR, padT, padB = 70, 30, 60, 90
pw = (W - padL - padR)
panelW = n == 1 ? pw : (pw - 40 * (n - 1)) ÷ n
h = H - padT - padB

io = IOBuffer()
println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" font-family="Helvetica,Arial,sans-serif">""")
println(io, """<rect width="$W" height="$H" fill="white"/>""")
println(io, """<text x="$(W÷2)" y="26" text-anchor="middle" font-size="18" font-weight="bold">Model2Vec.jl (native Julia) vs a native Rust reference — single thread</text>""")

for (panel, model) in enumerate(uniquemodels)
    x0 = padL + (panel - 1) * (panelW + 40)
    idx = findall(==(model), models)
    tokenizer = tokenizers[first(idx)]
    vals = rps[idx]
    labs = labels[idx]
    ymax = maximum(vals) * 1.25

    println(io, """<text x="$(x0 + panelW÷2)" y="50" text-anchor="middle" font-size="14" font-weight="bold">$model ($tokenizer)</text>""")
    println(io, """<rect x="$x0" y="$padT" width="$panelW" height="$h" fill="#fafafa" stroke="#ccc"/>""")

    barw = panelW ÷ (2 * length(vals))
    gap = panelW ÷ (length(vals) + 1)
    colors = ["#1f7a4d", "#d1495b"]
    for (i, v) in enumerate(vals)
        bx = x0 + i * gap - barw ÷ 2
        bh = round(Int, v / ymax * h)
        by = padT + h - bh
        color = occursin("Julia", labs[i]) ? colors[1] : colors[2]
        println(io, """<rect x="$bx" y="$by" width="$barw" height="$bh" fill="$color"/>""")
        println(io, """<text x="$(bx + barw÷2)" y="$(by - 6)" text-anchor="middle" font-size="11" fill="#333">$(round(Int, v))</text>""")
        println(io, """<text x="$(bx + barw÷2)" y="$(padT+h+16)" text-anchor="middle" font-size="10" fill="#333">$(occursin("Julia", labs[i]) ? "Julia" : "Rust")</text>""")
    end
    println(io, """<text x="$(x0+8)" y="$(padT+14)" font-size="11" fill="#555">records/s (higher = better)</text>""")
end

lx = padL
println(io, """<rect x="$lx" y="$(H-24)" width="14" height="14" fill="#1f7a4d"/><text x="$(lx+20)" y="$(H-12)" font-size="13">Model2Vec.jl (native Julia)</text>""")
println(io, """<rect x="$(lx+240)" y="$(H-24)" width="14" height="14" fill="#d1495b"/><text x="$(lx+260)" y="$(H-12)" font-size="13">Rust model2vec reference (native, no FFI)</text>""")

println(io, "</svg>")
write(joinpath(@__DIR__, "benchmark.svg"), String(take!(io)))
println("wrote bench/benchmark.svg")
