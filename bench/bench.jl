# Head-to-head benchmark: Model2Vec.jl (native Julia) vs a native Rust model2vec reference
# (bench/rust_ref — no FFI, so this is the fairest possible comparison; same tokenize -> pool ->
# normalize algorithm). Runs both on the SAME byte-identical corpus, batched (matching how
# model2vec-rs and MonsieurPapin's production code call the tokenizer, batch_size=64), using
# @allocated + a min-of-many-runs timer (BenchmarkTools-free, so bench has no extra deps).
#
#   julia --project=. bench/bench.jl [corpus_path] [model_dir]
using Model2Vec
using Printf

corpuspath = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "corpus.txt")
isfile(corpuspath) || error("corpus not found: $corpuspath (run: julia bench/make_corpus.jl)")
texts = readlines(corpuspath)
n = length(texts)

modeldir = if length(ARGS) >= 2
    ARGS[2]
else
    base = joinpath(homedir(), ".cache", "huggingface", "hub", "models--minishlab--potion-multilingual-128M", "snapshots")
    isdir(base) || error("no local model cache; pass a model dir explicitly")
    joinpath(base, first(readdir(base)))
end

model = Model2Vec.load(modeldir)
scratch = Scratch(model)

function juliapass(scratch, model, texts)
    touched = 0
    for text in texts
        v = encode!(scratch, model, text)
        touched += length(v)
    end
    touched
end

function bestns(f, runs)
    f()
    best = typemax(UInt64)
    for _ in 1:runs
        t = time_ns()
        f()
        d = time_ns() - t
        d < best && (best = d)
    end
    best
end

runs = 5
julians = bestns(() -> juliapass(scratch, model, texts), runs)

results = Tuple{String,Float64,Int}[("Julia (Model2Vec.jl)", Float64(julians), 0)]

rustbin = joinpath(@__DIR__, "rust_ref", "target", "release", "m2v_ref")
if isfile(rustbin)
    out = read(`$rustbin $modeldir $corpuspath $runs`, String)
    m = match(r"records=(\d+) min_ns=(\d+)", out)
    m === nothing && error("unexpected m2v_ref output: $out")
    rustns = parse(Float64, m.captures[2])
    push!(results, ("Rust (native, no FFI)", rustns, 0))
else
    @warn "Rust reference not built; skipping (build in bench/rust_ref: cargo build --release)"
end

allocated = GC.@preserve texts (@allocated juliapass(scratch, model, texts))

@printf("\nCorpus: %s (%d records, %s tokenizer)\n\n", basename(corpuspath), n, typeof(model))
@printf("%-26s %10s %12s %8s\n", "implementation", "min (ms)", "records/s", "vs Rust")
println("-"^60)
rustns = length(results) >= 2 ? results[2][2] : NaN
for (label, ns, _) in results
    ratio = isnan(rustns) ? "" : @sprintf("%.2fx", rustns / ns)
    @printf("%-26s %10.3f %12.1f %8s\n", label, ns / 1e6, n / (ns / 1e9), ratio)
end
@printf("\nJulia total allocated over %d records: %d bytes (%.1f B/record)\n", n, allocated, allocated / n)

open(joinpath(@__DIR__, "results.csv"), "w") do io
    println(io, "label,min_ns,records_per_s,ratio_vs_rust")
    for (label, ns, _) in results
        ratio = isnan(rustns) ? "" : @sprintf("%.4f", rustns / ns)
        @printf(io, "%s,%.1f,%.1f,%s\n", label, ns, n / (ns / 1e9), ratio)
    end
end
println("wrote bench/results.csv")
