"""
    Model2Vec

Native-Julia inference for [model2vec](https://github.com/MinishLab/model2vec) static
embedding models: **allocation-free** for both `WordPiece`- and `Unigram`-tokenized models,
and faster than an in-process Rust FFI bridge running the same algorithm.

A model2vec model is tokenize -> look up a per-token embedding row -> mean-pool -> optionally
L2-normalize. The tokenizer is the expensive part; this package implements both tokenizer
families model2vec models ship with directly in Julia (byte-level, no FFI):

  * **WordPiece** (BERT-style: e.g. `minishlab/potion-base-8M`) — greedy longest-match per word.
  * **Unigram** (SentencePiece-style: e.g. `minishlab/potion-multilingual-128M`) — Viterbi
    segmentation over a byte-trie of the vocabulary.

[`load`](@ref) auto-detects which one a model uses from its `tokenizer.json` and returns the
matching model type; [`encode`](@ref)/[`encode!`](@ref) work the same way for either.

# Example
```julia
model = Model2Vec.load(modeldir)          # a local model2vec snapshot directory
v = Model2Vec.encode(model, "cat dog")    # Vector{Float32}, length == model.dim

scratch = Model2Vec.Scratch(model)        # reusable buffers for the zero/low-alloc hot path
v2 = Model2Vec.encode!(scratch, model, "another sentence")
```

# Scope
WordPiece's case-folding and whitespace/punctuation handling are byte-level ASCII; its accent
stripping is approximated, not implemented byte-for-byte — see `README.md` for the measured
correctness impact on non-ASCII text. Unigram's SentencePiece `Precompiled` charsmap *is*
implemented byte-for-byte (a darts double-array trie decoded straight from tokenizer.json);
see the scope note atop `src/unigram.jl` for the two residual edge-case divergences.
"""
module Model2Vec

using JSON, StringViews, Base64

export load, encode, encode!, Scratch

abstract type StaticModel end

include("safetensors.jl")
include("wordpiece.jl")
include("unigram.jl")

# Populates lookup tables built from a top-level function call (not a literal) -- __init__ runs
# on every module load (including from a precompiled cache), unlike top-level `const X = f()`,
# which only executes once during precompilation and would otherwise be invisible to per-test-run
# code coverage. See PUNCTLUT's definition in unigram.jl.
function __init__()
    PUNCTLUT[] = buildpunctlut()
    nothing
end

"""
    load(dir::AbstractString) -> StaticModel

Load a model2vec snapshot directory (must contain `tokenizer.json`, `model.safetensors`,
`config.json` — the standard HuggingFace `snapshot_download`/`hf-hub` layout). Returns a
`WordPieceModel` or `UnigramModel`, auto-detected from `tokenizer.json`'s `model.type`.
"""
function load(dir::AbstractString)
    spec = JSON.parsefile(joinpath(dir, "tokenizer.json"))
    kind = spec["model"]["type"]
    if kind == "WordPiece"
        loadwordpiece(dir)
    elseif kind == "Unigram"
        loadunigram(dir)
    else
        error("unsupported tokenizer type: $kind (only WordPiece and Unigram are implemented)")
    end
end

"""
    encode(model::StaticModel, text::AbstractString) -> Vector{Float32}

One-shot encode: allocates a fresh [`Scratch`](@ref) and a fresh output vector. For repeated
calls (a hot loop over many texts), build a `Scratch` once and call [`encode!`](@ref) instead.
"""
encode(model::StaticModel, text::AbstractString) = copy(encode!(Scratch(model), model, text))

@doc """
    Scratch(model::StaticModel)

Build the reusable scratch buffers `encode!` needs for `model` — a `WordPieceScratch` or
`UnigramScratch`, matching `model`'s type. Construct once per (task, model) pair and reuse it
across calls; this is what makes the hot loop allocation-free. Not safe to
share across concurrent tasks (each needs its own).
""" Scratch

@doc """
    encode!(scratch::Scratch, model::StaticModel, text::AbstractString) -> AbstractVector{Float32}

Encode `text` into `scratch`'s pooling buffer and return it. The returned vector is owned by
`scratch` and will be overwritten by the next call — `copy` it if you need to keep the result
(this is what [`encode`](@ref) does for you). Allocation-free after warmup for both model
types (buffers grow to the longest input seen, then are reused).
""" encode!

end
