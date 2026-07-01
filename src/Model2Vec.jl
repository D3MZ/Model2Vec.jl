"""
    Model2Vec

Native-Julia inference for [model2vec](https://github.com/MinishLab/model2vec) static
embedding models: **allocation-free** for `WordPiece`-tokenized models, and faster than an
in-process Rust FFI bridge running the same algorithm for both `WordPiece` and `Unigram`
(SentencePiece) tokenizers.

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
Case-folding and whitespace/punctuation handling are byte-level ASCII; full Unicode
normalization (accent stripping for WordPiece, the SentencePiece `Precompiled` charsmap for
Unigram) is approximated, not implemented byte-for-byte — see `README.md` for the measured
correctness impact on non-ASCII text.
"""
module Model2Vec

using JSON, StringViews, Unicode

export load, encode, encode!, Scratch

abstract type StaticModel end

include("safetensors.jl")
include("wordpiece.jl")
include("unigram.jl")

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

end
