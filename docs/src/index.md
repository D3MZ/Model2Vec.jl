# Model2Vec.jl

Native-Julia inference for [model2vec](https://github.com/MinishLab/model2vec) static embedding
models: **allocation-free** for `WordPiece`-tokenized models, and faster than a native (no-FFI)
Rust reference running the same algorithm, for both tokenizer families model2vec ships with.

A model2vec model is *tokenize → look up a per-token embedding row → mean-pool → optionally
L2-normalize*. The tokenizer is the expensive part, and model2vec models use one of two
tokenizer families depending on the checkpoint:

  * **WordPiece** (BERT-style: e.g. `minishlab/potion-base-8M`) — greedy longest-match per word.
  * **Unigram** (SentencePiece-style: e.g. `minishlab/potion-multilingual-128M`) — Viterbi
    segmentation over a byte-trie of the vocabulary.

This package implements both directly in Julia, over raw UTF-8 bytes, with no FFI.

## Installation

```julia
pkg> add https://github.com/D3MZ/Model2Vec.jl
```

## Quick start

```julia
using Model2Vec

model = Model2Vec.load(modeldir)          # a local model2vec snapshot (tokenizer.json,
                                           # model.safetensors, config.json)
v = Model2Vec.encode(model, "cat dog")    # Vector{Float32}, length == model.dim

# Reuse buffers across many calls (this is what makes the hot loop allocation-free for
# WordPiece models):
scratch = Model2Vec.Scratch(model)
for text in many_texts
    v = Model2Vec.encode!(scratch, model, text)   # owned by scratch; copy if you keep it
end
```

`load` auto-detects the tokenizer family from `tokenizer.json` — the same `encode`/`encode!`
calls work for either.

## How it works

Both backends load the target model directory once (tokenizer vocab into either a `Dict`-pair
for WordPiece or a byte-trie for Unigram, plus the embedding matrix as a zero-copy reshape of
the safetensors file), then reuse a `Scratch` buffer across calls so tokenization never
allocates a fresh `String` per candidate: WordPiece looks up `Dict{String,Int32}` vocab maps
using `SubString`s of a `StringView` wrapped around a persistent scratch buffer (verified
empirically to be a zero-allocation lookup key); Unigram walks a trie built once at load time to
run Viterbi segmentation without per-position string allocation.

See the [README](https://github.com/D3MZ/Model2Vec.jl#readme) for the benchmark plot and full
write-up, [Scope](scope.md) for what's approximated on non-ASCII input, and the
[API reference](@ref api) for the functions.

## [API reference](@id api)

See [API](api.md) for full docstrings of [`load`](@ref), [`encode`](@ref), [`encode!`](@ref),
and [`Scratch`](@ref).
