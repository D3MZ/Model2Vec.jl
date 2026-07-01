# Model2Vec.jl

[![CI](https://github.com/D3MZ/Model2Vec.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/D3MZ/Model2Vec.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/D3MZ/Model2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/Model2Vec.jl)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://D3MZ.github.io/Model2Vec.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Native-Julia inference for [model2vec](https://github.com/MinishLab/model2vec) static embedding models: **allocation-free** for `WordPiece` models, and faster than a native (no-FFI) Rust reference — for both tokenizer families model2vec ships with, `WordPiece` (e.g. `potion-base-8M`) and `Unigram`/SentencePiece (e.g. `potion-multilingual-128M`).

<p align="center"><img src="bench/benchmark.svg" width="820" alt="benchmark"></p>

| 4,000-record synthetic corpus, M1 Max, single thread | throughput | vs Rust |
|---|---:|---:|
| **WordPiece** (`potion-base-8M`) — Julia | **457,841 records/s, 0 allocs** | **6.60x** |
| WordPiece — Rust (native, no FFI) | 69,332 records/s | 1.00x |
| **Unigram** (`potion-multilingual-128M`) — Julia | **79,553 records/s** | **1.11x** |
| Unigram — Rust (native, no FFI) | 71,398 records/s | 1.00x |

<sub>Reproduce: `bench/run.sh`.</sub>

In production: [MonsieurPapin](https://github.com/D3MZ/MonsieurPapin.jl) switched from a Rust FFI bridge to this package — **2.4-2.6x faster**, 0.998 score correlation, over 21,465 real web-crawl records.

## Install

```julia
pkg> add https://github.com/D3MZ/Model2Vec.jl
```

## Usage

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

`load` auto-detects the tokenizer family from `tokenizer.json`.

## Scope

Case-folding and punctuation handling are byte-level ASCII; full Unicode normalization is approximated, not byte-for-byte. WordPiece skips non-ASCII words rather than mis-tokenizing them. Unigram approximates SentencePiece's `Precompiled` charsmap with `Unicode.normalize` — exact on ASCII/Latin-script text, up to a few percent cosine-distance drift on non-Latin-script text. Neither gap crashes or produces garbage; every input still tokenizes and pools to a valid embedding.

## How it works

Both backends load the model once (tokenizer vocab into a `Dict`-pair for WordPiece or a byte-trie for Unigram, embeddings as a zero-copy reshape of the safetensors file), then reuse a `Scratch` buffer so tokenization never allocates a fresh `String` per candidate: WordPiece looks up `Dict{String,Int32}` vocab maps via `SubString`s of a `StringView` over a persistent buffer (a zero-allocation lookup key); Unigram walks a trie built at load time to run Viterbi segmentation with no per-position string allocation.

## Citing

```bibtex
@software{Model2Vec_jl,
  author  = {Michael, Demetrius},
  title   = {{Model2Vec.jl}: Native-Julia inference for model2vec static embedding models},
  url     = {https://github.com/D3MZ/Model2Vec.jl},
  version = {0.1.0},
  year    = {2026}
}
```

## License

MIT © Demetrius Michael · `bench/run.sh` reproduces the numbers above.
