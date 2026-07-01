# Model2Vec.jl

[![CI](https://github.com/D3MZ/Model2Vec.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/D3MZ/Model2Vec.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/D3MZ/Model2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/D3MZ/Model2Vec.jl)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://D3MZ.github.io/Model2Vec.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Native-Julia inference for [model2vec](https://github.com/MinishLab/model2vec) static embedding models: **allocation-free** for both tokenizer families model2vec ships with — `WordPiece` (e.g. `potion-base-8M`) and `Unigram`/SentencePiece (e.g. `potion-multilingual-128M`) — and faster than a native (no-FFI) Rust reference running the identical algorithm, for both.

<p align="center"><img src="bench/benchmark.svg" width="820" alt="benchmark"></p>

4,000 real Common Crawl WET records, M1 Max, single thread, Rust = native (no FFI):

| Tokenizer (model) | Julia | Rust | Speedup |
|---|---:|---:|---:|
| WordPiece (`potion-base-8M`) | **24,344 records/s, 0 allocs** | 7,604 records/s | **3.20x** |
| Unigram (`potion-multilingual-128M`) | **5,780 records/s, 0 allocs** | 3,576 records/s | **1.62x** |

<sub>Reproduce: `bench/run.sh` (extracts real page text from a Common Crawl WET file — see [`bench/extract_wet_corpus.jl`](bench/extract_wet_corpus.jl)). Unigram's vocab trie is a darts-style double-array built at load time, matching the same technique the `Precompiled` charsmap decoder already used — see [How it works](#how-it-works).</sub>

In production: [MonsieurPapin](https://github.com/D3MZ/MonsieurPapin.jl) switched from a Rust FFI bridge to this package — **8.95x faster**, 0.9999999... score correlation, over 21,465 real web-crawl records (this package's own isolated hot path is allocation-free; MonsieurPapin's wrapper carries a small residual ~3.25 allocs/record from its own WET-record handling, not from Model2Vec.jl).

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

## Features vs. the alternatives

| | Rust (`tokenizers`+`safetensors`, no FFI) | **Model2Vec.jl** |
|---|:---:|:---:|
| WordPiece · Unigram tokenizers | ✓ · ✓ | ✓ · ✓ |
| Embedding dtype | F32 · F16 · I8 | F32 · F16 · I8 |
| Per-token weights / dedup-mapping tensors | ✓ | ✓ |
| Load from local path · Hugging Face Hub | ✓ · ✓ | ✓ · by design, see [Scope](#scope) |
| Non-ASCII normalization | Exact | Unigram exact · WordPiece approximated |
| Allocation-free hot path | ✗ | ✓ |
| Speed (this repo's benchmark, single thread) | 1.00x | 1.62x–3.20x |
| Language · FFI | Rust · – | Julia · **none** |

Model2Vec.jl trades remote Hugging Face downloads for having no FFI dependency and an allocation-free hot path. Both cover the two tokenizer families model2vec ships with, all three embedding dtypes, and the `weights`/`mapping` tensors vocabulary-quantized checkpoints carry. Details: [Scope](#scope).

## Scope

Case-folding and punctuation handling are byte-level ASCII. WordPiece approximates accent stripping and skips non-ASCII words rather than mis-tokenizing them. Unigram implements SentencePiece's `Precompiled` charsmap byte-for-byte (the darts double-array table decoded straight from tokenizer.json) plus the reference crate's exact Viterbi/unk semantics — ≥ 0.9995 cosine agreement with the Rust reference on every one of 4,000 real multilingual web-crawl records (median 1.0). The two residual Unigram edge cases (utf8proc vs `unicode_segmentation` grapheme-cluster version drift, and U+FFFD counting on invalid UTF-8) are documented at the top of `src/unigram.jl`. Neither backend crashes or produces garbage; every input still tokenizes and pools to a valid embedding.

## How it works

Both backends load the model once (tokenizer vocab into a `Dict`-pair for WordPiece or a darts-style double-array trie for Unigram, embeddings as a zero-copy reshape of the safetensors file), then reuse a `Scratch` buffer so tokenization never allocates a fresh `String` per candidate: WordPiece looks up `Dict{String,Int32}` vocab maps via `SubString`s of a `StringView` over a persistent buffer (a zero-allocation lookup key); Unigram walks its double-array trie — one XOR and one label check per byte, no pointer-chasing — to run Viterbi segmentation with no per-position string allocation. The vocab trie is built with the same double-array technique the `Precompiled` charsmap decoder uses to read SentencePiece's own pre-built one, constructed here from scratch at load time.

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
