# Scope

Case-folding and whitespace/punctuation handling are byte-level ASCII.

  * **WordPiece**: no accent stripping; non-ASCII words are skipped (contribute no tokens)
    rather than mis-tokenized, since WordPiece's `##`-continuation search assumes single-byte
    characters.
  * **Unigram**: the SentencePiece `Precompiled` charsmap (a binary Unicode-folding table) is
    implemented byte-for-byte — the darts double-array trie is decoded straight from
    `tokenizer.json` and traversed exactly like the reference `spm_precompiled` crate,
    including its grapheme-cluster replacement rule — as are the reference crate's Viterbi
    lattice, `fuse_unk`, and unk-scoring semantics. Measured against the Rust reference on
    4,000 real multilingual web-crawl records: cosine agreement ≥ 0.9995 on every record,
    median 1.0. Two residual edge cases are documented at the top of `src/unigram.jl`
    (grapheme segmentation library version drift, and U+FFFD counting on invalid UTF-8).

Neither backend crashes or produces garbage — every input, ASCII or not, still tokenizes and
pools to a valid embedding.

## Other known limits

  * `load` only reads local model2vec snapshot directories (`tokenizer.json`,
    `model.safetensors`, `config.json`) — **by design**, not a missing feature: Hugging Face
    Hub downloads pull in HTTP, auth tokens, and etag/resume caching, which don't belong in a
    package whose entire value proposition is zero-dependency, allocation-free inference.
    MonsieurPapin's own `hubsnapshot` helper follows the same resolve-local-and-error pattern.
    Fetch the snapshot yourself first, e.g.:
    ```julia
    using HuggingFaceApi # ] add HuggingFaceApi
    modeldir = dirname(hf_hub_download("minishlab/potion-base-8M", "tokenizer.json"))
    ```
    or from the shell: `huggingface-cli download minishlab/potion-base-8M` (Python) or
    `hf download minishlab/potion-base-8M` (the `huggingface_hub` CLI), or clone the model repo
    directly.
  * Models with `weights` or `mapping` safetensors (the per-token pooling scale and
    deduplicated-row remap produced by the official package's `vocabulary_quantization`) are
    supported: each pooled token contributes `weights[t] * embeddings[mapping[t]]`, matching
    the Rust reference's `pool`. Weights decode from `F64`/`F32`/`F16`; mapping from
    `I32`/`I64` (the dtype set the official `model2vec-rs` crate validates against — sklearn's
    KMeans labels are int32, so `I32` is what checkpoints ship in practice). When absent, both
    are materialized as unit/identity vectors at load time so the pooling hot loops stay
    branch-free and allocation-free either way.
  * `F32`, `F16`, and `I8` embedding tensors are all supported (decoded to `F32` at load time,
    matching the Rust reference's unscaled direct cast for `I8`).
