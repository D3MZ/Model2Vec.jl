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
    `model.safetensors`, `config.json`) — it does not download from the Hugging Face Hub. Fetch
    the snapshot yourself first (e.g. with `hf-hub` from Rust, `huggingface_hub` from Python, or
    by cloning the model repo).
  * Models with `weights` or `mapping` safetensors (per-token Zipf scale, dedup row remap) are
    rejected with a clear error rather than silently pooled incorrectly — `potion-base-8M` and
    `potion-multilingual-128M` (the models this package is benchmarked against) have neither.
  * `F32`, `F16`, and `I8` embedding tensors are all supported (decoded to `F32` at load time,
    matching the Rust reference's unscaled direct cast for `I8`).
