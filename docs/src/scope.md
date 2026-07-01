# Scope

Case-folding and whitespace/punctuation handling are byte-level ASCII. Full Unicode
normalization is approximated, not implemented byte-for-byte:

  * **WordPiece**: no accent stripping; non-ASCII words are skipped (contribute no tokens)
    rather than mis-tokenized, since WordPiece's `##`-continuation search assumes single-byte
    characters.
  * **Unigram**: the SentencePiece `Precompiled` charsmap (a binary Unicode-folding table) is
    approximated with `Unicode.normalize` (NFKC + control/ignorable stripping). This matches the
    reference tokenizer closely on ASCII/Latin-script text; on non-Latin-script text (tested
    against real multilingual web text) embeddings can diverge from the reference by a few
    percent of cosine distance in the worst case.

Both gaps are about matching a reference byte-for-byte on non-ASCII input, not about crashing or
producing garbage — every input, ASCII or not, still tokenizes and pools to a valid embedding.

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
