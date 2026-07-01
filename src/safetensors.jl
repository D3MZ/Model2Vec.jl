# Minimal safetensors reader, shared by both tokenizer backends: reads the `embeddings` (or
# legacy `0`) tensor as a (dim, vocab) column-major Float32 matrix, where column j is the
# contiguous embedding for token id (j-1). safetensors stores tensors row-major (vocab, dim);
# reinterpreting the same bytes as Julia column-major (dim, vocab) is a relabeling of the same
# memory, not a transpose. F32 is a zero-copy reinterpret (materialized into an owned array once,
# same one-time cost `collect` always paid here, so `bytes` -- the whole file -- doesn't stay
# pinned in memory); F16/I8 need an actual numeric conversion to F32, matching the Rust
# reference's `floats()` (model.rs) exactly, including I8's unscaled direct cast.
function loadembeddings(path::AbstractString)
    bytes = read(path)
    headerlen = Int(only(reinterpret(UInt64, @view bytes[1:8])))
    header = JSON.parse(String(bytes[9:8+headerlen]))
    (haskey(header, "weights") || haskey(header, "mapping")) &&
        error("model has weights/mapping tensors (per-token scale + row remap) — unsupported; " *
              "this package's pooling assumes scale=1.0 and identity mapping")
    entry = haskey(header, "embeddings") ? header["embeddings"] : header["0"]
    dtype = entry["dtype"]
    rows, cols = entry["shape"]
    start, stop = entry["data_offsets"] # 0-indexed, relative to the start of the data section
    database = 8 + headerlen # bytes before the data section (1-indexed positions 1:database)
    raw = @view bytes[database+start+1:database+stop]

    flat = if dtype == "F32"
        collect(reinterpret(Float32, raw))
    elseif dtype == "F16"
        Float32.(reinterpret(Float16, raw))
    elseif dtype == "I8"
        Float32.(reinterpret(Int8, raw))
    else
        error("unsupported embedding dtype $dtype (supported: F32, F16, I8)")
    end
    reshape(flat, cols, rows)
end

loadnormalize(dir::AbstractString) = get(JSON.parsefile(joinpath(dir, "config.json")), "normalize", true)

const MAX_LENGTH = 512 # matches model2vec-rs's StaticModel::encode default

# Median byte-length of the raw tokenizer.json vocab keys ("##"/"▁" prefixes included), matching
# model.rs's `from_pretrained` exactly: sort byte lengths, take `lengths[len / 2]` (0-indexed;
# 1 for an empty vocab). Used by `truncateinput` to budget raw input length before tokenizing.
vocabmedian(lengths::Vector{Int}) = (sort!(lengths); isempty(lengths) ? 1 : lengths[div(end, 2) + 1])

# Pre-truncating raw input to MAX_LENGTH * median *characters* before any tokenization work
# mirrors model.rs's `truncate()`: `char_indices().nth(chars)` counts Unicode scalar values
# (not bytes) and slices right before character number `chars` (0-indexed), keeping exactly
# `chars` characters; shorter text is kept whole (Rust's `None` branch). `truncatebound` returns
# that cut as a codeunit (byte) count -- an Int, so the WordPiece hot path can honor it without
# materializing a substring (keeping `encode!` allocation-free even for over-budget text). The
# `ncodeunits` short-circuit skips char-counting entirely in the common already-short case (a
# string's character count never exceeds its byte count); `eachindex` never throws on invalid
# UTF-8 (real crawled text is not guaranteed valid -- each malformed byte counts as one
# character, the same count Rust sees after its from_utf8_lossy boundary conversion).
function truncatebound(text::AbstractString, median::Int)
    budget = MAX_LENGTH * median
    n = ncodeunits(text)
    n <= budget && return n
    seen = 0
    for i in eachindex(text)
        seen == budget && return i - 1 # `i` starts character budget+1: keep the bytes before it
        seen += 1
    end
    n
end

# String-level form of the same cut, for callers that need an AbstractString (the Unigram path,
# whose charsmap approximation consumes whole strings). Identity -- no allocation -- when the
# text is already within budget.
function truncateinput(text::AbstractString, median::Int)
    bound = truncatebound(text, median)
    bound == ncodeunits(text) ? text : SubString(text, 1, thisind(text, bound))
end
