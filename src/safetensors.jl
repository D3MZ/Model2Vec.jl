# Minimal safetensors reader, shared by both tokenizer backends: reads the `embeddings` (or
# legacy `0`) tensor as a (dim, rows) column-major Float32 matrix, where column j is the
# contiguous embedding for row (j-1). safetensors stores tensors row-major (rows, dim);
# reinterpreting the same bytes as Julia column-major (dim, rows) is a relabeling of the same
# memory, not a transpose. F32 is a zero-copy reinterpret (materialized into an owned array once,
# same one-time cost `collect` always paid here, so `bytes` -- the whole file -- doesn't stay
# pinned in memory); F64/F16/I8 need an actual numeric conversion to F32, matching the Rust
# reference's `floats()` (model.rs) exactly, including I8's unscaled direct cast.
#
# Also reads the two optional tensors some model2vec checkpoints ship (both produced by the
# official package's `vocabulary_quantization` -- KMeans over the embedding rows):
#   * `weights`:  per-token Float scale, multiplied into each token's row before mean-pooling
#     (F64/F32/F16, matching the Rust reference's `weights()` dtype set exactly);
#   * `mapping`:  per-token row index -- token id t pools row `mapping[t]`, so many vocab
#     tokens can share one deduplicated embedding row and `rows` may be smaller than the
#     vocab size (`length(mapping)` is the vocab size then).
# Absent tensors are materialized as unit/identity vectors at load time -- ~8 bytes/token vs
# ~4·dim bytes/token for the embeddings themselves, a rounding error -- so the pooling hot
# loops stay branch-free and allocation-free with a single unconditional indirection.
function loadembeddings(path::AbstractString)
    bytes = read(path)
    headerlen = Int(only(reinterpret(UInt64, @view bytes[1:8])))
    header = JSON.parse(String(bytes[9:8+headerlen]))
    database = 8 + headerlen # bytes before the data section (1-indexed positions 1:database)
    raw(entry) = @view bytes[database+entry["data_offsets"][1]+1:database+entry["data_offsets"][2]]

    entry = haskey(header, "embeddings") ? header["embeddings"] : header["0"]
    rows, cols = entry["shape"]
    embeddings = reshape(decodefloats(entry["dtype"], raw(entry), "embedding", ("F32", "F16", "I8")), cols, rows)

    # `mapping` is written by safetensors.numpy from sklearn's `kmeans.predict` labels --
    # int32 in practice, int64 if a checkpoint round-trips through numpy's default int. The
    # official minishlab/model2vec-rs crate (`decode_token_mapping`) validates the declared
    # dtype and accepts exactly {I64, I32}; we match it. (The vendored model.rs reference
    # instead reads raw LE i32 chunks *ignoring* the dtype field -- correct for every
    # checkpoint the official Python package produces today, but it would silently mangle an
    # I64 mapping; honoring the dtype is the behavior the official implementations agree on.)
    mapping = if haskey(header, "mapping")
        m = header["mapping"]
        dtype = m["dtype"]
        if dtype == "I32"
            collect(reinterpret(Int32, raw(m)))
        elseif dtype == "I64"
            Int32.(reinterpret(Int64, raw(m)))
        else
            error("unsupported mapping dtype $dtype (supported: I32, I64)")
        end
    else
        collect(Int32(0):Int32(rows - 1)) # identity: token id t pools row t
    end

    weights = haskey(header, "weights") ?
        decodefloats(header["weights"]["dtype"], raw(header["weights"]), "weights", ("F64", "F32", "F16")) :
        ones(Float32, length(mapping))

    embeddings, weights, mapping
end

# Decode a safetensors payload to a flat Vector{Float32}, restricted to the dtypes the Rust
# reference accepts for that tensor (`floats()` for embeddings, `weights()` for weights).
function decodefloats(dtype::AbstractString, raw::AbstractVector{UInt8}, what::AbstractString, allowed)
    dtype in allowed || error("unsupported $what dtype $dtype (supported: $(join(allowed, ", ")))")
    dtype == "F32" ? collect(reinterpret(Float32, raw)) :
    dtype == "F16" ? Float32.(reinterpret(Float16, raw)) :
    dtype == "F64" ? Float32.(reinterpret(Float64, raw)) :
    Float32.(reinterpret(Int8, raw)) # I8: unscaled direct cast, matching model.rs
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
# that cut as a codeunit (byte) count -- an Int, so both tokenizer hot paths can honor it
# without materializing a substring (keeping `encode!` allocation-free even for over-budget
# text). The `ncodeunits` short-circuit skips char-counting entirely in the common
# already-short case (a string's character count never exceeds its byte count); `eachindex`
# never throws on invalid UTF-8 (real crawled text is not guaranteed valid -- each malformed
# byte counts as one character, the same count Rust sees after its from_utf8_lossy boundary
# conversion).
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
