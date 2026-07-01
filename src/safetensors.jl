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
