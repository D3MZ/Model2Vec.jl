# Minimal safetensors reader, shared by both tokenizer backends: reads the `embeddings` (or
# legacy `0`) tensor as a (dim, vocab) column-major Float32 matrix, where column j is the
# contiguous embedding for token id (j-1). safetensors stores tensors row-major (vocab, dim);
# reinterpreting the same bytes as Julia column-major (dim, vocab) is a relabeling of the same
# memory, not a transpose, so this is a zero-copy reshape (`collect` below is only a one-time
# copy off the raw file bytes into an owned array, not a per-token cost).
function loadembeddings(path::AbstractString)
    bytes = read(path)
    headerlen = Int(only(reinterpret(UInt64, @view bytes[1:8])))
    header = JSON.parse(String(bytes[9:8+headerlen]))
    (haskey(header, "weights") || haskey(header, "mapping")) &&
        error("model has weights/mapping tensors (per-token scale + row remap) — unsupported; " *
              "this package's pooling assumes scale=1.0 and identity mapping")
    entry = haskey(header, "embeddings") ? header["embeddings"] : header["0"]
    entry["dtype"] == "F32" || error("expected F32 embeddings, got $(entry["dtype"])")
    rows, cols = entry["shape"]
    start, stop = entry["data_offsets"] # 0-indexed, relative to the start of the data section
    database = 8 + headerlen # bytes before the data section (1-indexed positions 1:database)
    lo = database + start + 1
    hi = database + stop
    flat = reinterpret(Float32, @view bytes[lo:hi])
    reshape(collect(flat), cols, rows)
end

loadnormalize(dir::AbstractString) = get(JSON.parsefile(joinpath(dir, "config.json")), "normalize", true)

const MAX_LENGTH = 512 # matches model2vec-rs's StaticModel::encode default
