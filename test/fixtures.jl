# Tiny, hand-built model2vec snapshots so tests are fully self-contained -- no network access,
# no dependence on a locally-cached HuggingFace download. This is what makes coverage in CI
# (a fresh runner with nothing cached) match coverage measured locally.
using JSON, Base64

# A minimal hand-assembled SentencePiece `Precompiled` charsmap blob (the darts double-array
# format src/unigram.jl's Charsmap decodes), so the trie traversal is exercised even when no
# real model is cached. Two mappings: 'A' (0x41) -> "a" and '…' (0xE2 0x80 0xA6) -> "...".
# Slot layout (node index = XOR chain of offsets and input bytes, see charsmaptransform):
#   root slot 0 has offset 0, so byte b lands on slot b directly;
#   'A':  slot 65  = leaf, offset 64  -> value slot 65⊻64  = 1   (pool offset 0, "a")
#   0xE2: slot 226 = offset 4         -> base 226⊻4 = 230
#   0x80: slot 230⊻0x80 = 102 = offset 8 -> base 102⊻8 = 110
#   0xA6: slot 110⊻0xA6 = 200 = leaf, offset 12 -> value slot 200⊻12 = 196 (pool offset 2, "...")
# Unused slots are 0 (label 0 mismatches every byte); the 240-unit array is deliberately
# smaller than 256 so first bytes >= 0xF0 (e.g. emoji) exercise the out-of-range -> miss path.
function buildcharsmapblob()
    trie = zeros(UInt32, 240)
    trie[65+1] = UInt32((64 << 10) | (1 << 8) | 0x41)
    trie[1+1] = UInt32(0x80000000)          # value node: pool offset 0 ("a")
    trie[226+1] = UInt32((4 << 10) | 0xE2)
    trie[102+1] = UInt32((8 << 10) | 0x80)
    trie[200+1] = UInt32((12 << 10) | (1 << 8) | 0xA6)
    trie[196+1] = UInt32(0x80000000) | UInt32(2) # value node: pool offset 2 ("...")
    pool = UInt8[0x61, 0x00, 0x2e, 0x2e, 0x2e, 0x00] # "a\0...\0"
    io = IOBuffer()
    write(io, htol(UInt32(4 * length(trie))))
    foreach(u -> write(io, htol(u)), trie)
    write(io, pool)
    take!(io)
end

# Writes a minimal valid safetensors file with an "embeddings" tensor (`rows` x `dim`,
# row-major, matching what real model2vec checkpoints ship) plus, optionally, the two extra
# tensors vocabulary-quantized checkpoints carry: `weights` (per-token pooling scale) and
# `mapping` (per-token 0-based embedding row -- with dedup, `rows` < vocab size). `dtype`/
# `weightsdtype`/`mappingdtype` select each tensor's on-disk encoding; values are cast
# accordingly, so pass fixture values already sized for the target dtype (e.g. small integers
# for "I8", since it's an unscaled direct cast, matching the Rust reference). An unsupported
# declared dtype (for load()-rejection tests) falls back to the default byte encoding -- the
# loader must error on the header before ever touching the payload.
function writesafetensors(path::AbstractString, embeddings::Matrix{Float32}; dtype="F32",
                          weights=nothing, weightsdtype="F32", mapping=nothing, mappingdtype="I32")
    rows, dim = size(embeddings) # embeddings[i, :] is row (i-1)
    rowmajor = vec(permutedims(embeddings))
    data = if dtype == "F32"
        reinterpret(UInt8, rowmajor)
    elseif dtype == "F16"
        reinterpret(UInt8, Float16.(rowmajor))
    elseif dtype == "I8"
        reinterpret(UInt8, Int8.(rowmajor))
    else
        error("fixtures.jl: unsupported dtype $dtype")
    end
    tensors = Tuple{String,String,Vector{Int},AbstractVector{UInt8}}[("embeddings", dtype, [rows, dim], data)]
    if weights !== nothing
        wdata = weightsdtype == "F64" ? reinterpret(UInt8, Float64.(weights)) :
                weightsdtype == "F16" ? reinterpret(UInt8, Float16.(weights)) :
                reinterpret(UInt8, Float32.(weights))
        push!(tensors, ("weights", weightsdtype, [length(weights)], wdata))
    end
    if mapping !== nothing # 0-based row indices, one per vocab token
        mdata = mappingdtype == "I64" ? reinterpret(UInt8, Int64.(mapping)) :
                reinterpret(UInt8, Int32.(mapping))
        push!(tensors, ("mapping", mappingdtype, [length(mapping)], mdata))
    end
    header = Dict{String,Any}()
    offset = 0
    for (name, dt, shape, tdata) in tensors
        header[name] = Dict("dtype" => dt, "shape" => shape, "data_offsets" => [offset, offset + length(tdata)])
        offset += length(tdata)
    end
    headerjson = JSON.json(header)
    open(path, "w") do io
        write(io, htol(UInt64(ncodeunits(headerjson))))
        write(io, headerjson)
        foreach(t -> write(io, t[4]), tensors)
    end
end

function writeconfig(path::AbstractString; normalize=true)
    open(path, "w") do io
        JSON.print(io, Dict("normalize" => normalize))
    end
end

# `pieces` is a Vector of piece strings; a piece starting with "##" is a WordPiece continuation.
# Token id = position in `pieces` - 1 (0-indexed), matching `embeddings`' row order.
function writewordpiecetokenizer(path::AbstractString, pieces::Vector{String}; max_input_chars_per_word=100)
    vocab = Dict(p => i - 1 for (i, p) in enumerate(pieces))
    spec = Dict("model" => Dict(
        "type" => "WordPiece", "vocab" => vocab, "unk_token" => "[UNK]",
        "max_input_chars_per_word" => max_input_chars_per_word,
    ))
    open(path, "w") do io
        JSON.print(io, spec)
    end
end

# `pieces` is a Vector of (piece::String, score::Float64); position = token id (0-indexed).
# `charsmap=true` embeds the synthetic Precompiled blob above, nested two `Sequence`s deep --
# the same shape as the real potion-multilingual-128M tokenizer.json, so the recursive
# normalizer-tree search is exercised too.
function writeunigramtokenizer(path::AbstractString, pieces::Vector{Tuple{String,Float64}}; unk_id=0, charsmap=false)
    spec = Dict{String,Any}("model" => Dict(
        "type" => "Unigram", "unk_id" => unk_id,
        "vocab" => [[p, s] for (p, s) in pieces],
    ))
    if charsmap
        precompiled = Dict("type" => "Precompiled", "precompiled_charsmap" => base64encode(buildcharsmapblob()))
        spec["normalizer"] = Dict("type" => "Sequence", "normalizers" => [
            Dict("type" => "Sequence", "normalizers" => [precompiled, Dict("type" => "Replace")]),
        ])
    end
    open(path, "w") do io
        JSON.print(io, spec)
    end
end

# Deterministic (not random) embeddings so downstream assertions (norm, equality, similarity)
# are reproducible: row i is a one-hot-ish pattern scaled so pooling/normalizing stays well-behaved.
function fixtureembeddings(vocab::Integer, dim::Integer)
    e = zeros(Float32, vocab, dim)
    for i in 1:vocab
        for j in 1:dim
            e[i, j] = Float32(sin(i * 0.7 + j * 1.3)) # smooth, distinct-per-row, no exact zeros/collisions
        end
    end
    e
end

# With a `mapping`, the embeddings matrix only carries the deduplicated rows (maximum row
# index + 1), matching what `vocabulary_quantization` produces -- fewer rows than vocab tokens.
function buildwordpiecefixture(dir::AbstractString; normalize=true, dtype="F32",
                               weights=nothing, weightsdtype="F32", mapping=nothing, mappingdtype="I32")
    mkpath(dir)
    pieces = ["[PAD]", "[UNK]", "cat", "dog", "run", "##ning", "the", "a", "b"]
    dim = 4
    rows = mapping === nothing ? length(pieces) : maximum(mapping) + 1
    # I8 is an unscaled direct cast (matching the Rust reference), so scale up first -- otherwise
    # the smooth sin()-based fixture values (all in [-1,1]) would mostly round to the same few
    # integers and the distinctness/similarity assertions below would be meaningless.
    embeddings = dtype == "I8" ? round.(fixtureembeddings(rows, dim) .* 20) : fixtureembeddings(rows, dim)
    writewordpiecetokenizer(joinpath(dir, "tokenizer.json"), pieces; max_input_chars_per_word=10)
    writesafetensors(joinpath(dir, "model.safetensors"), embeddings; dtype, weights, weightsdtype, mapping, mappingdtype)
    writeconfig(joinpath(dir, "config.json"); normalize)
    dir
end

function buildunigramfixture(dir::AbstractString; normalize=true, charsmap=false,
                             weights=nothing, weightsdtype="F32", mapping=nothing, mappingdtype="I32")
    mkpath(dir)
    pieces = [("[PAD]", -10.0), ("[UNK]", -10.0), ("▁", -1.0), ("▁cat", -1.0), ("▁dog", -1.0), ("a", -2.0), ("b", -2.0)]
    dim = 4
    rows = mapping === nothing ? length(pieces) : maximum(mapping) + 1
    writeunigramtokenizer(joinpath(dir, "tokenizer.json"), pieces; unk_id=1, charsmap)
    writesafetensors(joinpath(dir, "model.safetensors"), fixtureembeddings(rows, dim); weights, weightsdtype, mapping, mappingdtype)
    writeconfig(joinpath(dir, "config.json"); normalize)
    dir
end
