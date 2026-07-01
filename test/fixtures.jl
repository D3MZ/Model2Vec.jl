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

# Writes a minimal valid safetensors file with one "embeddings" tensor: `vocab` rows x `dim`
# cols, row-major (matching what real model2vec checkpoints ship). `dtype` selects the on-disk
# encoding ("F32", "F16", or "I8" -- matching what loadembeddings supports); values are cast
# accordingly, so pass fixture values already sized for the target dtype (e.g. small integers
# for "I8", since it's an unscaled direct cast, matching the Rust reference).
function writesafetensors(path::AbstractString, embeddings::Matrix{Float32}; dtype="F32")
    vocab, dim = size(embeddings) # embeddings[i, :] is token (i-1)'s row
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
    header = Dict("embeddings" => Dict(
        "dtype" => dtype, "shape" => [vocab, dim], "data_offsets" => [0, length(data)],
    ))
    headerjson = JSON.json(header)
    open(path, "w") do io
        write(io, htol(UInt64(ncodeunits(headerjson))))
        write(io, headerjson)
        write(io, data)
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

function buildwordpiecefixture(dir::AbstractString; normalize=true, dtype="F32")
    mkpath(dir)
    pieces = ["[PAD]", "[UNK]", "cat", "dog", "run", "##ning", "the", "a", "b"]
    dim = 4
    # I8 is an unscaled direct cast (matching the Rust reference), so scale up first -- otherwise
    # the smooth sin()-based fixture values (all in [-1,1]) would mostly round to the same few
    # integers and the distinctness/similarity assertions below would be meaningless.
    embeddings = dtype == "I8" ? round.(fixtureembeddings(length(pieces), dim) .* 20) : fixtureembeddings(length(pieces), dim)
    writewordpiecetokenizer(joinpath(dir, "tokenizer.json"), pieces; max_input_chars_per_word=10)
    writesafetensors(joinpath(dir, "model.safetensors"), embeddings; dtype)
    writeconfig(joinpath(dir, "config.json"); normalize)
    dir
end

function buildunigramfixture(dir::AbstractString; normalize=true, charsmap=false)
    mkpath(dir)
    pieces = [("[PAD]", -10.0), ("[UNK]", -10.0), ("▁", -1.0), ("▁cat", -1.0), ("▁dog", -1.0), ("a", -2.0), ("b", -2.0)]
    dim = 4
    writeunigramtokenizer(joinpath(dir, "tokenizer.json"), pieces; unk_id=1, charsmap)
    writesafetensors(joinpath(dir, "model.safetensors"), fixtureembeddings(length(pieces), dim))
    writeconfig(joinpath(dir, "config.json"); normalize)
    dir
end
