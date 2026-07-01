# ---- WordPiece tokenizer (BERT-style, ASCII fast path) ----
#
# Mirrors `tokenizers::Tokenizer` configured as:
#   BertNormalizer(lowercase=true) -> BertPreTokenizer (whitespace + ASCII punctuation split)
#   -> WordPiece greedy-longest-match with "##" continuation prefix, unk_token="[UNK]".
# Non-ASCII text (accents, CJK) is not normalized identically to the `tokenizers` crate — a
# known scope limitation; see README.md for the measured impact.

struct WordPieceVocab
    initial::Dict{String,Int32}      # word-initial pieces, keyed by raw text (no "##")
    continuation::Dict{String,Int32} # continuation pieces, keyed by text with "##" stripped
    unk_id::Int32
    max_input_chars_per_word::Int    # words longer than this become a single [UNK] (dropped)
end

struct WordPieceModel <: StaticModel
    vocab::WordPieceVocab
    embeddings::Matrix{Float32} # (dim, vocab_size), column j = embedding of token id (j-1)
    dim::Int
    normalize::Bool
    median::Int # median byte-length of raw vocab keys ("##" included); input budget for truncateinput
end

# `view` wraps `word` *by reference* (StringView holds the Vector object, not a copy), so
# resize!(word, ...) growing the buffer is reflected in `view` automatically with no rebuild —
# constructed once, valid for the lifetime of the Scratch. This is what makes `encode!`
# allocation-free after warmup.
mutable struct WordPieceScratch
    word::Vector{UInt8}                # lowercased current word bytes
    view::StringView{Vector{UInt8}}    # zero-copy string view over `word`
    ids::Vector{Int32}                 # token ids collected for the whole text
    sum::Vector{Float32}               # pooling accumulator, length == dim
end

function Scratch(model::WordPieceModel)
    word = Vector{UInt8}(undef, 256)
    WordPieceScratch(word, StringView(word), Vector{Int32}(undef, 0), Vector{Float32}(undef, model.dim))
end

function loadwordpiecevocab(path::AbstractString)
    spec = JSON.parsefile(path)
    model = spec["model"]
    vocab = model["vocab"] # Dict{String,Int} token -> id
    unk_token = get(model, "unk_token", "[UNK]")
    max_input_chars_per_word = get(model, "max_input_chars_per_word", 100)

    initial = Dict{String,Int32}()
    continuation = Dict{String,Int32}()
    for (tok, id) in vocab
        if startswith(tok, "##")
            continuation[tok[3:end]] = Int32(id)
        else
            initial[tok] = Int32(id)
        end
    end
    unk_id = Int32(get(vocab, unk_token, -1))
    WordPieceVocab(initial, continuation, unk_id, max_input_chars_per_word), vocabmedian([sizeof(tok) for tok in keys(vocab)])
end

function loadwordpiece(dir::AbstractString)
    vocab, median = loadwordpiecevocab(joinpath(dir, "tokenizer.json"))
    embeddings = loadembeddings(joinpath(dir, "model.safetensors"))
    WordPieceModel(vocab, embeddings, size(embeddings, 1), loadnormalize(dir), median)
end

@inline isasciipunctwp(b::UInt8) =
    (0x21 <= b <= 0x2f) || (0x3a <= b <= 0x40) || (0x5b <= b <= 0x60) || (0x7b <= b <= 0x7e)
@inline isspacebytewp(b::UInt8) = b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d || b == 0x0c || b == 0x0b
@inline lowerbyte(b::UInt8) = (0x41 <= b <= 0x5a) ? (b + 0x20) : b

# Greedy longest-match WordPiece over `view[1:len]` (already lowercased ASCII bytes).
# Appends resulting ids to `ids`; on failure (no path segments this word), appends nothing
# (mirrors model2vec-rs which drops the produced [UNK] id before pooling).
@inline function isasciiword(word::Vector{UInt8}, len::Int)
    @inbounds for i in 1:len
        word[i] >= 0x80 && return false
    end
    true
end

# Byte-range candidate slicing (`endpos` walking down one byte at a time) is only valid for
# single-byte-per-character text: a `SubString` cut mid-UTF-8-character throws. This backend's
# scope is ASCII (see module docstring), so non-ASCII words are dropped here rather than
# byte-sliced -- correctness-preserving (matches the "unmapped word -> no tokens" path already
# used for OOV words) rather than crashing on the first multi-byte word it sees.
function wordpiece!(ids::Vector{Int32}, vocab::WordPieceVocab, view::StringView{Vector{UInt8}}, len::Int)
    len == 0 && return
    len > vocab.max_input_chars_per_word && return
    isasciiword(view.data, len) || return
    start = 1
    base = length(ids)
    # SubString of a StringView is a zero-allocation Dict{String,V} lookup key: hash/isequal
    # dispatch generically over codeunits, matching a String key with the same bytes without
    # ever materializing one.
    while start <= len
        found = Int32(-1)
        endpos = len
        dict = start == 1 ? vocab.initial : vocab.continuation
        while endpos >= start
            key = SubString(view, start, endpos)
            id = get(dict, key, Int32(-1))
            if id >= 0
                found = id
                break
            end
            endpos -= 1
        end
        found < 0 && (resize!(ids, base); return)
        push!(ids, found)
        start = endpos + 1
    end
end

# Tokenize the first `n` codeunits of `text` (a char-boundary cut from `truncatebound` -- passed
# as a byte count rather than a substring to stay allocation-free), writing resulting ids into
# `scratch.ids` (cleared first). ASCII fast path: splits on whitespace, treats each ASCII
# punctuation byte as its own single-char word.
function tokenizewp!(scratch::WordPieceScratch, vocab::WordPieceVocab, text::AbstractString, n::Int)
    ids = scratch.ids
    empty!(ids)
    word = scratch.word
    view = scratch.view
    units = codeunits(text)
    wlen = 0
    i = 1
    @inbounds while i <= n
        b = units[i]
        if isspacebytewp(b)
            wlen > 0 && (wordpiece!(ids, vocab, view, wlen); wlen = 0)
        elseif isasciipunctwp(b)
            wlen > 0 && (wordpiece!(ids, vocab, view, wlen); wlen = 0)
            word[1] = lowerbyte(b)
            wordpiece!(ids, vocab, view, 1)
        else
            wlen += 1
            wlen > length(word) && resize!(word, 2 * length(word))
            word[wlen] = lowerbyte(b)
        end
        i += 1
    end
    wlen > 0 && wordpiece!(ids, vocab, view, wlen)
    ids
end

# Mean-pool + (optional) L2-normalize the embedding rows selected by `ids` into `scratch.sum`.
function poolwp!(scratch::WordPieceScratch, model::WordPieceModel)
    sum = scratch.sum
    fill!(sum, 0f0)
    ids = scratch.ids
    E = model.embeddings
    count = 0
    @inbounds for id in ids
        col = Int(id) + 1
        @simd for k in 1:model.dim
            sum[k] += E[k, col]
        end
        count += 1
    end
    denom = Float32(max(count, 1))
    @inbounds @simd for k in 1:model.dim
        sum[k] /= denom
    end
    if model.normalize
        norm = 0f0
        @inbounds @simd for k in 1:model.dim
            norm += sum[k] * sum[k]
        end
        norm = max(sqrt(norm), 1f-12)
        @inbounds @simd for k in 1:model.dim
            sum[k] /= norm
        end
    end
    sum
end

# Encode `text` into `scratch.sum` (returned, owned by scratch — copy if you need to keep it).
# Allocation-free after warmup: no candidate-lookup String is ever materialized (see `view`
# above), and `scratch`'s buffers only grow, never reallocate on a steady-state text-length mix.
function encode!(scratch::WordPieceScratch, model::WordPieceModel, text::AbstractString)
    # parity with model.rs: budget raw input (as a byte bound, no substring) before tokenizing
    tokenizewp!(scratch, model.vocab, text, truncatebound(text, model.median))
    length(scratch.ids) > MAX_LENGTH && resize!(scratch.ids, MAX_LENGTH)
    poolwp!(scratch, model)
end
