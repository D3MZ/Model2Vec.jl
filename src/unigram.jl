# ---- SentencePiece Unigram tokenizer (used by e.g. minishlab/potion-multilingual-128M) ----
#
# Mirrors `tokenizers::Tokenizer` configured as:
#   Normalizer::Sequence[Sequence[Precompiled(charsmap), Replace(/ {2,}/->" ")],
#                        Replace('!'->' ! '), ... (space out ASCII punctuation) ...,
#                        Replace(/\s+/->" "), Strip(left,right)]
#   -> Metaspace(replacement="▁", prepend_scheme=always, split=false)   # whole text is one "word"
#   -> Unigram: Viterbi segmentation maximizing sum of per-piece log-probs, byte_fallback=false.
#
# Scope limitation (documented, not implemented): the `Precompiled` charsmap step is a
# SentencePiece binary normalization table (NFKC-ish folding, mostly affecting non-ASCII text
# -- fullwidth forms, unusual whitespace/dashes, etc). It is approximated with
# `Unicode.normalize` (NFKC compose + strip default-ignorable/control chars), not implemented
# byte-for-byte. For ASCII/English text this matches the reference tokenizer to ~1e-8; for
# non-Latin-script text (tested on real multilingual web text) embeddings can diverge by up to
# a few percent — see README.md for measured numbers.

const PUNCT = raw"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

function buildpunctlut()
    lut = fill(false, 256)
    for b in codeunits(PUNCT)
        lut[Int(b)+1] = true
    end
    Tuple(lut)
end

# A top-level `const X = f()` only executes once, during precompilation -- invisible to
# per-test-run coverage tracking (which only sees code the coverage-instrumented process itself
# executes; a precompiled module load skips re-running top-level statements). `__init__()` is
# guaranteed to run on every module load, so building the LUT there instead keeps it both
# correctly initialized and actually covered.
const PUNCTLUT = Ref{NTuple{256,Bool}}()

struct Trie
    # node 1 = root. children[node] : Dict{UInt8,Int32} byte -> child node index.
    children::Vector{Dict{UInt8,Int32}}
    terminalid::Vector{Int32}   # -1 if node is not a complete piece
    terminalscore::Vector{Float32}
end

Trie() = Trie([Dict{UInt8,Int32}()], Int32[-1], Float32[0f0])

function trieinsert!(trie::Trie, bytes::AbstractVector{UInt8}, id::Int32, score::Float32)
    node = 1
    for b in bytes
        children = trie.children[node]
        next = get(children, b, Int32(0))
        if next == 0
            push!(trie.children, Dict{UInt8,Int32}())
            push!(trie.terminalid, Int32(-1))
            push!(trie.terminalscore, 0f0)
            next = Int32(length(trie.children))
            children[b] = next
        end
        node = next
    end
    trie.terminalid[node] = id
    trie.terminalscore[node] = score
end

struct UnigramVocab
    trie::Trie
    unk_id::Int32
    unk_score::Float32
end

struct UnigramModel <: StaticModel
    vocab::UnigramVocab
    embeddings::Matrix{Float32} # (dim, vocab)
    dim::Int
    normalize::Bool
end

mutable struct UnigramScratch
    normbuf::Vector{UInt8}   # normalize! output (ASCII-punct-spaced, whitespace-collapsed)
    bytes::Vector{UInt8}     # metaspace-transformed byte buffer (▁ prepended/substituted)
    best::Vector{Float64}    # best[i] = best score of segmenting bytes[1:i-1]
    backpos::Vector{Int32}   # backpos[i] = start position of the last piece ending at i-1
    backid::Vector{Int32}    # backid[i] = token id of the last piece ending at i-1
    ids::Vector{Int32}       # reconstructed token ids (reused buffer)
    sum::Vector{Float32}
end
Scratch(model::UnigramModel) = UnigramScratch(
    Vector{UInt8}(undef, 1024), Vector{UInt8}(undef, 1024), Float64[], Int32[], Int32[], Int32[],
    Vector{Float32}(undef, model.dim))

function loadunigramvocab(path::AbstractString)
    spec = JSON.parsefile(path)
    model = spec["model"]
    vocab = model["vocab"] # Vector of [piece::String, score::Float64], position = id (0-indexed)
    unk_id = Int32(model["unk_id"])

    trie = Trie()
    for (i, entry) in enumerate(vocab)
        piece = entry[1]::AbstractString
        score = Float32(entry[2])
        trieinsert!(trie, codeunits(piece), Int32(i - 1), score)
    end
    unk_score = Float32(vocab[unk_id+1][2])
    UnigramVocab(trie, unk_id, unk_score)
end

function loadunigram(dir::AbstractString)
    vocab = loadunigramvocab(joinpath(dir, "tokenizer.json"))
    embeddings = loadembeddings(joinpath(dir, "model.safetensors"))
    UnigramModel(vocab, embeddings, size(embeddings, 1), loadnormalize(dir))
end

@inline isspacebyteug(b::UInt8) = b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d || b == 0x0c || b == 0x0b
@inline isasciipunctug(b::UInt8) = @inbounds PUNCTLUT[][Int(b)+1]

# Normalize + space out ASCII punctuation + collapse whitespace + strip, mirroring the explicit
# Replace/Strip rules in the tokenizer.json normalizer sequence (the Precompiled charsmap step is
# approximated separately, see `approxcharsmap`). Writes into `dst`, returns the valid length.
function normalizeug!(dst::Vector{UInt8}, text::AbstractString)
    units = codeunits(text)
    n = length(units)
    cap = 3n + 8
    length(dst) < cap && resize!(dst, cap)
    len = 0
    lastspace = true # emulate leading Strip by not emitting a leading space
    @inbounds for i in 1:n
        b = units[i]
        if isspacebyteug(b)
            lastspace || (len += 1; dst[len] = 0x20)
            lastspace = true
        elseif isasciipunctug(b)
            lastspace || (len += 1; dst[len] = 0x20)
            len += 1; dst[len] = b
            len += 1; dst[len] = 0x20
            lastspace = true
        else
            len += 1; dst[len] = b
            lastspace = false
        end
    end
    lastspace && len > 0 && (len -= 1) # strip trailing space
    len
end

# Metaspace: replace each ASCII space with the 3-byte "▁" (U+2581, E2 96 81) and prepend one at
# the very start (prepend_scheme=always). Writes into scratch.bytes, returns length.
const META = (0xe2, 0x96, 0x81)
function metaspace!(scratch::UnigramScratch, normalized::Vector{UInt8}, len::Int)
    len == 0 && return 0 # matches the `tokenizers` crate: no prepend on empty (post-normalize) input
    cap = 3 * (len + 1) + 8
    length(scratch.bytes) < cap && resize!(scratch.bytes, cap)
    dst = scratch.bytes
    out = 0
    for b in META
        out += 1; dst[out] = b
    end
    @inbounds for i in 1:len
        b = normalized[i]
        if b == 0x20
            for m in META
                out += 1; dst[out] = m
            end
        else
            out += 1; dst[out] = b
        end
    end
    out
end

const NEG_INF = -1.0e30

# Viterbi segmentation over `bytes[1:n]` using the trie; writes token ids into `scratch.ids`.
function viterbi!(scratch::UnigramScratch, vocab::UnigramVocab, n::Int)
    best = scratch.best; backpos = scratch.backpos; backid = scratch.backid
    length(best) < n + 1 && (resize!(best, n + 1); resize!(backpos, n + 1); resize!(backid, n + 1))
    bytes = scratch.bytes
    trie = vocab.trie
    children = trie.children; terminalid = trie.terminalid; terminalscore = trie.terminalscore

    @inbounds begin
        best[1] = 0.0
        for i in 2:n+1
            best[i] = NEG_INF
        end

        for i in 1:n
            best[i] <= NEG_INF && continue
            node = 1
            j = i
            matchedany = false
            while j <= n
                nextnode = get(children[node], bytes[j], Int32(0))
                nextnode == 0 && break
                node = nextnode
                j += 1
                tid = terminalid[node]
                if tid >= 0
                    matchedany = true
                    cand = best[i] + terminalscore[node]
                    if cand > best[j]
                        best[j] = cand
                        backpos[j] = i
                        backid[j] = tid
                    end
                end
            end
            if !matchedany
                # single-byte UNK fallback so the DP always stays connected
                cand = best[i] + vocab.unk_score
                if cand > best[i+1]
                    best[i+1] = cand
                    backpos[i+1] = i
                    backid[i+1] = vocab.unk_id
                end
            end
        end
    end

    ids = scratch.ids
    empty!(ids)
    pos = n + 1
    @inbounds while pos > 1
        push!(ids, backid[pos])
        pos = backpos[pos]
    end
    reverse!(ids)
    ids
end

function poolug!(scratch::UnigramScratch, model::UnigramModel)
    sum = scratch.sum
    fill!(sum, 0f0)
    ids = scratch.ids # unk ids already filtered out in encode! (before MAX_LENGTH truncation)
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

# Approximates the `Precompiled` charsmap step (a SentencePiece binary Unicode-folding table)
# not implemented byte-for-byte: NFKC composition + stripping default-ignorable/BOM and control
# characters. Allocates (Unicode.normalize returns a new String) -- the one place this backend
# is not allocation-free; see the module/README scope note.
approxcharsmap(text::AbstractString) = Unicode.normalize(text; compose=true, stripignore=true, stripcc=true)

function encode!(scratch::UnigramScratch, model::UnigramModel, text::AbstractString)
    text = approxcharsmap(text)
    normlen = normalizeug!(scratch.normbuf, text)
    n = metaspace!(scratch, scratch.normbuf, normlen)
    viterbi!(scratch, model.vocab, n)
    # Match model.rs's order exactly: drop unk *then* truncate to 512, not the reverse —
    # truncating first could keep fewer than 512 real tokens when unk ids fall within the prefix.
    unk = model.vocab.unk_id
    filter!(!=(unk), scratch.ids)
    length(scratch.ids) > MAX_LENGTH && resize!(scratch.ids, MAX_LENGTH)
    poolug!(scratch, model)
end
