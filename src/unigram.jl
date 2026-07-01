# ---- SentencePiece Unigram tokenizer (used by e.g. minishlab/potion-multilingual-128M) ----
#
# Mirrors `tokenizers::Tokenizer` configured as:
#   Normalizer::Sequence[Sequence[Precompiled(charsmap), Replace(/ {2,}/->" ")],
#                        Replace('!'->' ! '), ... (space out ASCII punctuation) ...,
#                        Replace(/\s+/->" "), Strip(left,right)]
#   -> Metaspace(replacement="▁", prepend_scheme=always, split=false)   # whole text is one "word"
#   -> Unigram: Viterbi segmentation maximizing sum of per-piece log-probs, byte_fallback=false.
#
# The `Precompiled` charsmap step -- SentencePiece's binary normalization table (NFKC-ish
# folding: fullwidth forms, ligatures, unusual whitespace/dashes, halfwidth katakana, ...) --
# is implemented byte-for-byte (see `Charsmap` below), matching the `spm_precompiled` Rust
# crate the reference `tokenizers` stack uses. Two known residual divergences from the Rust
# reference, both confined to unusual input:
#   * grapheme segmentation comes from the utf8proc library linked into Julia itself, while
#     Rust uses the `unicode_segmentation` crate -- Unicode-version drift between the two
#     could group edge-case clusters (new emoji ZWJ sequences) differently;
#   * invalid UTF-8 is folded to U+FFFD per malformed `Char` as Julia iterates them, while
#     Rust's `String::from_utf8_lossy` boundary emits U+FFFD per maximal invalid subsequence
#     (WHATWG counting) -- replacement-character *counts* can differ on garbage bytes. The
#     synthesized U+FFFD also bypasses the charsmap trie lookup Rust would still perform on it;
#     no real-model effect (NFKC-derived charsmaps don't map U+FFFD), only reachable on already-
#     invalid input.

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

# ---- `Precompiled` charsmap: SentencePiece's binary normalization table ----
#
# The blob (base64 in tokenizer.json's normalizer tree) is a "darts" double-array trie plus a
# replacement-string pool: [u32 LE trie_size_bytes][trie_size/4 x u32 LE darts units][pool:
# raw bytes, NUL-terminated entries]. Trie keys are the UTF-8 byte sequences to fold; a leaf's
# value is the byte offset of its replacement in the pool. An empty `Charsmap` (a tokenizer
# with no Precompiled normalizer) is the identity: every lookup misses, all bytes pass through.
struct Charsmap
    trie::Vector{UInt32} # darts double-array units
    pool::Vector{UInt8}  # replacement strings, NUL-terminated
    asciipass::Vector{Bool} # 128 entries; see the 2-arg constructor below charsmaptransform
end
Charsmap() = Charsmap(UInt32[], UInt8[])

# darts unit fields, exactly as the reference `spm_precompiled` crate decodes them: bit 8 marks
# "a leaf value node is reachable from here"; the low byte (plus bit 31, which tags value nodes
# so they can never match a real input byte) is the label checked against the input; the offset
# is XORed into the node index to advance (bit 9 selects a <<8 scale for large offsets).
@inline dartsleaf(u::UInt32) = (u >> 8) & 0x1 == 0x1
@inline dartsvalue(u::UInt32) = u & 0x7fffffff
@inline dartslabel(u::UInt32) = u & ((UInt32(1) << 31) | 0xff)
@inline dartsoffset(u::UInt32) = Int(u >> 10) << ((u & (UInt32(1) << 9)) >> 6)

# Depth-first search of the normalizer JSON tree for a {"type": "Precompiled"} node -- the
# nesting varies by model (potion-multilingual-128M buries it two `Sequence`s deep), so walk
# every dict/array rather than hard-coding a path.
findprecompiled(::Any) = nothing
findprecompiled(node::AbstractVector) = searchprecompiled(node)
function findprecompiled(node::AbstractDict)
    get(node, "type", nothing) == "Precompiled" ? node : searchprecompiled(values(node))
end
function searchprecompiled(nodes)
    for v in nodes
        r = findprecompiled(v)
        r !== nothing && return r
    end
    nothing
end

function loadcharsmap(spec::AbstractDict)
    node = findprecompiled(get(spec, "normalizer", nothing))
    node === nothing && return Charsmap()
    blob = base64decode(node["precompiled_charsmap"]::AbstractString)
    triesize = Int(ltoh(only(reinterpret(UInt32, @view blob[1:4]))))
    trie = ltoh.(collect(reinterpret(UInt32, @view blob[5:4+triesize])))
    Charsmap(trie, blob[5+triesize:end])
end

# Common-prefix search of `bytes[i:stop]` in the darts trie, returning the pool offset
# (0-based) of the *first* (shortest) matching prefix's replacement, or -1 for no match --
# matching `spm_precompiled::transform`'s `results[0]` semantics exactly, including replacing
# a whole chunk with its shortest matching prefix's replacement and stopping at NUL bytes.
# Out-of-range node indices are treated as a miss (the reference would panic; real blobs are
# sized so it never happens, but a miss keeps malformed/synthetic tables safe).
@inline function charsmaptransform(cm::Charsmap, bytes, i::Int, stop::Int)
    trie = cm.trie
    isempty(trie) && return -1 # identity charsmap
    @inbounds begin
        nodepos = dartsoffset(trie[1]) # root: node 0 XOR its offset
        for k in i:stop
            c = bytes[k]
            c == 0x00 && break
            nodepos ⊻= Int(c)
            nodepos >= length(trie) && return -1
            u = trie[nodepos+1]
            dartslabel(u) != UInt32(c) && return -1
            nodepos ⊻= dartsoffset(u)
            dartsleaf(u) && return nodepos < length(trie) ? Int(dartsvalue(trie[nodepos+1])) : -1
        end
    end
    -1
end

# `asciipass[b+1]`: byte `b` is a printable-ASCII char the charsmap can never rewrite, so runs
# of such bytes can be copied verbatim (see the fast path in charsmapnormalize!). True iff the
# single-byte transform misses -- sufficient because extended grapheme clusters group pure-ASCII
# chars one per cluster (no ASCII char has Extend/Prepend/ZWJ properties), so the reference
# algorithm only ever offers *single* ASCII chars to the trie; a multi-byte key starting with an
# ASCII byte (letter + combining mark) can only fire on a cluster whose *head* is that letter,
# which the fast path leaves to the slow path by holding back a run's last char whenever the
# next char is non-ASCII. The one pure-ASCII multi-char cluster, "\r\n", is excluded by forcing
# CR off the table (real NFKC-derived charsmaps map every C0 control anyway).
function Charsmap(trie::Vector{UInt32}, pool::Vector{UInt8})
    cm = Charsmap(trie, pool, fill(true, 128))
    for b in 0x01:0x7f
        cm.asciipass[b+1] = b != 0x0d && charsmaptransform(cm, (b,), 1, 1) < 0
    end
    cm
end

# Byte length of the NUL-terminated pool entry starting at 0-based offset `off`.
@inline function poollen(cm::Charsmap, off::Int)
    pool = cm.pool
    j = off
    @inbounds while j < length(pool) && pool[j+1] != 0x00
        j += 1
    end
    j - off
end

# Byte-trie over the vocab. Built with per-node Dicts (TrieBuilder, load time only), then
# frozen into a darts-style double-array layout for the Viterbi hot loop -- the same technique
# the `Precompiled` charsmap above ships pre-built, constructed here from scratch at load time.
# Slots are 0-based; slot `s` owns two adjacent UInt64 words: `slots[2s+1]` packs the slot's
# in-edge byte label (9 bits; 0x100 marks a free slot, unmatched by any real byte) and the
# node's child `base` (bits 9+), and `slots[2s+2]` packs its terminal (score_bits << 32) |
# id_bits, id -1 = not a complete piece. A node's child along byte `b` lives at slot
# `base XOR b` and is valid iff that slot's stored label equals `b` -- one interleaved 16-byte
# pair per node visit (meta + terminal on the same cache line), vs. head/edges/terminal spread
# over three arrays in the previous CSR layout. Leaves share a `base` aimed at a 256-aligned
# all-free "dead block" at the end of the array, so the lookup is branchless and, because the
# slot count is a multiple of 256 and every base < slot count, always in bounds. The root
# (slot 0, never returned as a child) additionally gets a dense 256-entry table: every Viterbi
# position starts with a root lookup, making it the hottest of all.
struct Trie
    rootlut::Vector{Int32} # dense byte -> child slot for the root (0 = no child)
    slots::Vector{UInt64}  # 2 words per slot: [label | base << 9], [terminal]
end

const TRIEFREE = UInt64(0x100) # label field of a free slot: no input byte can match it

@inline trielabel(m::UInt64) = m & 0x1ff
@inline triebase(m::UInt64) = Int(m >> 9)
@inline trieterminal(trie::Trie, node::Int32) = @inbounds trie.slots[2 * Int(node) + 2]
@inline terminalid(t::UInt64) = reinterpret(Int32, t % UInt32)
@inline terminalscore(t::UInt64) = reinterpret(Float32, (t >> 32) % UInt32)
@inline packterminal(id::Int32, score::Float32) =
    (UInt64(reinterpret(UInt32, score)) << 32) | UInt64(reinterpret(UInt32, id))

struct TrieBuilder
    children::Vector{Dict{UInt8,Int32}} # node 1 = root; byte -> child node index
    terminalid::Vector{Int32}
    terminalscore::Vector{Float32}
end
TrieBuilder() = TrieBuilder([Dict{UInt8,Int32}()], Int32[-1], Float32[0f0])

function trieinsert!(trie::TrieBuilder, bytes::AbstractVector{UInt8}, id::Int32, score::Float32)
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

# Mutable double-array under construction: label/base metas and terminals kept as separate
# grow-by-doubling arrays (interleaved only at the end), a doubly-linked list threading the
# free slots (O(1) removal, first-fit iteration from `freehead`), and a used-`base` bitset --
# two distinct nodes must never share a base, or a query on one could label-match a child of
# the other. Capacity stays a multiple of 256 so `base XOR byte` never escapes it.
mutable struct DartsBuild
    meta::Vector{UInt64}
    term::Vector{UInt64}
    nextfree::Vector{Int32} # 0-based successor free slot, -1 = list end
    prevfree::Vector{Int32} # 0-based predecessor free slot, -1 = list head
    usedbase::BitVector
    freehead::Int32
    maxused::Int
end

function DartsBuild(cap::Int)
    cap = max(256, 256 * cld(cap, 256))
    b = DartsBuild(fill(TRIEFREE, cap), fill(packterminal(Int32(-1), 0f0), cap),
                   Vector{Int32}(undef, cap), Vector{Int32}(undef, cap),
                   falses(cap), Int32(0), 0)
    for s in 0:cap-1
        b.nextfree[s+1] = Int32(s + 1 < cap ? s + 1 : -1)
        b.prevfree[s+1] = Int32(s - 1)
    end
    b
end

function dartsgrow!(b::DartsBuild)
    old = length(b.meta)
    cap = 2 * old
    resize!(b.meta, cap); resize!(b.term, cap)
    resize!(b.nextfree, cap); resize!(b.prevfree, cap)
    resize!(b.usedbase, cap)
    for s in old:cap-1
        b.meta[s+1] = TRIEFREE
        b.term[s+1] = packterminal(Int32(-1), 0f0)
        b.nextfree[s+1] = Int32(s + 1 < cap ? s + 1 : -1)
        b.prevfree[s+1] = Int32(s - 1)
        b.usedbase[s+1] = false
    end
    # the old range was fully allocated (growth only happens once first-fit runs off the end),
    # so the fresh tail is the entire free list
    b.freehead = Int32(old)
    b.prevfree[old+1] = Int32(-1)
    nothing
end

@inline dartsfree(b::DartsBuild, s::Int) = @inbounds trielabel(b.meta[s+1]) == TRIEFREE

function dartsoccupy!(b::DartsBuild, s::Int, label::UInt64)
    nxt = b.nextfree[s+1]; prv = b.prevfree[s+1]
    nxt >= 0 && (b.prevfree[nxt+1] = prv)
    prv >= 0 ? (b.nextfree[prv+1] = nxt) : (b.freehead = nxt)
    b.meta[s+1] = label # base filled in when this node's own children are placed
    s > b.maxused && (b.maxused = s)
    nothing
end

# First-fit base search: walk the free list, aim the first child byte at each free slot, and
# accept the first base whose full child set lands on free slots and that no other node uses.
function dartsfindbase!(b::DartsBuild, bytes::Vector{UInt8})
    c1 = Int(bytes[1])
    e = Int(b.freehead)
    while true
        if e < 0
            e = length(b.meta)
            dartsgrow!(b)
        end
        base = e ⊻ c1
        if !b.usedbase[base+1]
            ok = true
            for k in 2:length(bytes)
                t = base ⊻ Int(bytes[k]) # < cap: capacity is a multiple of 256 and e < cap
                if !dartsfree(b, t)
                    ok = false
                    break
                end
            end
            if ok
                b.usedbase[base+1] = true
                return base
            end
        end
        e = Int(b.nextfree[e+1])
    end
end

# Freeze the builder into the double-array, processing nodes in builder-index order (a child's
# builder index always exceeds its parent's, so every node's slot is assigned before its own
# children are placed) -- which is vocab insertion order, keeping prefix-sharing pieces in
# nearby slots, the same locality argument the previous CSR layout benchmarked as a win.
function freeze(builder::TrieBuilder)
    children = builder.children
    nnodes = length(children)
    b = DartsBuild(nnodes + nnodes ÷ 2)
    # reserve slot 0 for the root, so 0 stays the "no child" sentinel; its label (0x101) is
    # neither the free marker (construction must see slot 0 as occupied) nor matchable by any
    # real byte (a query can compute candidate slot 0 whenever base == byte)
    dartsoccupy!(b, 0, UInt64(0x101))
    slotof = Vector{Int32}(undef, nnodes)
    slotof[1] = 0
    bytesbuf = UInt8[]
    for i in 1:nnodes
        s = Int(slotof[i])
        b.term[s+1] = packterminal(builder.terminalid[i], builder.terminalscore[i])
        node = children[i]
        isempty(node) && continue # leaves get the shared dead-block base below
        resize!(bytesbuf, length(node))
        copyto!(bytesbuf, sort!(collect(keys(node))))
        base = dartsfindbase!(b, bytesbuf)
        b.meta[s+1] = (b.meta[s+1] & 0x1ff) | (UInt64(base) << 9)
        for c in bytesbuf
            t = base ⊻ Int(c)
            dartsoccupy!(b, t, UInt64(c))
            slotof[node[c]] = Int32(t)
        end
    end
    # dead block: 256 aligned, guaranteed-free slots past everything used; leaves point their
    # base here so lookups on them miss branchlessly. No real node can have been assigned this
    # base: a 256-aligned base B places children at B+byte >= B, so B <= maxused < deadbase.
    deadbase = 256 * cld(b.maxused + 1, 256)
    nslots = deadbase + 256
    for i in 1:nnodes
        isempty(children[i]) && (b.meta[slotof[i]+1] = (b.meta[slotof[i]+1] & 0x1ff) | (UInt64(deadbase) << 9))
    end
    slots = Vector{UInt64}(undef, 2 * nslots)
    for s in 0:nslots-1
        inrange = s < length(b.meta)
        slots[2s+1] = inrange ? b.meta[s+1] : TRIEFREE
        slots[2s+2] = inrange ? b.term[s+1] : packterminal(Int32(-1), 0f0)
    end
    rootlut = zeros(Int32, 256)
    for (c, node) in children[1]
        rootlut[Int(c)+1] = slotof[node]
    end
    Trie(rootlut, slots)
end

# Child of `node` along edge byte `b`, or 0: one XOR to the candidate slot, one label check.
# The root -- the one genuinely wide node -- never gets here, it has the dense rootlut.
@inline function triechild(trie::Trie, node::Int32, b::UInt8)
    slots = trie.slots
    @inbounds begin
        t = triebase(slots[2 * Int(node) + 1]) ⊻ Int(b)
        ifelse(trielabel(slots[2t+1]) == b, Int32(t), Int32(0))
    end
end

struct UnigramVocab
    trie::Trie
    unk_id::Int32
    unk_score::Float32 # min vocab score - 10, matching tokenizers' kUnkPenalty lattice score
end

struct UnigramModel <: StaticModel
    vocab::UnigramVocab
    charsmap::Charsmap
    embeddings::Matrix{Float32} # (dim, rows), column mapping[t]+1 = embedding of token id t
    weights::Vector{Float32}    # per-token pooling scale (all ones when the tensor is absent)
    mapping::Vector{Int32}      # per-token 0-based embedding row (identity when absent)
    dim::Int
    normalize::Bool
    median::Int # median byte-length of raw vocab keys ("▁" included); input budget for truncatebound
end

mutable struct UnigramScratch
    charsmapbuf::Vector{UInt8}          # charsmapnormalize! output (Precompiled folding applied)
    graphemestate::Base.RefValue{Int32} # utf8proc grapheme-break state (a Ref allocates; reuse it)
    normbuf::Vector{UInt8}   # normalizeug! output (ASCII-punct-spaced, whitespace-collapsed)
    bytes::Vector{UInt8}     # metaspace-transformed byte buffer (▁ prepended/substituted)
    best::Vector{Float64}    # best[i] = best score of segmenting bytes[1:i-1]
    backpos::Vector{Int32}   # backpos[i] = start position of the last piece ending at i-1
    backid::Vector{Int32}    # backid[i] = token id of the last piece ending at i-1
    ids::Vector{Int32}       # reconstructed token ids (reused buffer)
    sum::Vector{Float32}
end
Scratch(model::UnigramModel) = UnigramScratch(
    Vector{UInt8}(undef, 1024), Ref(Int32(0)), Vector{UInt8}(undef, 1024),
    Vector{UInt8}(undef, 1024), Float64[], Int32[], Int32[], Int32[],
    Vector{Float32}(undef, model.dim))

function loadunigramvocab(spec::AbstractDict)
    model = spec["model"]
    vocab = model["vocab"] # Vector of [piece::String, score::Float64], position = id (0-indexed)
    unk_id = Int32(model["unk_id"])

    builder = TrieBuilder()
    minscore = Inf
    for (i, entry) in enumerate(vocab)
        piece = entry[1]::AbstractString
        score = Float64(entry[2])
        minscore = min(minscore, score)
        trieinsert!(builder, codeunits(piece), Int32(i - 1), Float32(score))
    end
    # tokenizers scores UNK lattice nodes at min_score - kUnkPenalty (10.0), *not* the unk
    # token's own vocab score -- see models/unigram/model.rs in the reference crate.
    unk_score = Float32(minscore - 10.0)
    UnigramVocab(freeze(builder), unk_id, unk_score), vocabmedian([sizeof(entry[1]::AbstractString) for entry in vocab])
end

function loadunigram(dir::AbstractString)
    spec = JSON.parsefile(joinpath(dir, "tokenizer.json"))
    vocab, median = loadunigramvocab(spec)
    charsmap = loadcharsmap(spec)
    embeddings, weights, mapping = loadembeddings(joinpath(dir, "model.safetensors"))
    UnigramModel(vocab, charsmap, embeddings, weights, mapping, size(embeddings, 1), loadnormalize(dir), median)
end

@inline isspacebyteug(b::UInt8) = b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d || b == 0x0c || b == 0x0b
@inline isasciipunctug(b::UInt8) = @inbounds PUNCTLUT[][Int(b)+1]

# Grow-only append helpers for the charsmap output buffer: ensure capacity, copy, advance.
@inline function charsmapgrow!(scratch::UnigramScratch, need::Int)
    buf = scratch.charsmapbuf
    length(buf) < need && resize!(buf, max(need, 2 * length(buf)))
    buf
end
@inline function emitpool!(scratch::UnigramScratch, cm::Charsmap, off::Int, out::Int)
    n = poollen(cm, off)
    buf = charsmapgrow!(scratch, out + n)
    pool = cm.pool
    @inbounds for k in 1:n
        buf[out+k] = pool[off+k]
    end
    out + n
end
@inline function emitbytes!(scratch::UnigramScratch, units, s::Int, e::Int, out::Int)
    n = e - s + 1
    buf = charsmapgrow!(scratch, out + n)
    @inbounds for k in 1:n
        buf[out+k] = units[s+k-1]
    end
    out + n
end
const FFFD = (0xef, 0xbf, 0xbd) # U+FFFD replacement character, as Rust's from_utf8_lossy emits
@inline function emitfffd!(scratch::UnigramScratch, out::Int)
    buf = charsmapgrow!(scratch, out + 3)
    @inbounds for k in 1:3
        buf[out+k] = FFFD[k]
    end
    out + 3
end

# Extended-grapheme-cluster boundary between adjacent chars, threading utf8proc's break state
# (the same library the Unicode stdlib wraps; linked into every Julia process, so this is a
# plain ccall with no new dependency). ASCII fast path: no ASCII pair joins except CR+LF, and
# an ASCII char can't be mid-way through any joining sequence, so resetting the state is exact.
# Malformed chars mirror Base.Unicode.isgraphemebreak!: always a break, state reset.
@inline function graphemebreakug(state::Base.RefValue{Int32}, c1::Char, c2::Char)
    if isascii(c1) && isascii(c2)
        state[] = Int32(0)
        return !(c1 == '\r' && c2 == '\n')
    end
    if Base.ismalformed(c1) || Base.ismalformed(c2)
        state[] = Int32(0)
        return true
    end
    ccall(:utf8proc_grapheme_break_stateful, Bool, (UInt32, UInt32, Ref{Int32}), c1, c2, state)
end

# One grapheme cluster (`units[s:e]`, char-aligned) through the charsmap, appended at `out`.
# Matches `spm_precompiled::normalize_string`: a cluster shorter than 6 bytes that hits the
# trie is replaced whole (even when the hit is only a prefix of it -- see charsmaptransform);
# otherwise each char is independently replaced-or-passed-through. Malformed chars (invalid
# UTF-8) become U+FFFD, mirroring the Rust reference's from_utf8_lossy input boundary.
function foldcluster!(scratch::UnigramScratch, cm::Charsmap, text::AbstractString, units, s::Int, e::Int, out::Int)
    if e - s < 5 # cluster < 6 bytes: try replacing it whole
        off = charsmaptransform(cm, units, s, e)
        off >= 0 && return emitpool!(scratch, cm, off, out)
    end
    i = s
    while i <= e
        c, nexti = iterate(text, i)::Tuple{Char,Int}
        if Base.ismalformed(c)
            out = emitfffd!(scratch, out)
        else
            off = charsmaptransform(cm, units, i, nexti - 1)
            out = off >= 0 ? emitpool!(scratch, cm, off, out) : emitbytes!(scratch, units, i, nexti - 1, out)
        end
        i = nexti
    end
    out
end

# The full `Precompiled` charsmap normalization pass over `text[1:bound]` (a codeunit bound
# from truncatebound, so no substring is ever materialized): group chars into extended grapheme
# clusters, fold each through the darts trie, write the result into scratch.charsmapbuf and
# return its length. Replaces the former Unicode.normalize-based approximation -- byte-for-byte
# the reference algorithm, and allocation-free after buffer warmup.
function charsmapnormalize!(scratch::UnigramScratch, cm::Charsmap, text::AbstractString, bound::Int)
    units = codeunits(text)
    charsmapgrow!(scratch, bound + 8) # common case: output ≈ input size; emits grow further as needed
    out = 0
    state = scratch.graphemestate
    state[] = Int32(0)
    asciipass = cm.asciipass
    clusterstart = 1
    prevc = '\0'
    i = 1
    @inbounds while i <= bound
        # Fast path, valid only at a cluster head (no pending cluster): bulk-copy a run of
        # charsmap-inert ASCII bytes (see asciipass) -- each is its own grapheme cluster and
        # can never be rewritten, so Char decoding, grapheme-break state calls, and trie probes
        # are all skipped. When the byte after the run is non-ASCII the run's *last* char is
        # held back for the slow path: it may head a longer cluster (e.g. letter + combining
        # mark) whose whole-cluster trie lookup could genuinely hit.
        if i == clusterstart && units[i] < 0x80 && asciipass[units[i]+1]
            j = i + 1
            while j <= bound && units[j] < 0x80 && asciipass[units[j]+1]
                j += 1
            end
            stop = (j > bound || units[j] < 0x80) ? j - 1 : j - 2
            if stop >= i
                out = emitbytes!(scratch, units, i, stop, out)
                clusterstart = i = stop + 1
                state[] = Int32(0) # exact: the last two chars seen are ASCII, which resets it
                continue
            end
        end
        c, nexti = iterate(text, i)::Tuple{Char,Int}
        if i > clusterstart && graphemebreakug(state, prevc, c) # i > clusterstart: c has a predecessor
            out = foldcluster!(scratch, cm, text, units, clusterstart, i - 1, out)
            clusterstart = i
            # re-enter the loop top without consuming c: the new cluster head may start an
            # ASCII fast-path run (the one decoded-Char redundancy per run entry is cheap)
            units[i] < 0x80 && asciipass[units[i]+1] && continue
        end
        prevc = c
        i = nexti
    end
    clusterstart <= bound && (out = foldcluster!(scratch, cm, text, units, clusterstart, bound, out))
    out
end

# Space out ASCII punctuation + collapse whitespace + strip, mirroring the explicit
# Replace/Strip rules that follow the Precompiled step in the tokenizer.json normalizer
# sequence. Consumes charsmapnormalize!'s output bytes; writes into `dst`, returns the length.
function normalizeug!(dst::Vector{UInt8}, src::Vector{UInt8}, n::Int)
    cap = 3n + 8
    length(dst) < cap && resize!(dst, cap)
    len = 0
    lastspace = true # emulate leading Strip by not emitting a leading space
    @inbounds for i in 1:n
        b = src[i]
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

# UTF-8 sequence length from a lead byte (continuation/invalid lead bytes count 1, for
# robustness -- post-charsmap input is always valid UTF-8, malformed bytes became U+FFFD).
@inline utf8lenug(b::UInt8) = b < 0xc0 ? 1 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : 4

# Viterbi segmentation over `bytes[1:n]` using the trie; writes token ids into `scratch.ids`.
# Mirrors `tokenizers::models::unigram`'s `encode_optimized` (a port of sentencepiece's
# unigram_model.cc) exactly: the DP advances one *character* (not byte) at a time; piece
# candidates start at char boundaries; and whenever no piece covers exactly the current char
# (`has_single_node` in the reference), an UNK candidate spanning that whole char keeps the
# lattice connected. Consecutive UNK ids fuse into one during backtrace (`fuse_unk`, the
# crate's Unigram default -- the field is never serialized into tokenizer.json).
function viterbi!(scratch::UnigramScratch, vocab::UnigramVocab, n::Int)
    best = scratch.best; backpos = scratch.backpos; backid = scratch.backid
    length(best) < n + 1 && (resize!(best, n + 1); resize!(backpos, n + 1); resize!(backid, n + 1))
    bytes = scratch.bytes
    trie = vocab.trie
    rootlut = trie.rootlut
    unk = vocab.unk_id

    @inbounds begin
        best[1] = 0.0
        for i in 2:n+1
            best[i] = NEG_INF
        end

        i = 1
        while i <= n
            mblen = min(Int(utf8lenug(bytes[i])), n - i + 1)
            node = rootlut[Int(bytes[i])+1] # the piece under consideration is bytes[i:j-1]
            j = i + 1
            singlenode = false # whether some piece covers exactly this one char
            while node != 0
                t = trieterminal(trie, node)
                tid = terminalid(t)
                if tid >= 0
                    cand = best[i] + terminalscore(t)
                    if cand > best[j]
                        best[j] = cand
                        backpos[j] = i
                        backid[j] = tid
                    end
                    j - i == mblen && (singlenode = true)
                end
                j > n && break
                node = triechild(trie, node, bytes[j])
                j += 1
            end
            if !singlenode
                # UNK candidate spanning the whole char, so the DP always stays connected
                k = i + mblen
                cand = best[i] + vocab.unk_score
                if cand > best[k]
                    best[k] = cand
                    backpos[k] = i
                    backid[k] = unk
                end
            end
            i += mblen
        end
    end

    # Backtrace in two walks (count, then fill right-to-left) instead of push!-and-reverse!:
    # `empty!` releases a large Vector's buffer entirely (Julia >= 1.11 shrinks on big
    # deletions), which would make every long record re-grow `ids` -- `resize!` never
    # releases capacity, keeping the hot path allocation-free. A run of consecutive UNKs
    # contributes a single fused UNK id (walking right-to-left keeps runs intact).
    ids = scratch.ids
    count = 0
    pos = n + 1
    prevunk = false
    @inbounds while pos > 1
        isunk = backid[pos] == unk
        isunk && prevunk || (count += 1)
        prevunk = isunk
        pos = backpos[pos]
    end
    resize!(ids, count)
    pos = n + 1
    prevunk = false
    @inbounds while pos > 1
        id = backid[pos]
        isunk = id == unk
        if !(isunk && prevunk)
            ids[count] = id
            count -= 1
        end
        prevunk = isunk
        pos = backpos[pos]
    end
    ids
end

# Weights/mapping indirection identical to poolwp! -- see the comment there.
function poolug!(scratch::UnigramScratch, model::UnigramModel)
    sum = scratch.sum
    fill!(sum, 0f0)
    ids = scratch.ids # fused unk ids included, matching model.rs (see encode! below)
    E = model.embeddings
    weights = model.weights
    mapping = model.mapping
    count = 0
    @inbounds for id in ids
        t = Int(id) + 1
        col = Int(mapping[t]) + 1
        w = weights[t]
        @simd for k in 1:model.dim
            sum[k] += w * E[k, col]
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

function encode!(scratch::UnigramScratch, model::UnigramModel, text::AbstractString)
    bound = truncatebound(text, model.median) # parity with model.rs: budget raw input before tokenizing
    cmlen = charsmapnormalize!(scratch, model.charsmap, text, bound)
    normlen = normalizeug!(scratch.normbuf, scratch.charsmapbuf, cmlen)
    n = metaspace!(scratch, scratch.normbuf, normlen)
    viterbi!(scratch, model.vocab, n)
    # model.rs drops unk ids only when tokenizer.json has an `unk_token` key -- SentencePiece
    # Unigram specs never do (they carry `unk_id` instead), so the Rust reference keeps the
    # (fused) unk ids and pools their embedding rows; match that, applying only the 512 cap.
    length(scratch.ids) > MAX_LENGTH && resize!(scratch.ids, MAX_LENGTH)
    poolug!(scratch, model)
end
