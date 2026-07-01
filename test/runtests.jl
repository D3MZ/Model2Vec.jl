using Model2Vec
using Test

include("fixtures.jl")

# Local HuggingFace hub cache paths for the two reference models (one per tokenizer family).
# Tests that need a model skip gracefully if it isn't cached locally -- no network access is
# taken during tests.
function hubsnapshot(repo::AbstractString)
    base = joinpath(homedir(), ".cache", "huggingface", "hub", "models--" * replace(repo, "/" => "--"), "snapshots")
    isdir(base) || return nothing
    snaps = readdir(base)
    isempty(snaps) ? nothing : joinpath(base, first(snaps))
end

const WORDPIECE_DIR = hubsnapshot("minishlab/potion-base-8M")
const UNIGRAM_DIR = hubsnapshot("minishlab/potion-multilingual-128M")

# Run the Precompiled charsmap pass alone and materialize its output (test helper only; the
# hot path never builds this String).
function cmfold(model, scratch, s)
    n = Model2Vec.charsmapnormalize!(scratch, model.charsmap, s, ncodeunits(s))
    String(scratch.charsmapbuf[1:n])
end

function basicchecks(model, label)
    @testset "$label" begin
        scratch = Scratch(model)

        v1 = copy(encode!(scratch, model, "cat dog"))
        @test length(v1) == model.dim
        @test isapprox(sqrt(sum(v1 .^ 2)), model.normalize ? 1.0f0 : sqrt(sum(v1 .^ 2)); atol=1f-4)

        # deterministic
        v2 = copy(encode!(scratch, model, "cat dog"))
        @test v1 == v2

        # different text -> different embedding
        v3 = copy(encode!(scratch, model, "unrelated text about weather"))
        @test v1 != v3

        # empty input -> zero vector (no tokens to pool)
        vempty = copy(encode!(scratch, model, ""))
        @test all(iszero, vempty)

        # a query is more similar to itself than to unrelated text (cosine similarity)
        cossim(a, b) = sum(a .* b) / (sqrt(sum(a .^ 2)) * sqrt(sum(b .^ 2)) + eps())
        vself = copy(encode!(scratch, model, "the cat sat on the mat"))
        vsame = copy(encode!(scratch, model, "the cat sat on the mat"))
        vother = copy(encode!(scratch, model, "quantum physics and string theory"))
        @test cossim(vself, vsame) > cossim(vself, vother)

        # one-shot `encode` matches the scratch-based `encode!`
        @test encode(model, "cat dog") == v1
    end
end

@testset "Model2Vec.jl" begin
    if WORDPIECE_DIR === nothing
        @info "skipping WordPiece tests: minishlab/potion-base-8M not cached locally"
    else
        model = load(WORDPIECE_DIR)
        @test model isa Model2Vec.WordPieceModel
        basicchecks(model, "WordPiece (potion-base-8M)")

        @testset "WordPiece is allocation-free after warmup" begin
            scratch = Scratch(model)
            encode!(scratch, model, "warmup") # compile + first grow
            allocated = @allocated encode!(scratch, model, "cat dog")
            @test allocated == 0

            # over-budget text (pre-truncation path) must stay allocation-free too: the cut is a
            # byte bound handed to tokenizewp!, never a materialized substring
            long = repeat("the quick brown fox jumps over the lazy dog ", 300) # ≫ 512*median chars
            encode!(scratch, model, long) # warm up buffer growth for this length
            @test (@allocated encode!(scratch, model, long)) == 0
        end

        @testset "WordPiece handles multi-byte UTF-8 without erroring (out of scope, must not crash)" begin
            scratch = Scratch(model)
            for text in ("交易策略 金融市场", "торговая стратегия", "no côntrol chars héllo", "mixed ascii and 中文 words")
                v = encode!(scratch, model, text)
                @test length(v) == model.dim
                @test !any(isnan, v)
            end
        end
    end

    if UNIGRAM_DIR === nothing
        @info "skipping Unigram tests: minishlab/potion-multilingual-128M not cached locally"
    else
        model = load(UNIGRAM_DIR)
        @test model isa Model2Vec.UnigramModel
        basicchecks(model, "Unigram (potion-multilingual-128M)")

        @testset "Unigram handles multi-byte UTF-8 without erroring" begin
            scratch = Scratch(model)
            for text in ("交易策略 金融市场", "торговая стратегия", "no côntrol chars héllo")
                v = encode!(scratch, model, text)
                @test length(v) == model.dim
                @test !any(isnan, v)
            end
        end

        @testset "Precompiled charsmap: real potion-multilingual-128M blob foldings" begin
            @test length(model.charsmap.trie) == 44288 # 237,539-byte blob: 44,288 darts units + 60,383 pool bytes
            @test length(model.charsmap.pool) == 60383
            scratch = Scratch(model)
            for (s, expect) in (("Ａ", "A"), ("１", "1"), ("…", "..."), ("ﬁ", "fi"),
                                ("　", " "), ("ｶ", "カ"), (" ", " ")) # ideographic space, NBSP
                @test cmfold(model, scratch, s) == expect
            end
            @test cmfold(model, scratch, "plain ASCII text") == "plain ASCII text" # no match: unchanged
            @test cmfold(model, scratch, "Ａ１…ﬁ mixed") == "A1...fi mixed"
        end

        @testset "Unigram is allocation-free after warmup" begin
            scratch = Scratch(model)
            texts = ("cat dog", "交易策略 金融市场 Ａ１… test", "Ｈｅｌｌｏ ｶﾀｶﾅ ﬁnance… done")
            foreach(t -> encode!(scratch, model, t), texts) # compile + first buffer growth
            for t in texts
                @test (@allocated encode!(scratch, model, t)) == 0
            end
        end
    end

    if WORDPIECE_DIR === nothing && UNIGRAM_DIR === nothing
        @warn "no model2vec models cached locally -- only load() error-path tests ran"
    end

    @testset "load() rejects unsupported tokenizer types" begin
        mktempdir() do dir
            write(joinpath(dir, "tokenizer.json"), """{"model": {"type": "BPE"}}""")
            @test_throws ErrorException load(dir)
        end
    end

    # Synthetic, hand-built model2vec snapshots (test/fixtures.jl) -- no network access, no
    # dependence on a locally-cached HuggingFace download, so these run identically in CI and
    # locally. Covers everything the real-model tests above cover (gracefully skipped when a
    # model isn't cached) plus specific branches real vocabs make hard to hit deterministically:
    # WordPiece's ASCII-punctuation and max-length-skip paths, and Unigram's UNK-fallback path.
    @testset "synthetic fixtures (self-contained, no network)" begin
        mktempdir() do dir
            wpdir = buildwordpiecefixture(joinpath(dir, "wp"))
            model = load(wpdir)
            @test model isa Model2Vec.WordPieceModel
            basicchecks(model, "WordPiece (synthetic fixture)")

            @testset "WordPiece is allocation-free after warmup" begin
                scratch = Scratch(model)
                # Warm up with the exact text we'll measure: "warmup" itself matches no piece in
                # this small fixture vocab (unlike the real 8M model, whose much larger vocab
                # happens to match something), so it never grows scratch.ids -- warming up with
                # "cat dog" instead ensures the buffer-growth cost is paid before measuring.
                encode!(scratch, model, "cat dog")
                @test (@allocated encode!(scratch, model, "cat dog")) == 0
            end

            @testset "WordPiece branch coverage: continuation split, OOV, max-length skip, punctuation, non-ASCII" begin
                scratch = Scratch(model)
                # "running" (7 <= max=10) greedily splits into "run" + "##ning"; the 12-char word
                # exceeds max_input_chars_per_word=10 (early-return skip, no tokens contributed);
                # "the," and "!" exercise the ASCII-punctuation branch; "日本語" exercises the
                # non-ASCII (isasciiword=false) skip.
                v = encode!(scratch, model, "cat dog running xyzxyzxyzxyz xyz the, end! a 日本語")
                @test length(v) == model.dim
                @test !any(isnan, v)
                @test !all(iszero, v) # "cat"/"dog"/"run"/"##ning"/"the"/"a" all match real pieces
            end

            @testset "input pre-truncation parity with model.rs truncate()" begin
                # Raw vocab key byte lengths sorted: [1,1,3,3,3,3,5,5,6] ("##ning" counted with
                # its prefix, like Rust's get_vocab keys); Rust takes lengths[9/2] (0-indexed) = 3.
                @test model.median == 3
                limit = Model2Vec.MAX_LENGTH * model.median # 1536-char budget

                short = "cat dog"
                @test Model2Vec.truncatebound(short, model.median) == ncodeunits(short) # under budget: whole

                atlimit = "a"^limit
                @test Model2Vec.truncatebound(atlimit, model.median) == ncodeunits(atlimit) # exactly at budget: Rust's None branch

                over = atlimit * "ZZZ"
                bound = Model2Vec.truncatebound(over, model.median)
                @test SubString(over, 1, bound) == atlimit # cut right before character budget+1, matching nth(chars)'s 0-indexing
                @test bound == limit

                # budget counts characters, not bytes: "é" is 2 bytes but 1 char, so the full
                # `limit` chars (2*limit bytes) survive and only the char past the budget is cut
                mb = "é"^limit * "Z"
                @test SubString(mb, 1, thisind(mb, Model2Vec.truncatebound(mb, model.median))) == "é"^limit
                # over budget in *bytes* but exactly at budget in *chars*: kept whole (the loop
                # runs -- byte short-circuit can't apply -- but never reaches character budget+1)
                mbat = "é"^limit
                @test Model2Vec.truncatebound(mbat, model.median) == ncodeunits(mbat)

                # end-to-end: content past the char budget cannot influence the embedding...
                head = "cat "^(limit ÷ 4) # exactly `limit` chars
                @test encode(model, head * "dog dog dog") == encode(model, head)
                # ...but the same word placed *inside* the budget does
                @test encode(model, "cat "^(limit ÷ 4 - 1) * "dog ") != encode(model, head)

                # long invalid UTF-8: char-counting must not crash (each malformed byte = 1 char)
                invalid = String(repeat(UInt8[0x63, 0x61, 0x74, 0xff, 0x20], 500)) # "cat\xff " x500 > limit bytes
                @test !isvalid(invalid)
                v = encode!(Scratch(model), model, invalid)
                @test length(v) == model.dim
                @test !any(isnan, v)
            end
        end

        @testset "embedding dtype support: F32, F16, I8" begin
            for dtype in ("F32", "F16", "I8")
                mktempdir() do dir
                    wpdir = buildwordpiecefixture(dir; dtype)
                    model = load(wpdir)
                    v = encode(model, "cat dog")
                    @test length(v) == model.dim
                    @test !any(isnan, v)
                    @test !all(iszero, v)
                end
            end
            @testset "load() rejects unsupported embedding dtypes" begin
                mktempdir() do dir
                    write(joinpath(dir, "tokenizer.json"), """{"model": {"type": "WordPiece", "vocab": {"a": 0}, "unk_token": "[UNK]"}}""")
                    header = JSON.json(Dict("embeddings" => Dict("dtype" => "I16", "shape" => [1, 2], "data_offsets" => [0, 4])))
                    open(joinpath(dir, "model.safetensors"), "w") do io
                        write(io, htol(UInt64(ncodeunits(header))))
                        write(io, header)
                        write(io, zeros(UInt8, 4))
                    end
                    writeconfig(joinpath(dir, "config.json"))
                    @test_throws ErrorException load(dir)
                end
            end
        end

        mktempdir() do dir
            ugdir = buildunigramfixture(joinpath(dir, "ug"))
            model = load(ugdir)
            @test model isa Model2Vec.UnigramModel
            basicchecks(model, "Unigram (synthetic fixture)")

            @testset "Unigram branch coverage: piece match, standalone-meta match, UNK fallback, punctuation" begin
                scratch = Scratch(model)
                # "▁cat"/"▁dog" match real pieces; "z", ",", "!" each follow a lone "▁" match
                # with no continuation -> per-char UNK fallback; "," and "!" also exercise the
                # ASCII-punctuation branch in normalizeug!.
                v = encode!(scratch, model, "cat dog z, cat!")
                @test length(v) == model.dim
                @test !any(isnan, v)
                @test !all(iszero, v)
            end

            @testset "Unigram tolerates invalid UTF-8 (real WET content is not guaranteed valid)" begin
                # `String` is just a byte buffer in Julia -- this constructs one that is *not*
                # valid UTF-8 (a lone continuation byte, 0xFF, spliced into otherwise-valid text),
                # the same class of input that crashed Unicode.normalize before approxcharsmap
                # started sanitizing first (matching Rust's String::from_utf8_lossy tolerance).
                invalid = String(UInt8[0x63, 0x61, 0x74, 0xff, 0x64, 0x6f, 0x67]) # "cat" 0xFF "dog"
                @test !isvalid(invalid)
                scratch = Scratch(model)
                v = encode!(scratch, model, invalid)
                @test length(v) == model.dim
                @test !any(isnan, v)
            end

            @testset "identity charsmap: no Precompiled normalizer in tokenizer.json" begin
                @test isempty(model.charsmap.trie) # fixture has no normalizer section at all
                # a normalizer tree *without* any Precompiled node anywhere is identity too
                cm = Model2Vec.loadcharsmap(Dict{String,Any}("normalizer" => Dict{String,Any}(
                    "type" => "Sequence", "normalizers" => [Dict{String,Any}("type" => "Strip")])))
                @test isempty(cm.trie) && isempty(cm.pool)
                scratch = Scratch(model)
                @test cmfold(model, scratch, "cat Ａ… dog") == "cat Ａ… dog" # nothing folds without a table
                # malformed bytes still become U+FFFD (the from_utf8_lossy input boundary), never a crash
                @test cmfold(model, scratch, String(UInt8[0x63, 0xff, 0x64])) == "c�d"
            end

            @testset "Unigram Viterbi UNK fallback (direct, isolated from real-vocab luck)" begin
                # Low-level: bypasses load()/encode! entirely so this doesn't depend on whether a
                # real or fixture vocab happens to have full byte coverage -- 'z' is deliberately
                # absent from the fixture vocab (root has no child for it), so viterbi! must take
                # the `!singlenode` UNK branch for it. metaspace! unconditionally prepends "▁" to any
                # non-empty input first, which the fixture vocab *does* have as a standalone
                # piece (id 2), so the expected segmentation is ["▁", UNK] not just [UNK].
                vocab = model.vocab
                scratch2 = Scratch(model)
                n = Model2Vec.metaspace!(scratch2, Vector{UInt8}(codeunits("z")), 1)
                ids = Model2Vec.viterbi!(scratch2, vocab, n)
                @test ids == Int32[2, vocab.unk_id]
            end

            @testset "Unigram input pre-truncation (char budget from raw vocab median)" begin
                # Raw piece byte lengths sorted: [1,1,3,5,5,6,6] ("▁cat" counted with its 3-byte
                # metaspace prefix); Rust takes lengths[7/2] (0-indexed) = 5 -> 2560-char budget.
                @test model.median == 5
                limit = Model2Vec.MAX_LENGTH * model.median
                # Each 20-char word tokenizes to one real "▁" piece + one *fused* UNK run, so
                # 128 words = 256 tokens, comfortably under the 512-token cap: the char budget
                # binds *before* the token cap, making pre-truncation observable end-to-end.
                head = ("z"^19 * " ")^(limit ÷ 20) # exactly `limit` chars
                @test encode(model, head * "dog") == encode(model, head) # "dog" past the budget: dropped
                @test encode(model, ("z"^19 * " ")^(limit ÷ 20 - 1) * "z"^15 * " dog ") != encode(model, head) # inside: kept

                # long invalid UTF-8 through the full Unigram path (truncation + charsmap sanitize)
                invalid = String(repeat(UInt8[0x63, 0x61, 0x74, 0xff, 0x20], 3 * Model2Vec.MAX_LENGTH))
                @test ncodeunits(invalid) > limit
                @test !isvalid(invalid)
                v = encode!(Scratch(model), model, invalid)
                @test length(v) == model.dim
                @test !any(isnan, v)
            end
        end

        mktempdir() do dir
            cmdir = buildunigramfixture(joinpath(dir, "ugcm"); charsmap=true)
            model = load(cmdir)

            @testset "synthetic Precompiled charsmap: darts traversal semantics" begin
                @test length(model.charsmap.trie) == 240 # found through the nested Sequence tree
                scratch = Scratch(model)
                @test cmfold(model, scratch, "A") == "a"      # single-byte key
                @test cmfold(model, scratch, "…") == "..."    # multi-byte key, longer replacement
                @test cmfold(model, scratch, "B") == "B"      # no entry: passthrough
                @test cmfold(model, scratch, "cAt…B") == "cat...B"
                # first-(shortest-)match semantics, straight from spm_precompiled's results[0]: a
                # grapheme cluster < 6 bytes is replaced whole by its shortest matching *prefix*'s
                @test cmfold(model, scratch, "A\u0301") == "a"
                @test cmfold(model, scratch, "Á") == "a"
                @test cmfold(model, scratch, "A\u0301\u0301\u0301") == "a\u0301\u0301\u0301"
                @test cmfold(model, scratch, "Á́́") == "á́́"
                # lead byte beyond the (deliberately small) trie array: out-of-range -> miss
                @test cmfold(model, scratch, "😀A") == "😀a"
                # NUL bytes stop the prefix search (spm_precompiled parity) and pass through
                @test cmfold(model, scratch, "\0A") == "\0a"
                # CR+LF is the one ASCII pair that joins into a single cluster (misses -> per-char)
                @test cmfold(model, scratch, "A\r\nA") == "a\r\na"
                # malformed byte between mapped chars -> U+FFFD, neighbors still folded
                @test cmfold(model, scratch, String(UInt8[0x41, 0xff, 0x41])) == "a�a"
                # end-to-end: folding runs before tokenization ('A' reaches the vocab as 'a')
                @test encode(model, "cAt") == encode(model, "cat")
            end
        end
    end
end
