using Model2Vec
using Test

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
end
