# Extracts real page text from a Common Crawl WET file into bench/corpus.txt (one record per
# line), replacing the old synthetic LCG-generated corpus so the benchmark reflects real web
# text (mixed record lengths, real punctuation/whitespace patterns, real multilingual content --
# not a fixed rotation of 12 template sentences).
#
# Standalone: does not depend on MonsieurPapin (no package dependency), just enough of the WET
# body-extraction logic (WARC-Type: conversion records, "Content-Length:"-delimited bodies) to
# pull out real text. Requires CodecZlib for gunzip (a light, common dependency; not added to
# Project.toml since this is a bench-only, not package, concern -- install it into your global
# environment or a temp one if you don't already have it).
#
#   julia bench/extract_wet_corpus.jl <path/to/warc.wet.gz> [n_records] [out]
#
# If no WET file is given, looks for a sibling MonsieurPapin checkout's data/warc.wet.gz
# (../MonsieurPapin/data/warc.wet.gz) -- the repo this package's use case comes from, and a
# convenient source of a real, already-downloaded WET sample during development.
using CodecZlib

function findwet()
    candidate = joinpath(@__DIR__, "..", "..", "MonsieurPapin", "data", "warc.wet.gz")
    isfile(candidate) && return candidate
    error("no WET file given and no sibling MonsieurPapin checkout found at $candidate; " *
          "pass a path: julia bench/extract_wet_corpus.jl <path/to/warc.wet.gz>")
end

# Minimal WET record body extractor: each record is `WARC/1.0`, a header block (blank-line-
# terminated), then exactly `Content-Length` bytes of body, then a variable number of blank
# lines before the next record's `WARC/1.0` (not a fixed separator width, so we skip blank lines
# rather than reading a fixed byte count). Keeps only `WARC-Type: conversion` records (the
# extracted-plaintext ones -- what a tokenizer benchmark cares about).
# Reads lines, skipping blanks, until it finds a non-empty one (the next record's `WARC/1.0`,
# or eof). No lookahead/pushback needed since that first non-empty line is exactly what the
# caller wants next.
function nextnonblank(io)
    while !eof(io)
        line = readline(io)
        isempty(line) || return line
    end
    ""
end

# MonsieurPapin caps WET content at 12,000 bytes/record in production (src/wets.jl's
# `contentlimit`) -- matching that here keeps the benchmark corpus realistic (what a tokenizer
# actually sees in that pipeline) and the file a sane size (uncapped, 4,000 records ran ~44 MB).
const CONTENTLIMIT = 12_000

function extractbodies(path::AbstractString, n::Int)
    io = GzipDecompressorStream(open(path))
    bodies = String[]
    try
        while !eof(io) && length(bodies) < n
            firstline = nextnonblank(io)
            eof(io) && isempty(firstline) && break
            startswith(firstline, "WARC/1.0") || error("expected WARC/1.0, got $(repr(firstline))")

            headerlines = String[]
            while !eof(io)
                line = readline(io)
                isempty(line) && break
                push!(headerlines, line)
            end
            isconversion = any(l -> startswith(l, "WARC-Type:") && contains(l, "conversion"), headerlines)
            lenline = findfirst(l -> startswith(l, "Content-Length:"), headerlines)
            len = lenline === nothing ? 0 : parse(Int, strip(split(headerlines[lenline], ':', limit=2)[2]))
            bytes = read(io, len)
            if isconversion && len > 0
                kept = bytes[1:min(length(bytes), CONTENTLIMIT)]
                # Trim to the last valid UTF-8 boundary so a hard byte cut doesn't split a
                # multi-byte character (mirrors src/wets.jl's utf8boundary).
                while !isempty(kept) && (kept[end] & 0xc0) == 0x80
                    kept = @view kept[1:end-1]
                end
                push!(bodies, String(kept))
            end
        end
    finally
        close(io)
    end
    bodies
end

function main()
    wetpath = length(ARGS) >= 1 ? ARGS[1] : findwet()
    n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000
    out = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "corpus.txt")

    bodies = extractbodies(wetpath, n)
    open(out, "w") do io
        for body in bodies
            # one record per line: real WET bodies can contain newlines, which would otherwise
            # be misread as extra records by bench.jl's readlines().
            println(io, replace(body, '\n' => ' ', '\r' => ' '))
        end
    end
    println("wrote $out ($(length(bodies)) real WET records, $(filesize(out)) bytes)")
end
main()
