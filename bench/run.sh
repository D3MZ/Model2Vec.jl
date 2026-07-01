#!/usr/bin/env bash
# Reproduce the Julia-vs-Rust benchmark end-to-end for both tokenizer families.
#   ./bench/run.sh [records] [wet_file]
# wet_file defaults to a sibling MonsieurPapin checkout's data/warc.wet.gz if present; otherwise
# falls back to the synthetic corpus generator (bench/make_corpus.jl).
set -euo pipefail
cd "$(dirname "$0")/.."
RECORDS="${1:-4000}"
WETFILE="${2:-../MonsieurPapin/data/warc.wet.gz}"

if [ -f "$WETFILE" ]; then
    echo "==> extracting corpus from real WET file ($RECORDS records, $WETFILE)"
    julia --project=bench bench/extract_wet_corpus.jl "$WETFILE" "$RECORDS" bench/corpus.txt
else
    echo "==> no WET file found at $WETFILE; falling back to synthetic corpus ($RECORDS records)"
    julia bench/make_corpus.jl "$RECORDS" bench/corpus.txt
fi

echo "==> building native Rust reference (release, lto)"
( cd bench/rust_ref && cargo build --release --quiet )

echo "==> running benchmark: WordPiece (potion-base-8M)"
D8M=$(ls -d ~/.cache/huggingface/hub/models--minishlab--potion-base-8M/snapshots/* | head -1)
julia --project=. -t 1 bench/bench.jl bench/corpus.txt "$D8M"

echo "==> running benchmark: Unigram (potion-multilingual-128M)"
D128M=$(ls -d ~/.cache/huggingface/hub/models--minishlab--potion-multilingual-128M/snapshots/* | head -1)
julia --project=. -t 1 bench/bench.jl bench/corpus.txt "$D128M"

echo "==> plotting"
julia bench/plot.jl
