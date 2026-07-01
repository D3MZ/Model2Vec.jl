#!/usr/bin/env bash
# Reproduce the Julia-vs-Rust benchmark end-to-end for both tokenizer families.
#   ./bench/run.sh [records]
set -euo pipefail
cd "$(dirname "$0")/.."
RECORDS="${1:-4000}"

echo "==> generating corpus ($RECORDS records)"
julia bench/make_corpus.jl "$RECORDS" bench/corpus.txt

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
