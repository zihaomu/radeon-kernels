#!/usr/bin/env bash
set -euo pipefail

architecture=${1:?usage: run_paged_attention_acceptance.sh ARCHITECTURE}
expected_winners=${2:?usage: run_paged_attention_acceptance.sh ARCHITECTURE EXPECTED_WINNERS}
test -d repo/src/radeon_kernels
export PYTHONPATH="$PWD/repo/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p packs output
found=0
for archive in input/rk-paged-attention-decode-${architecture}-m6-20260918-*.tar.gz; do
    test -f "$archive" || continue
    found=1
    release_file=$(basename "$archive")
    python3 -m radeon_kernels.lab.cli pack install \
        --operator paged_attention_decode \
        --semantic-version 1.0 \
        --release-file "$release_file" \
        --archive "$archive" \
        --destination packs
done
test "$found" -eq 1

RADEON_KERNELS_PACK_PATH=$PWD/packs \
    python3 input/verify_paged_attention_release.py \
    --expected-architecture "$architecture" \
    --expected-winners "$expected_winners" \
    --output "output/acceptance-${architecture}.json"
