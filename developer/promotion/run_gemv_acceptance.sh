#!/usr/bin/env bash
set -euo pipefail

architecture=${1:?usage: run_gemv_acceptance.sh ARCHITECTURE}
test -d repo/src/radeon_kernels
export PYTHONPATH="$PWD/repo/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p packs output
for archive in input/rk-gemv-${architecture}-m6-20260918-*.tar.gz; do
    test -f "$archive"
    release_file=$(basename "$archive")
    python3 -m radeon_kernels.lab.cli pack install \
        --operator gemv \
        --semantic-version 1.0 \
        --release-file "$release_file" \
        --archive "$archive" \
        --destination packs
done

RADEON_KERNELS_PACK_PATH=$PWD/packs \
    python3 input/verify_gemv_release.py \
    --expected-architecture "$architecture" \
    --output "output/acceptance-${architecture}.json"
