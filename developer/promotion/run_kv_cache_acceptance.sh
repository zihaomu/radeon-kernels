#!/usr/bin/env bash
set -euo pipefail

architecture=${1:?usage: run_kv_cache_acceptance.sh ARCHITECTURE}
test -d repo/src/radeon_kernels
export PYTHONPATH="$PWD/repo/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p packs output
for operator in append copy; do
    for archive in input/rk-kv-cache-${operator}-${architecture}-m6-20260918-*.tar.gz; do
        test -f "$archive"
        release_file=$(basename "$archive")
        python3 -m radeon_kernels.lab.cli pack install \
            --operator "kv_cache_${operator}" \
            --semantic-version 1.0 \
            --release-file "$release_file" \
            --archive "$archive" \
            --destination packs
    done
done

RADEON_KERNELS_PACK_PATH=$PWD/packs \
    python3 input/verify_kv_cache_release.py \
    --expected-architecture "$architecture" \
    --output "output/acceptance-${architecture}.json"
