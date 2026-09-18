#!/usr/bin/env bash
set -euo pipefail

architecture=${1:?usage: run_llm_acceptance.sh ARCHITECTURE}
wheel=$(find input -maxdepth 1 -name 'radeon_kernels-*.whl' -print -quit)
test -n "$wheel"
python3 -m pip install --no-deps --force-reinstall "$wheel"

for operator in rms_norm add_rms_norm rope swiglu softmax; do
    release_operator=${operator//_/-}
    archive=$(find input -maxdepth 1 \
        -name "rk-${release_operator}-${architecture}-m5-20260917-0.1.0.tar.gz" \
        -print -quit)
    test -n "$archive"
    python3 -m radeon_kernels.lab.cli pack install \
        --operator "$operator" \
        --semantic-version 1.0 \
        --archive "$archive" \
        --destination packs
done

RADEON_KERNELS_PACK_PATH=$PWD/packs \
    python3 input/verify_llm_release.py \
    --expected-architecture "$architecture" \
    --output "output/acceptance-${architecture}.json"
