#!/usr/bin/env python3
"""Generate a private Ed25519 release key and print its public key record."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from developer.promotion.signing import generate_private_key


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--key-id", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    public_record = generate_private_key(args.output, args.key_id)
    print(json.dumps(public_record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
