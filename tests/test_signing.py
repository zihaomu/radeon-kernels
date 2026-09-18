from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from radeon_kernels.runtime.signing import (
    SignatureVerificationError,
    TrustedKeyring,
    sha256_file,
    signature_document,
    verify_detached_file,
)


class SigningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.signed = self.root / "manifest.json"
        self.signed.write_text('{"pack_id":"test"}\n', encoding="utf-8")
        self.private_key = Ed25519PrivateKey.generate()
        public_bytes = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        import base64

        self.keyring = TrustedKeyring.from_mapping(
            {
                "schema_version": "1.0",
                "keys": [
                    {
                        "key_id": "test-release-key",
                        "algorithm": "ed25519",
                        "public_key": base64.b64encode(public_bytes).decode("ascii"),
                        "status": "active",
                        "created_at": "2026-09-17T00:00:00+00:00",
                    }
                ],
            }
        )
        self.signature_path = self.root / "manifest.sig.json"
        self.signature_path.write_text(
            json.dumps(
                signature_document(
                    private_key=self.private_key,
                    key_id="test-release-key",
                    signed_file="manifest.json",
                    signed_sha256=sha256_file(self.signed),
                )
            ),
            encoding="utf-8",
        )

    def test_verifies_valid_detached_signature(self) -> None:
        envelope = verify_detached_file(self.signed, self.signature_path, self.keyring)

        self.assertEqual(envelope.key_id, "test-release-key")

    def test_rejects_tampered_signed_file(self) -> None:
        self.signed.write_text('{"pack_id":"tampered"}\n', encoding="utf-8")

        with self.assertRaisesRegex(SignatureVerificationError, "SHA-256 mismatch"):
            verify_detached_file(self.signed, self.signature_path, self.keyring)

    def test_rejects_signature_for_different_filename(self) -> None:
        with self.assertRaisesRegex(SignatureVerificationError, "expected 'index.json'"):
            verify_detached_file(
                self.signed,
                self.signature_path,
                self.keyring,
                signed_file="index.json",
            )

    def test_rejects_revoked_key(self) -> None:
        key = next(iter(self.keyring._keys.values()))
        revoked = TrustedKeyring([type(key)(key.key_id, key.public_key, "revoked", key.created_at)])

        with self.assertRaisesRegex(SignatureVerificationError, "is not active"):
            verify_detached_file(self.signed, self.signature_path, revoked)


if __name__ == "__main__":
    unittest.main()
