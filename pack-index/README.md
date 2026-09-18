# Kernel Pack Index

This directory indexes signed precompiled kernel packs by architecture and ROCm
ABI. Pack records refer to immutable release artifacts by SHA-256; compiled
binaries are not committed here.

Consumers first verify the detached index signature, then the release SHA-256,
and finally the manifest signature inside the archive. `release_file` is
resolved by the release channel that publishes the index and archive together.
