"""Static dispatch runtime for published Radeon kernel packs."""

from radeon_kernels.runtime.dispatcher import (
    DispatchDecision,
    Dispatcher,
    DispatchRequest,
)
from radeon_kernels.runtime.explain import explain_last_dispatch, get_last_dispatch
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.installer import (
    IndexedPack,
    PackInstallError,
    compatible_records,
    fetch_release,
    index_records,
    install_release,
    verify_installed_pack,
)
from radeon_kernels.runtime.loader import PrecompiledArtifactLoader
from radeon_kernels.runtime.pack import KernelPack
from radeon_kernels.runtime.packs import (
    KernelPackRegistry,
    PackResolutionError,
    default_pack_roots,
)
from radeon_kernels.runtime.registry import DispatchRegistry
from radeon_kernels.runtime.resources import (
    builtin_dispatch_registry,
    builtin_pack_index,
    builtin_pack_indexes,
)
from radeon_kernels.runtime.signing import (
    SignatureVerificationError,
    TrustedKeyring,
    builtin_trusted_keyring,
    verify_detached_content,
    verify_detached_file,
)

__all__ = [
    "DispatchDecision",
    "DispatchRegistry",
    "DispatchRequest",
    "Dispatcher",
    "EnvironmentFingerprint",
    "IndexedPack",
    "KernelPack",
    "KernelPackRegistry",
    "PackResolutionError",
    "PackInstallError",
    "PrecompiledArtifactLoader",
    "SignatureVerificationError",
    "TrustedKeyring",
    "builtin_dispatch_registry",
    "builtin_pack_index",
    "builtin_pack_indexes",
    "builtin_trusted_keyring",
    "default_pack_roots",
    "compatible_records",
    "explain_last_dispatch",
    "get_last_dispatch",
    "fetch_release",
    "index_records",
    "install_release",
    "verify_installed_pack",
    "verify_detached_content",
    "verify_detached_file",
]
