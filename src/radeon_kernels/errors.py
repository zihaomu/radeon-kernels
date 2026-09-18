class RadeonKernelsError(Exception):
    """Base error for expected user-facing failures."""


class ConfigError(RadeonKernelsError):
    """Raised when workspace or target configuration is invalid."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")

