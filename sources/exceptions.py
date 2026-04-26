"""Custom exceptions for the PyTorch Architecture Viewer."""


class AppError(Exception):
    """Base exception for all application errors."""


class RepoCloneError(AppError):
    """Raised when git clone fails (bad URL, network error, auth)."""


class ModelImportError(AppError):
    """Raised when dynamic import of a model file fails."""


class TraceError(AppError):
    """Raised when torch.fx symbolic trace fails."""


class NoModelError(AppError):
    """Raised when no nn.Module subclasses are found in a repository."""
