"""Exception hierarchy for ComfyTrace."""


class ComfyTraceError(Exception):
    """Base exception for all ComfyTrace errors."""


class IngestionError(ComfyTraceError):
    """Raised when memory dump ingestion or layer setup fails."""


class SymbolResolutionError(IngestionError):
    """Raised when Windows kernel symbol tables cannot be resolved offline."""


class ProcessNotFoundError(IngestionError):
    """Raised when target ComfyUI PID is absent or terminated."""


class RegenerationError(ComfyTraceError):
    """Base exception for workflow regeneration failures."""


class RunnerConnectionError(RegenerationError):
    """Raised when ComfyUI runner HTTP endpoint is unreachable."""


class PromptSubmissionError(RegenerationError):
    """Raised when API prompt graph is rejected by runner /prompt endpoint."""


class ExecutionTimeoutError(RegenerationError):
    """Raised when generation execution exceeds configured timeout."""


class MissingDependencyError(RegenerationError):
    """Raised when optional [regen] dependencies (imagehash, scikit-image) are missing."""
