"""Exceptions of the extraction layer. Messages never contain credentials."""

from __future__ import annotations


class ExtractionError(Exception):
    """Base of all extraction-layer errors."""


class ModelConfigError(ExtractionError):
    """A model configuration is missing, invalid, inactive or lacks its credentials."""


class ProviderError(ExtractionError):
    """The LLM provider failed. `retryable` says whether the same request may be sent again."""

    code = "provider_error"
    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    retryable = True


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    retryable = True


class ProviderConnectionError(ProviderError):
    """The service could not be reached, or answered with a server error."""

    code = "provider_unavailable"
    retryable = True


class ProviderResponseError(ProviderError):
    """The provider answered, but not in its documented response format."""

    code = "provider_malformed_response"
    retryable = True


class ProviderAuthError(ProviderError):
    code = "provider_auth_failed"


class ProviderRequestError(ProviderError):
    """The provider rejected the request itself (bad parameters, too long). Retrying will not help."""

    code = "provider_request_rejected"
