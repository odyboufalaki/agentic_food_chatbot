"""Shared configuration and failure policy for Mistral SDK boundaries."""

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import MistralError


Result = TypeVar("Result")


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(default=None, repr=False)
    model: str = "mistral-small-latest"
    timeout_ms: int = 20_000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        )


class ModelFailure(Exception):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class RetryableModelFailure(Exception):
    """A response-level failure allowed to share the bounded request retry."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


def validate_configuration(settings: Settings, client: Mistral | None) -> None:
    if (
        not settings.model.strip()
        or settings.timeout_ms <= 0
        or (client is None and not (settings.api_key or "").strip())
    ):
        raise ModelFailure("configuration")


def client_context(
    settings: Settings,
    client: Mistral | None,
) -> AbstractContextManager[Mistral]:
    if client is not None:
        return nullcontext(client)
    return Mistral(
        api_key=settings.api_key,
        retry_config=None,
        timeout_ms=settings.timeout_ms,
    )


def error_category(error: MistralError) -> str:
    if error.status_code in {401, 403}:
        return "authentication"
    if error.status_code in {400, 404, 422}:
        return "configuration"
    if error.status_code == 429:
        return "rate_limit"
    if error.status_code in {408, 500, 502, 503, 504}:
        return "model_unavailable"
    return "model_error"


def run_bounded_request(
    settings: Settings,
    client: Mistral | None,
    request: Callable[[Mistral], Result],
) -> Result:
    """Run at most two attempts under the shared Mistral failure policy."""

    validate_configuration(settings, client)
    with client_context(settings, client) as active_client:
        for attempt in range(2):
            try:
                return request(active_client)
            except RetryableModelFailure as failure:
                category = failure.category
            except httpx.TimeoutException:
                category = "model_timeout"
            except httpx.TransportError:
                category = "model_unavailable"
            except MistralError as error:
                category = error_category(error)
                if category in {"authentication", "configuration", "model_error"}:
                    raise ModelFailure(category) from None
            if attempt == 1:
                raise ModelFailure(category) from None
    raise ModelFailure("model_error")
