"""Shared configuration and failure policy for Mistral SDK boundaries."""

import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

from mistralai.client import Mistral
from mistralai.client.errors import MistralError


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
