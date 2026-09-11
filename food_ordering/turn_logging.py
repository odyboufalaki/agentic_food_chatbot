import json
import os
import re
import sys
from pathlib import Path
from typing import Any


class TurnLogger:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else Path(os.getenv("FOOD_ORDER_LOG_PATH", "logs/turns.jsonl"))
        self._secrets = tuple(value for value in [os.getenv("MISTRAL_API_KEY")] if value)

    @staticmethod
    def _sensitive_key(key: object) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        return any(marker in normalized for marker in (
            "authorization", "apikey", "token", "secret", "password", "cookie", "header",
        ))

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return re.sub(
                r"(?i)(bearer\s+)[^\s,;]+",
                r"\1[REDACTED]",
                value,
            )
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if self._sensitive_key(key) else self._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def write(self, record: dict[str, Any]) -> None:
        try:
            encoded = json.dumps(self._redact(record), ensure_ascii=True, allow_nan=False)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        except (OSError, ValueError, TypeError):
            # Never include exception text: it can contain paths or customer data.
            try:
                print("Turn logging failed; the conversation result is unchanged.", file=sys.stderr)
            except (OSError, ValueError):
                pass
