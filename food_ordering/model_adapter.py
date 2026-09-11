"""Provider-neutral model messages used by turn orchestration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: object


@dataclass(frozen=True)
class CustomerMessage:
    content: str


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    name: str
    payload: Mapping[str, Any]


ModelMessage: TypeAlias = CustomerMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True)
class TranscriptTurn:
    messages: tuple[ModelMessage, ...]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    parameters: Mapping[str, Any]
    description: str = ""


class TurnModel(Protocol):
    def complete(
        self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
    ) -> AssistantMessage: ...
