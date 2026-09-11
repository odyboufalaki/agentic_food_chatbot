"""Provider-neutral model messages used by turn orchestration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from food_ordering.tool_protocol import ModelResult, StrictModel


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
    completion_status: Literal["complete", "truncated"] = "complete"


AbortReason: TypeAlias = Literal[
    "customer_input_required",
    "model_response_truncated",
    "tool_call_budget_exhausted",
]


class AbortedToolResult(StrictModel):
    """Orchestration result for an emitted call that was not dispatched."""

    error: Literal["turn_aborted"]
    reason: AbortReason
    resolution: str


ToolResultPayload: TypeAlias = ModelResult | AbortedToolResult


@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    name: str
    payload: ToolResultPayload


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
