from collections.abc import Sequence
from typing import Any

from food_ordering.model_adapter import AssistantMessage, ModelMessage, ToolSpec
from food_ordering.submission import SubmissionResult


class ScriptedModel:
    def __init__(self, *responses: AssistantMessage) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []

    def complete(
        self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
    ) -> AssistantMessage:
        self.requests.append(tuple(messages))
        return next(self._responses)


class FailingAfterScriptModel(ScriptedModel):
    def complete(
        self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
    ) -> AssistantMessage:
        try:
            return super().complete(messages=messages, tools=tools)
        except StopIteration:
            raise RuntimeError("controlled provider failure") from None


class RecordingSubmitter:
    def __init__(self, *outcomes: SubmissionResult) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []

    def submit(self, payload: dict[str, Any]) -> SubmissionResult:
        self.calls.append(payload)
        return next(self._outcomes)
