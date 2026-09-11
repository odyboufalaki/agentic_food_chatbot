"""Public synchronous facade for the food-ordering conversation."""

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

from food_ordering.menu import load_menu
from food_ordering.mistral_adapter import MistralToolModel
from food_ordering.model_adapter import AssistantMessage, ToolCall, ToolResultMessage, TurnModel
from food_ordering.session import Session
from food_ordering.submission import MCPSubmitter, SubmissionResult, Submitter
from food_ordering.tool_protocol import Malformed, Operation, TOOL_PROTOCOL_VERSION
from food_ordering.turn_logging import TurnLogger
from food_ordering.turn_processor import TurnProcessor


class SessionStateSnapshot(TypedDict):
    status: str
    revision: int
    reviewed_revision: int | None
    total_cents: int


class FoodOrderAgent:
    """Keep conversation state and delegate each customer turn to TurnProcessor."""

    def __init__(
        self,
        *,
        model: TurnModel | None = None,
        log_path: Path | None = None,
        submitter: Submitter | None = None,
        session: Session | None = None,
    ) -> None:
        self._model = model if model is not None else MistralToolModel()
        self._menu = load_menu()
        self._session = session if session is not None else Session()
        self._logger = TurnLogger(log_path)
        self._submitter = submitter if submitter is not None else MCPSubmitter()

    def send(self, message: str) -> dict[str, Any]:
        """Process one customer turn and return its customer-visible result."""

        started = perf_counter()
        before = self._state_snapshot()
        validated_operations: list[tuple[ToolCall, Operation]] = []
        submission_calls: list[dict[str, Any]] = []

        def observe_operation(call: ToolCall, operation: Operation) -> None:
            validated_operations.append((call, operation))

        def observe_submission(
            payload: dict[str, Any], outcome: SubmissionResult,
        ) -> None:
            if outcome.invoked:
                submission_calls.append({
                    "name": "submit_order",
                    "arguments": payload,
                    "result": outcome.result,
                })

        if not isinstance(message, str) or not message.strip():
            self._session.turn_id += 1
            response: dict[str, Any] = {
                "message": "Please enter a nonempty message.",
            }
            error_category: str | None = "invalid_input"
            turn_messages: tuple[object, ...] = ()
        else:
            processor = TurnProcessor(
                model=self._model,
                menu=self._menu,
                session=self._session,
                submitter=self._submitter,
                operation_observer=observe_operation,
                submission_observer=observe_submission,
            )
            error_category = None
            try:
                response = processor.process(message)
            except Exception:
                # The processor contains expected failures. This guard protects the
                # public facade from unexpected integration defects without exposing them.
                self._session.invalidate_review()
                response = {
                    "message": (
                        "The order may have been accepted. Please check with the "
                        "restaurant; this session will not submit again."
                        if self._session.status == "uncertain"
                        else "I could not process that request safely. Please try again."
                    ),
                }
                error_category = "internal_error"
            turn_messages = self._session.latest_turn_messages(message)

        if submission_calls:
            response["tool_calls"] = submission_calls
        if not isinstance(response.get("message"), str) or not response["message"].strip():
            response["message"] = "I could not finish that request safely. Please try again."
            error_category = error_category or "empty_response"

        self._write_turn_record(
            message=message,
            response=response,
            before=before,
            turn_messages=turn_messages,
            validated_operations=validated_operations,
            submission_calls=submission_calls,
            error_category=error_category,
            elapsed_ms=round((perf_counter() - started) * 1000, 3),
        )
        return response

    def _state_snapshot(self) -> SessionStateSnapshot:
        return {
            "status": self._session.status,
            "revision": self._session.revision,
            "reviewed_revision": self._session.reviewed_revision,
            "total_cents": sum(line.total_cents for line in self._session.lines),
        }

    def _write_turn_record(
        self,
        *,
        message: object,
        response: dict[str, Any],
        before: SessionStateSnapshot,
        turn_messages: tuple[object, ...],
        validated_operations: list[tuple[ToolCall, Operation]],
        submission_calls: list[dict[str, Any]],
        error_category: str | None,
        elapsed_ms: float,
    ) -> None:
        model_calls: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        parse_failures: list[dict[str, Any]] = []
        model_iterations = 0
        for entry in turn_messages:
            if isinstance(entry, AssistantMessage):
                model_iterations += 1
                model_calls.extend({
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                } for call in entry.tool_calls)
            elif isinstance(entry, ToolResultMessage):
                serialized = entry.payload.model_dump(mode="json", exclude_none=True)
                outcome = {
                    "call_id": entry.call_id,
                    "name": entry.name,
                    **serialized,
                }
                outcomes.append(outcome)
                if isinstance(entry.payload, Malformed):
                    parse_failures.append(outcome)
        after = self._state_snapshot()
        self._logger.write({
            "tool_protocol_version": TOOL_PROTOCOL_VERSION,
            "session_id": self._session.session_id,
            "turn_id": self._session.turn_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "input": message if isinstance(message, str) else None,
            "response": response,
            "model_calls": model_calls,
            "validated_operations": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": operation.model_dump(mode="json", exclude_none=True),
                }
                for call, operation in validated_operations
            ],
            "parse_failures": parse_failures,
            "outcomes": outcomes,
            "commit_effects": [
                outcome["effect"]
                for outcome in outcomes
                if outcome.get("outcome") == "APPLIED"
            ],
            "totals": {
                "before_cents": before["total_cents"],
                "after_cents": after["total_cents"],
            },
            "state_transition": {"before": before, "after": after},
            "model_budget": {
                "iterations": model_iterations,
                "tool_calls": len(model_calls),
            },
            "tool_calls": submission_calls,
            "submission_outcome": (
                self._session.submission_outcome.model_dump(
                    mode="json", exclude_none=True,
                )
                if self._session.submission_outcome is not None else None
            ),
            "error_category": error_category,
            "elapsed_ms": elapsed_ms,
        })
