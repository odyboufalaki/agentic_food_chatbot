import json
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from food_ordering.clarification import validate_pending_resolution
from food_ordering.interpretation import ClarificationRenderer, Interpreter, MistralInterpreter, ModelFailure, ResponseRenderer, TemplateClarificationRenderer, TemplateResponseRenderer
from food_ordering.menu import load_menu, menu_context
from food_ordering.order import ClarificationContext, ClarificationNeeded, InvalidSelection, OrderLine, ResponseContext, build_clarification_context, change_quantity, display_groups, edit_servings, normalize, render_draft, render_menu, resolve_target, submission_payload
from food_ordering.proposals import AbandonPending, Add, CancelPending, ChangeQuantity, Clarify, ClearDraft, Edit, MenuQuestion, NewOrder, Proposal, RemoveLine, RetrySubmission, Review, SetInstructions, Submit, Summary, Unsupported
from food_ordering.submission import MCPSubmitter, Submitter, render_receipt
from food_ordering.turn_logging import TurnLogger


DRAFT_CHANGE_TYPES = (Add, Edit, ChangeQuantity, RemoveLine, ClearDraft, NewOrder, SetInstructions)


@dataclass(frozen=True)
class PendingChange:
    original_message: str
    proposal: Proposal
    clarification: ClarificationContext

    def snapshot(self) -> dict[str, Any]:
        return {"original_message": self.original_message, "proposal": self.proposal.model_dump(),
                "reason": self.clarification.reason,
                "question": self.clarification.fallback_question,
                "subject": self.clarification.subject, "field": self.clarification.field,
                "choices": list(self.clarification.choices)}


class TurnHandled(Exception):
    def __init__(self, response: dict[str, Any], operations: list[dict[str, Any]]) -> None:
        self.response = response
        self.operations = operations


class FoodOrderAgent:
    def __init__(self, *, interpreter: Interpreter | None = None, log_path: Path | None = None,
                 submitter: Submitter | None = None,
                 clarification_renderer: ClarificationRenderer | None = None,
                 response_renderer: ResponseRenderer | None = None) -> None:
        default_interpreter = interpreter is None
        self._interpreter = interpreter if interpreter is not None else MistralInterpreter()
        self._clarification_renderer = (clarification_renderer if clarification_renderer is not None
                                        else self._interpreter if isinstance(self._interpreter, MistralInterpreter)
                                        else TemplateClarificationRenderer())
        self._response_renderer = (response_renderer if response_renderer is not None
                                   else self._interpreter if default_interpreter
                                   and isinstance(self._interpreter, MistralInterpreter)
                                   else TemplateResponseRenderer())
        self._menu = load_menu()
        self._lines: list[OrderLine] = []
        self._instructions = ""
        self._next_line_number = 1
        self._history: deque[dict[str, str]] = deque(maxlen=12)
        self._logger = TurnLogger(log_path)
        self._session_id = uuid4().hex
        self._turn_id = 0
        self._revision = 0
        self._submitter = submitter if submitter is not None else MCPSubmitter()
        self._reviewed_revision: int | None = None
        self._status = "draft"
        self._receipt_message = ""
        self._rejected_payload: dict[str, Any] | None = None
        self._application_error_payload: dict[str, Any] | None = None
        self._retry_requires_review = False
        self._pending_change: PendingChange | None = None

    def send(self, message: str) -> dict[str, Any]:
        started = perf_counter()
        self._turn_id += 1
        before_revision = self._revision
        before_total = sum(line.total_cents for line in self._lines)
        before_status = self._status
        before_reviewed_revision = self._reviewed_revision
        before_pending_change = self._pending_change is not None
        operations: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        error_category = None
        draft_changed = False
        failed_conversation_turn = False
        pending_at_start = self._pending_change
        abandoned_pending = False
        preserve_pending_proposal = False
        lifecycle_operations: list[dict[str, Any]] = []
        proposal: Proposal | None = None
        interpreted_proposal: Proposal | None = None
        try:
            if not isinstance(message, str) or not message.strip():
                raise InvalidSelection("Please enter a nonempty message.")
            proposal = Proposal.model_validate(self._interpreter.interpret(
                message=message, menu=menu_context(self._menu),
                draft=[line.snapshot() for line in self._lines], history=[dict(entry) for entry in self._history],
                order_state={"status": self._status, "revision": self._revision,
                             "reviewed_revision": self._reviewed_revision,
                             "instructions": self._instructions,
                             "display_groups": [[line.line_id for line in group]
                                                for group in display_groups(self._lines)]},
                pending_clarification=pending_at_start.snapshot() if pending_at_start is not None else None,
            ))
            interpreted_proposal = proposal
            if any(isinstance(operation, CancelPending) for operation in proposal.operations):
                if pending_at_start is None or len(proposal.operations) != 1:
                    raise InvalidSelection("There is no single pending change to cancel.")
                self._pending_change = None
                raise TurnHandled(
                    {"message": "Pending change canceled.\n" + render_draft(self._lines, self._instructions)},
                    [{"type": "cancel_pending"}],
                )
            abandoning = any(isinstance(operation, AbandonPending) for operation in proposal.operations)
            if abandoning:
                if pending_at_start is None:
                    raise InvalidSelection("There is no pending change to abandon.")
                if not isinstance(proposal.operations[0], AbandonPending):
                    raise InvalidSelection("Abandon the pending change before making a new request.")
                self._pending_change = None
                abandoned_pending = True
                remaining = proposal.operations[1:]
                if not remaining:
                    raise TurnHandled(
                        {"message": "Pending change abandoned.\n" + render_draft(self._lines, self._instructions)},
                        [{"type": "abandon_pending"}],
                    )
                proposal = Proposal(operations=remaining)
                lifecycle_operations.append({"type": "abandon_pending"})
            clears_pending = any(isinstance(operation, ClearDraft) for operation in proposal.operations)
            if (pending_at_start is not None and not abandoned_pending and not clears_pending
                    and any(isinstance(operation, DRAFT_CHANGE_TYPES) for operation in proposal.operations)):
                validate_pending_resolution(pending_at_start.proposal, proposal, self._menu)
            clarification = next((operation for operation in proposal.operations if isinstance(operation, Clarify)), None)
            if clarification is not None:
                preserve_pending_proposal = (
                    pending_at_start is not None and not abandoned_pending
                    and all(isinstance(operation, Clarify) for operation in proposal.operations)
                )
                raise ClarificationNeeded(build_clarification_context(
                    clarification.reason, self._menu,
                    item_id=clarification.item_id, field=clarification.field,
                ))
            if pending_at_start is not None and not abandoned_pending and any(
                isinstance(operation, (Submit, Review, RetrySubmission)) for operation in proposal.operations
            ) and not any(isinstance(operation, DRAFT_CHANGE_TYPES) for operation in proposal.operations):
                preserve_pending_proposal = True
                raise ClarificationNeeded(pending_at_start.clarification)
            attempts_change = any(isinstance(operation, DRAFT_CHANGE_TYPES) for operation in proposal.operations)
            if attempts_change:
                self._reviewed_revision = None
                if self._status == "uncertain":
                    raise InvalidSelection("The order may already have been accepted. Check with the restaurant before starting another order.")
                if self._status == "submitted" and not isinstance(proposal.operations[0], NewOrder):
                    raise InvalidSelection("This order was already accepted. Explicitly start a new order to add food; accepted orders cannot be edited or canceled here.")
            if any(isinstance(operation, NewOrder) for operation in proposal.operations[1:]):
                raise InvalidSelection("Start a new order before specifying its selections.")
            for operation in proposal.operations:
                if isinstance(operation, Unsupported):
                    explanations = {
                        "not_available": "That request is not available in this version. You can ask about the menu, add or edit selections, remove servings, clear your draft, or review it.",
                        "unclear": "Please specify the menu items and changes you want in a complete request.",
                        "dietary_guarantee": "The menu does not verify ingredients or dietary guarantees. Please check with the restaurant.",
                    }
                    raise InvalidSelection(explanations[operation.reason])
            candidate = list(self._lines)
            instructions = self._instructions
            next_line_number = self._next_line_number
            validated_operations = []
            answers = []
            changed = False
            for operation in proposal.operations:
                if isinstance(operation, Add):
                    line = normalize(operation, self._menu, line_id=f"L{next_line_number}")
                    next_line_number += 1
                    candidate.append(line)
                    validated_operations.append({"type": "add", **line.snapshot()})
                    changed = True
                elif isinstance(operation, Edit):
                    candidate, next_line_number, edits = edit_servings(
                        operation, candidate, self._menu, next_line_number,
                    )
                    validated_operations.extend(edits)
                    changed = True
                elif isinstance(operation, ChangeQuantity):
                    line = resolve_target(operation.target, candidate)
                    resized = change_quantity(operation, line)
                    if resized is None:
                        candidate.remove(line)
                    else:
                        candidate[candidate.index(line)] = resized
                    validated_operations.append({
                        **operation.model_dump(), "line_id": line.line_id,
                        "before_quantity": line.quantity, "after_quantity": resized.quantity if resized else 0,
                    })
                    changed = True
                elif isinstance(operation, RemoveLine):
                    line = resolve_target(operation.target, candidate)
                    candidate.remove(line)
                    validated_operations.append({"type": "remove_line", **line.snapshot()})
                    changed = True
                elif isinstance(operation, ClearDraft):
                    validated_operations.append({"type": "clear_draft", "line_ids": [line.line_id for line in candidate]})
                    candidate.clear()
                    instructions = ""
                    changed = True
                elif isinstance(operation, NewOrder):
                    candidate.clear()
                    instructions = ""
                    validated_operations.append(operation.model_dump())
                    changed = True
                elif isinstance(operation, SetInstructions):
                    instructions = operation.instructions.strip()
                    validated_operations.append({"type": "set_instructions", "instructions": instructions})
                    changed = True
                else:
                    if isinstance(operation, MenuQuestion):
                        answers.append(render_menu(self._menu, operation.item_ids))
                    validated_operations.append(operation.model_dump())
            if changed or any(isinstance(operation, Summary) for operation in proposal.operations):
                if self._status == "submitted" and not changed:
                    answers.append(self._receipt_message + "\n" + render_draft(candidate, instructions).removeprefix("Draft order:\n"))
                else:
                    answers.append(render_draft(candidate, instructions))
            if changed:
                try:
                    acknowledgement = self._response_renderer.render_response(ResponseContext(
                        kind="acknowledgement",
                        request=message,
                        operations=tuple(dict(operation) for operation in validated_operations),
                    )).strip()
                except Exception:
                    acknowledgement = ""
                if acknowledgement:
                    answers.insert(0, acknowledgement)
            response: dict[str, Any] = {"message": "\n\n".join(answers)}
            if pending_at_start is not None and changed and not abandoned_pending and not clears_pending:
                lifecycle_operations.append({
                    "type": "clarification_resolved", "reason": pending_at_start.clarification.reason,
                })
            operations = lifecycle_operations + validated_operations
            self._lines = candidate
            self._instructions = instructions
            self._next_line_number = next_line_number
            if changed:
                draft_changed = True
                self._revision += 1
                self._status = "draft"
                self._receipt_message = ""
                self._rejected_payload = None
                self._retry_requires_review = False
                if pending_at_start is not None:
                    self._pending_change = None
            review_requested = any(isinstance(operation, Review) for operation in proposal.operations)
            submit_requested = any(isinstance(operation, Submit) for operation in proposal.operations)
            retry_requested = any(isinstance(operation, RetrySubmission) for operation in proposal.operations)
            if review_requested or submit_requested or retry_requested:
                if self._status == "submitted":
                    response["message"] = self._receipt_message
                elif self._status == "uncertain":
                    response["message"] = "The order may already have been accepted. Please check with the restaurant; this session will not submit again."
                elif self._status == "application_error":
                    response["message"] = "The restaurant reported an application error. This order cannot retry unchanged; edit it and request a new review."
                elif self._status == "rejected" and (review_requested or (retry_requested and self._retry_requires_review)):
                    self._status = "draft"
                    self._reviewed_revision = self._revision
                    response["message"] = render_draft(self._lines, self._instructions) + "\nPlease confirm: submit this exact order?"
                elif self._status == "rejected" and not retry_requested:
                    response["message"] = "The previous submission was rejected. You may explicitly request another attempt or edit the order and review it again."
                else:
                    payload = self._rejected_payload if self._status == "rejected" else submission_payload(self._lines, self._menu, self._instructions)
                    if payload is None:
                        raise InvalidSelection("There is no rejected submission available to retry.")
                    if self._status == "draft" and (retry_requested
                                                    or (review_requested and not submit_requested) or changed
                                                    or self._reviewed_revision != self._revision):
                        self._reviewed_revision = self._revision
                        response["message"] = render_draft(self._lines, self._instructions) + "\nPlease confirm: submit this exact order?"
                    else:
                        frozen = json.dumps(payload)
                        if self._application_error_payload == payload:
                            self._status = "application_error"
                            error_category = "submission_application_error"
                            response["message"] = "The restaurant reported an application error. The outgoing order is unchanged, so it cannot be submitted again. Edit it and request a new review."
                        else:
                            self._application_error_payload = None
                            self._reviewed_revision = None
                            self._status = "uncertain"
                            outcome = self._submitter.submit(json.loads(frozen))
                            if outcome.invoked:
                                tool_calls.append({"name": "submit_order", "arguments": json.loads(frozen), "result": outcome.result})
                            self._status = "draft" if outcome.status == "not_sent" else outcome.status
                            if outcome.status == "submitted":
                                self._rejected_payload = None
                                self._receipt_message = render_receipt(
                                    outcome.result, sum(line.total_cents for line in self._lines),
                                )
                                response["message"] = self._receipt_message
                            elif outcome.status == "not_sent":
                                response["message"] = "The order was not sent. Check submission configuration or connection, then request a new review."
                            elif outcome.status == "rejected":
                                self._application_error_payload = None
                                self._rejected_payload = json.loads(frozen)
                                self._retry_requires_review = False
                                response["message"] = f"The restaurant rejected the order: {outcome.result.get('error', 'No explanation supplied')}. Your selections are preserved. You may edit the order or explicitly request a retry submission."
                            elif outcome.status == "application_error":
                                self._application_error_payload = json.loads(frozen)
                                response["message"] = "The restaurant reported an application error while validating the request. Your selections are preserved, but this order cannot retry unchanged."
                            else:
                                response["message"] = "The order may have been accepted, but its outcome is uncertain. Please check with the restaurant; this session will not submit again."
                            if outcome.status != "submitted":
                                error_category = "submission_" + outcome.status
        except TurnHandled as handled:
            response = handled.response
            operations = handled.operations
        except ClarificationNeeded as error:
            self._reviewed_revision = None
            if self._status == "rejected":
                self._retry_requires_review = True
            source_message = (pending_at_start.original_message
                              if pending_at_start is not None and not abandoned_pending else message)
            stored_proposal = (pending_at_start.proposal if pending_at_start is not None
                               and preserve_pending_proposal
                               else proposal)
            if stored_proposal is None:
                raise RuntimeError("Clarification requested without a proposal") from error
            self._pending_change = PendingChange(source_message, stored_proposal, error.context)
            operations = lifecycle_operations + [{
                "type": "clarification_requested", "reason": error.context.reason,
            }]
            try:
                question = self._clarification_renderer.render(error.context).strip()
            except Exception:
                question = error.context.fallback_question
            if not question:
                question = error.context.fallback_question
            response = {"message": question + " I haven't changed your order yet."}
        except InvalidSelection as error:
            failed_conversation_turn = True
            error_category = "invalid_selection"
            operations = lifecycle_operations
            if pending_at_start is not None and self._pending_change is not None:
                response = {"message": (
                    "I couldn't apply that answer to the change we're working on. "
                    f"{self._pending_change.clarification.fallback_question} "
                    "I haven't changed your order yet."
                )}
            else:
                if draft_changed:
                    response = {"message": render_draft(self._lines, self._instructions)
                                + f"\n{error} The order was not submitted."}
                else:
                    try:
                        explanation = self._response_renderer.render_response(ResponseContext(
                            kind="rejection", request=message, reason=str(error),
                        )).strip()
                    except Exception:
                        explanation = ""
                    response = {"message": (
                        explanation + " I haven't changed your order."
                        if explanation else
                        f"{error} Your draft is unchanged. Please restate the complete request."
                    )}
        except ValidationError:
            failed_conversation_turn = True
            error_category = "invalid_structured_output"
            response = {"message": "I could not understand a valid selection. Your draft is unchanged. Please restate the complete request."}
        except ModelFailure as error:
            failed_conversation_turn = True
            error_category = error.category
            problem = "Model configuration needs attention" if error.category in {"configuration", "authentication"} else "I could not interpret your request right now"
            response = {"message": f"{problem}. Your draft is unchanged. Please try again after the issue is resolved."}
        except Exception:
            failed_conversation_turn = True
            # Contain unexpected integration failures without retaining SDK internals.
            error_category = "internal_error"
            response = {"message": "The submission outcome is uncertain. Please check with the restaurant; this session will not submit again."
                        if self._status == "uncertain" else "I could not process that request. Your draft is unchanged. Please try again."}
        if failed_conversation_turn and self._status == "rejected":
            self._retry_requires_review = True
        if error_category is not None:
            self._reviewed_revision = None
        if tool_calls:
            response["tool_calls"] = tool_calls
        if isinstance(message, str):
            self._history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": response["message"]}])
        self._logger.write({
            "session_id": self._session_id, "turn_id": self._turn_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "input": message if isinstance(message, str) else None,
            "response": response,
            "proposal": (interpreted_proposal.model_dump(mode="json")
                         if interpreted_proposal is not None else None),
            "operations": operations,
            "totals": {"before_cents": before_total, "after_cents": sum(line.total_cents for line in self._lines)},
            "state_transition": {
                "before": {"status": before_status, "revision": before_revision,
                           "reviewed_revision": before_reviewed_revision, "pending_change": before_pending_change},
                "after": {"status": self._status, "revision": self._revision,
                          "reviewed_revision": self._reviewed_revision, "pending_change": self._pending_change is not None},
            },
            "tool_calls": tool_calls, "error_category": error_category,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        })
        return response
