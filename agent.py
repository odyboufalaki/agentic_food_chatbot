from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from food_ordering.interpretation import Interpreter, MistralInterpreter, ModelFailure
from food_ordering.menu import load_menu, menu_context
from food_ordering.order import InvalidSelection, OrderLine, change_quantity, edit_line, normalize, render_draft, render_menu, resolve_target
from food_ordering.proposals import Add, ChangeQuantity, ClearDraft, Edit, MenuQuestion, Proposal, RemoveLine, Summary, Unsupported
from food_ordering.turn_logging import TurnLogger


class FoodOrderAgent:
    def __init__(self, *, interpreter: Interpreter | None = None, log_path: Path | None = None) -> None:
        self._interpreter = interpreter if interpreter is not None else MistralInterpreter()
        self._menu = load_menu()
        self._lines: list[OrderLine] = []
        self._history: deque[dict[str, str]] = deque(maxlen=12)
        self._logger = TurnLogger(log_path)
        self._session_id = uuid4().hex
        self._turn_id = 0
        self._revision = 0

    def send(self, message: str) -> dict[str, Any]:
        started = perf_counter()
        self._turn_id += 1
        before_revision = self._revision
        before_total = sum(line.total_cents for line in self._lines)
        operations: list[dict[str, Any]] = []
        error_category = None
        try:
            if not isinstance(message, str) or not message.strip():
                raise InvalidSelection("Please enter a nonempty message.")
            proposal = Proposal.model_validate(self._interpreter.interpret(
                message=message, menu=menu_context(self._menu),
                draft=[line.snapshot() for line in self._lines], history=[dict(entry) for entry in self._history],
            ))
            for operation in proposal.operations:
                if isinstance(operation, Unsupported):
                    explanations = {
                        "not_available": "That request is not available in this version. You can ask about the menu, add or edit selections, remove servings, clear your draft, or review it.",
                        "unclear": "Please specify the menu items and changes you want in a complete request.",
                        "dietary_guarantee": "The menu does not verify ingredients or dietary guarantees. Please check with the restaurant.",
                    }
                    raise InvalidSelection(explanations[operation.reason])
            candidate = list(self._lines)
            validated_operations = []
            answers = []
            changed = False
            for operation in proposal.operations:
                if isinstance(operation, Add):
                    line = normalize(operation, self._menu)
                    candidate.append(line)
                    validated_operations.append({"type": "add", **line.snapshot()})
                    changed = True
                elif isinstance(operation, Edit):
                    line = resolve_target(operation.target, candidate)
                    updated = edit_line(operation, line, self._menu)
                    candidate[candidate.index(line)] = updated
                    validated_operations.append({"type": "edit", **updated.snapshot()})
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
                    changed = True
                else:
                    if isinstance(operation, MenuQuestion):
                        answers.append(render_menu(self._menu, operation.item_ids))
                    validated_operations.append(operation.model_dump())
            if changed or any(isinstance(operation, Summary) for operation in proposal.operations):
                answers.append(render_draft(candidate))
            response = {"message": "\n\n".join(answers)}
            operations = validated_operations
            self._lines = candidate
            if changed:
                self._revision += 1
        except InvalidSelection as error:
            error_category = "invalid_selection"
            response = {"message": f"{error} Your draft is unchanged. Please restate the complete request."}
        except ValidationError:
            error_category = "invalid_structured_output"
            response = {"message": "I could not understand a valid selection. Your draft is unchanged. Please restate the complete request."}
        except ModelFailure as error:
            error_category = error.category
            problem = "Model configuration needs attention" if error.category in {"configuration", "authentication"} else "I could not interpret your request right now"
            response = {"message": f"{problem}. Your draft is unchanged. Please try again after the issue is resolved."}
        except Exception:
            # Contain unexpected integration failures without retaining SDK internals.
            error_category = "internal_error"
            response = {"message": "I could not process that request. Your draft is unchanged. Please try again."}
        if isinstance(message, str):
            self._history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": response["message"]}])
        self._logger.write({
            "session_id": self._session_id, "turn_id": self._turn_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "input": message if isinstance(message, str) else None,
            "response": response, "operations": operations,
            "totals": {"before_cents": before_total, "after_cents": sum(line.total_cents for line in self._lines)},
            "state_transition": {
                "before": {"status": "draft", "revision": before_revision},
                "after": {"status": "draft", "revision": self._revision},
            },
            "tool_calls": [], "error_category": error_category,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        })
        return response
