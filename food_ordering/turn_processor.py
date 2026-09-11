"""Bounded, provider-neutral orchestration for one customer turn."""

from collections.abc import Mapping
from typing import Any

from food_ordering.menu import ALIASES, Menu
from food_ordering.model_adapter import (
    AssistantMessage,
    CustomerMessage,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    TranscriptTurn,
    TurnModel,
)
from food_ordering.order import OrderLine, display_groups
from food_ordering.session import Session
from food_ordering.tool_protocol import (
    DisplayGroupSnapshot,
    DraftSnapshot,
    MenuChoiceSnapshot,
    MenuExtraSnapshot,
    MenuItemSnapshot,
    MenuOptionSnapshot,
    MenuSnapshot,
    Malformed,
    ResultPayload,
    SchemaIssue,
    ShowDraft,
    ShowMenu,
    StoredLineSnapshot,
    Unsatisfiable,
    parse_tool_call,
    protocol_schema,
)


MAX_TOOL_CALLS_PER_TURN = 8
MAX_MALFORMED_CORRECTIONS = 2
SAFE_FALLBACK = (
    "I couldn't finish that request safely. Your draft is unchanged. Please try again."
)


def _tool_specs() -> tuple[ToolSpec, ...]:
    schemas = protocol_schema()["tools"]
    descriptions = {
        "show_menu": "Return the complete Menu or the requested menu items.",
        "show_draft": "Return the complete authoritative Draft order.",
    }
    return tuple(
        ToolSpec(name=name, parameters=schemas[name], description=description)
        for name, description in descriptions.items()
    )


def _menu_snapshot(menu: Menu, item_ids: list[str]) -> MenuSnapshot:
    requested = set(item_ids)
    items = [item for item in menu.menu if not requested or item.id in requested]
    return MenuSnapshot(items=[
        MenuItemSnapshot(
            item_id=item.id,
            name=item.name,
            base_price_cents=item.base_price,
            options=[
                MenuOptionSnapshot(
                    name=name,
                    required=option.required,
                    default=option.default,
                    choices=[
                        MenuChoiceSnapshot(
                            value=value,
                            price_delta_cents=option.price_modifier.get(value, 0),
                        )
                        for value in option.choices
                    ],
                )
                for name, option in item.options.items()
            ],
            extras=[
                MenuExtraSnapshot(value=extra.id, price_cents=extra.price)
                for extra in item.extras.choices
            ],
        )
        for item in items
    ])


def _line_snapshot(line: OrderLine) -> StoredLineSnapshot:
    return StoredLineSnapshot(
        line_id=line.line_id,
        item_id=line.item_id,
        name=line.name,
        quantity=line.quantity,
        options=dict(line.options),
        extras=list(line.extras),
        instructions=line.instructions,
        unit_cents=line.unit_cents,
        total_cents=line.total_cents,
    )


def _draft_snapshot(session: Session) -> DraftSnapshot:
    checkout_state = (
        "reviewed"
        if session.status == "draft" and session.reviewed_revision == session.revision
        else session.status
    )
    return DraftSnapshot(
        revision=session.revision,
        lines=[_line_snapshot(line) for line in session.lines],
        display_groups=[
            DisplayGroupSnapshot(
                line_ids=[line.line_id for line in group],
                item_id=group[0].item_id,
                name=group[0].name,
                quantity=sum(line.quantity for line in group),
                options=dict(group[0].options),
                extras=list(group[0].extras),
                instructions=group[0].instructions,
                unit_cents=group[0].unit_cents,
                total_cents=sum(line.total_cents for line in group),
            )
            for group in display_groups(session.lines)
        ],
        general_instructions=session.instructions,
        total_cents=sum(line.total_cents for line in session.lines),
        checkout_state=checkout_state,
    )


def _violation_fingerprint(
    payload: Mapping[str, Any],
) -> tuple[str, str | None, str | None] | None:
    outcome = payload.get("outcome")
    if outcome not in {"INCOMPLETE", "UNSATISFIABLE", "CART_INVALID"}:
        return None
    subject = payload.get("subject")
    key = payload.get("key")
    return (
        str(outcome),
        subject if isinstance(subject, str) else None,
        key if isinstance(key, str) else None,
    )


class TurnProcessor:
    """Run one read-only model/tool/result loop."""

    def __init__(self, *, model: TurnModel, menu: Menu, session: Session) -> None:
        self._model = model
        self._menu = menu
        self._session = session

    def process(self, customer_message: str) -> dict[str, str]:
        current_turn: list[ModelMessage] = [CustomerMessage(customer_message)]
        messages = [
            message
            for turn in self._session.transcript
            for message in turn.messages
        ]
        messages.extend(current_turn)
        malformed_calls = 0
        tool_calls = 0
        violations: set[tuple[str, str | None, str | None]] = set()

        while True:
            try:
                response = self._model.complete(messages=messages, tools=_tool_specs())
            except Exception:
                return self._fallback(current_turn)
            current_turn.append(response)
            messages.append(response)
            if not response.tool_calls:
                if response.content.strip():
                    self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
                    return {"message": response.content}
                return self._fallback(current_turn)

            budget_exhausted = False
            repeated_violation = False
            for call in response.tool_calls:
                if tool_calls >= MAX_TOOL_CALLS_PER_TURN:
                    budget_exhausted = True
                    result = ToolResultMessage(
                        call.call_id,
                        call.name,
                        Malformed(
                            outcome="MALFORMED",
                            remedy="correct_tool",
                            reason="Per-turn tool-call budget exhausted.",
                            resolution="Wait for a new customer turn before using another tool.",
                            tool_name=call.name,
                            issues=[SchemaIssue(path=[], message="Tool-call budget exhausted")],
                        ).model_dump(mode="json", exclude_none=True),
                    )
                else:
                    tool_calls += 1
                    result = self._dispatch(call)
                current_turn.append(result)
                messages.append(result)
                if not budget_exhausted and result.payload["outcome"] == "MALFORMED":
                    malformed_calls += 1
                fingerprint = _violation_fingerprint(result.payload)
                if fingerprint is not None:
                    if fingerprint in violations:
                        repeated_violation = True
                    violations.add(fingerprint)
            if (
                budget_exhausted
                or repeated_violation
                or malformed_calls > MAX_MALFORMED_CORRECTIONS
            ):
                return self._fallback(current_turn)

    def _fallback(self, current_turn: list[ModelMessage]) -> dict[str, str]:
        current_turn.append(AssistantMessage(content=SAFE_FALLBACK))
        self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
        return {"message": SAFE_FALLBACK}

    def _dispatch(self, call: ToolCall) -> ToolResultMessage:
        operation = parse_tool_call(call.name, call.arguments)
        if isinstance(operation, Malformed):
            return ToolResultMessage(
                call.call_id,
                call.name,
                operation.model_dump(mode="json", exclude_none=True),
            )
        if isinstance(operation, ShowMenu):
            available_ids = {item.id for item in self._menu.menu}
            unknown = next(
                (
                    item_id
                    for item_id in operation.item_ids
                    if ALIASES.get(item_id, item_id) not in available_ids
                ),
                None,
            )
            if unknown is not None:
                payload = Unsatisfiable(
                    outcome="UNSATISFIABLE",
                    remedy="change_request",
                    reason="The requested item is not on the Menu.",
                    resolution="Choose an item listed in the Menu.",
                    subject=unknown,
                ).model_dump(mode="json", exclude_none=True)
                return ToolResultMessage(call.call_id, call.name, payload)
            payload = ResultPayload(
                outcome="RESULT",
                result=_menu_snapshot(
                    self._menu,
                    [ALIASES.get(item_id, item_id) for item_id in operation.item_ids],
                ),
            ).model_dump(mode="json", exclude_none=True)
            return ToolResultMessage(call.call_id, call.name, payload)
        if isinstance(operation, ShowDraft):
            payload = ResultPayload(
                outcome="RESULT",
                result=_draft_snapshot(self._session),
            ).model_dump(mode="json", exclude_none=True)
            return ToolResultMessage(call.call_id, call.name, payload)
        payload = Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="That operation is unavailable in the read-only processor.",
            resolution="Use show_menu or show_draft, or wait for mutation support.",
            subject=call.name,
        ).model_dump(mode="json", exclude_none=True)
        return ToolResultMessage(call.call_id, call.name, payload)
