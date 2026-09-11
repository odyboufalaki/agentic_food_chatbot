"""Bounded, provider-neutral orchestration for one customer turn."""

from food_ordering.menu import Menu
from food_ordering.draft_operations import validate_add_item
from food_ordering.model_adapter import (
    AbortReason,
    AbortedToolResult,
    AssistantMessage,
    CustomerMessage,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    TranscriptTurn,
    TurnModel,
)
from food_ordering.order import OrderLine, display_groups, render_draft
from food_ordering.session import Session
from food_ordering.tool_protocol import (
    AddItem,
    AlreadyAppliedPayload,
    AppliedEffect,
    AppliedPayload,
    CartInvalid,
    DisplayGroupSnapshot,
    DraftSnapshot,
    DraftState,
    Incomplete,
    MenuChoiceSnapshot,
    MenuExtraSnapshot,
    MenuItemSnapshot,
    MenuOptionSnapshot,
    MenuSnapshot,
    Malformed,
    ModelResult,
    ResultPayload,
    SchemaIssue,
    ShowDraft,
    ShowMenu,
    StoredLineSnapshot,
    Unsatisfiable,
    Valid,
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
        "add_item": "Add one configured menu item to the Draft order.",
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
    payload: ModelResult | AbortedToolResult,
) -> tuple[str, str | None, str | None] | None:
    if isinstance(payload, (Incomplete, Unsatisfiable)):
        return payload.outcome, payload.subject, payload.key
    if isinstance(payload, CartInvalid):
        return payload.outcome, None, None
    return None


def _aborted_tool_result(
    call: ToolCall,
    reason: AbortReason,
) -> ToolResultMessage:
    return ToolResultMessage(
        call.call_id,
        call.name,
        AbortedToolResult(
            error="turn_aborted",
            reason=reason,
            resolution="Wait for a new customer turn before using another tool.",
        ),
    )


class TurnProcessor:
    """Run one read-only model/tool/result loop."""

    def __init__(self, *, model: TurnModel, menu: Menu, session: Session) -> None:
        self._model = model
        self._menu = menu
        self._session = session

    def process(self, customer_message: str) -> dict[str, str]:
        starting_revision = self._session.revision
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
        processed_calls: dict[str, tuple[ToolCall, ToolResultMessage]] = {}

        while True:
            try:
                response = self._model.complete(messages=messages, tools=_tool_specs())
            except Exception:
                return self._fallback(current_turn, starting_revision)
            if response.completion_status == "truncated":
                current_turn.append(response)
                for call in response.tool_calls:
                    current_turn.append(_aborted_tool_result(
                        call,
                        "model_response_truncated",
                    ))
                return self._fallback(current_turn, starting_revision)
            current_turn.append(response)
            messages.append(response)
            if not response.tool_calls:
                if response.content.strip():
                    message = response.content
                    if self._session.revision != starting_revision:
                        message += "\n\n" + render_draft(
                            self._session.lines,
                            self._session.instructions,
                        )
                        current_turn[-1] = AssistantMessage(content=message)
                    self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
                    return {"message": message}
                return self._fallback(current_turn, starting_revision)

            budget_exhausted = False
            repeated_violation = False
            for call in response.tool_calls:
                if tool_calls >= MAX_TOOL_CALLS_PER_TURN:
                    budget_exhausted = True
                    result = _aborted_tool_result(
                        call,
                        "tool_call_budget_exhausted",
                    )
                else:
                    tool_calls += 1
                    previous = processed_calls.get(call.call_id)
                    if previous is None:
                        result = self._dispatch(call)
                        processed_calls[call.call_id] = (call, result)
                    else:
                        result = self._replay_result(call, *previous)
                current_turn.append(result)
                messages.append(result)
                if isinstance(result.payload, Malformed):
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
                return self._fallback(current_turn, starting_revision)

    def _fallback(
        self,
        current_turn: list[ModelMessage],
        starting_revision: int,
    ) -> dict[str, str]:
        message = SAFE_FALLBACK
        if self._session.revision != starting_revision:
            message = (
                "I couldn't finish that request safely, but some changes were applied.\n"
                + render_draft(self._session.lines, self._session.instructions)
            )
        current_turn.append(AssistantMessage(content=message))
        self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
        return {"message": message}

    def _replay_result(
        self,
        call: ToolCall,
        previous_call: ToolCall,
        previous_result: ToolResultMessage,
    ) -> ToolResultMessage:
        if call.name != previous_call.name or call.arguments != previous_call.arguments:
            return ToolResultMessage(
                call.call_id,
                call.name,
                Malformed(
                    outcome="MALFORMED",
                    remedy="correct_tool",
                    reason="A tool call ID was reused with different call data.",
                    resolution="Use a new call ID for a different tool call.",
                    tool_name=call.name,
                    issues=[SchemaIssue(
                        path=["call_id"],
                        message="Call ID was already used by a different call",
                    )],
                ),
            )
        if isinstance(previous_result.payload, (AppliedPayload, AlreadyAppliedPayload)):
            return ToolResultMessage(
                call.call_id,
                call.name,
                AlreadyAppliedPayload(
                    outcome="ALREADY_APPLIED",
                    operation=call.name,
                    draft=_draft_snapshot(self._session),
                ),
            )
        return ToolResultMessage(call.call_id, call.name, previous_result.payload)

    def _dispatch(self, call: ToolCall) -> ToolResultMessage:
        operation = parse_tool_call(call.name, call.arguments)
        if isinstance(operation, Malformed):
            return ToolResultMessage(
                call.call_id,
                call.name,
                operation,
            )
        if isinstance(operation, ShowMenu):
            available_ids = {item.id for item in self._menu.menu}
            unknown = next(
                (
                    item_id
                    for item_id in operation.item_ids
                    if item_id not in available_ids
                ),
                None,
            )
            if unknown is not None:
                payload: ModelResult = Unsatisfiable(
                    outcome="UNSATISFIABLE",
                    remedy="change_request",
                    reason="The requested item is not on the Menu.",
                    resolution="Choose an item listed in the Menu.",
                    subject=unknown,
                )
                return ToolResultMessage(call.call_id, call.name, payload)
            payload = ResultPayload(
                outcome="RESULT",
                result=_menu_snapshot(self._menu, operation.item_ids),
            )
            return ToolResultMessage(call.call_id, call.name, payload)
        if isinstance(operation, ShowDraft):
            payload = ResultPayload(
                outcome="RESULT",
                result=_draft_snapshot(self._session),
            )
            return ToolResultMessage(call.call_id, call.name, payload)
        if isinstance(operation, AddItem):
            validation = validate_add_item(
                operation,
                DraftState(
                    lines=tuple(self._session.lines),
                    general_instructions=self._session.instructions,
                    next_line_number=self._session.next_line_number,
                ),
                self._menu,
            )
            if not isinstance(validation, Valid):
                return ToolResultMessage(call.call_id, call.name, validation)
            self._session.lines = list(validation.candidate.lines)
            self._session.instructions = validation.candidate.general_instructions
            self._session.next_line_number = validation.candidate.next_line_number
            self._session.revision += 1
            self._session.reviewed_revision = None
            self._session.status = "draft"
            self._session.receipt_message = ""
            self._session.rejected_payload = None
            self._session.application_error_payload = None
            self._session.retry_requires_review = False
            payload = AppliedPayload(
                outcome="APPLIED",
                effect=AppliedEffect(
                    operation=validation.effect.operation,
                    affected_line_ids=list(validation.effect.affected_line_ids),
                    created_line_ids=list(validation.effect.created_line_ids),
                    removed_line_ids=list(validation.effect.removed_line_ids),
                ),
                draft=_draft_snapshot(self._session),
            )
            return ToolResultMessage(call.call_id, call.name, payload)
        payload = Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="That operation is unavailable in the read-only processor.",
            resolution="Use show_menu or show_draft, or wait for mutation support.",
            subject=call.name,
        )
        return ToolResultMessage(call.call_id, call.name, payload)
