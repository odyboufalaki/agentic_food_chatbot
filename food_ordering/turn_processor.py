"""Bounded, provider-neutral orchestration for one customer turn."""

from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from uuid import uuid4

from food_ordering.draft_operations import (
    validate_add_item,
    validate_change_quantity,
    validate_clear_draft,
    validate_customize_item,
    validate_remove_item,
    validate_replace_item,
    validate_set_order_instructions,
)
from food_ordering.menu import Menu
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
from food_ordering.order import (
    InvalidSelection,
    OrderLine,
    display_groups,
    render_draft,
    render_menu,
    submission_payload,
)
from food_ordering.session import Session
from food_ordering.submission import (
    MCPSubmitter,
    SubmissionResult as TransportSubmissionResult,
    Submitter,
    render_receipt,
)
from food_ordering.tool_protocol import (
    AddItem,
    AlreadyAppliedPayload,
    ApplicationErrorPayload,
    AppliedEffect,
    AppliedPayload,
    CartInvalid,
    ChangeQuantity,
    ClearDraft,
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
    NotSentPayload,
    Operation,
    ProposeSubmission,
    ResultPayload,
    RemoveItem,
    RejectedSubmissionPayload,
    RestaurantPayload,
    ReviewSnapshot,
    ReviewedPayload,
    SchemaIssue,
    SetOrderInstructions,
    ShowDraft,
    ShowMenu,
    StartNewOrder,
    StoredLineSnapshot,
    SubmitOrder,
    SubmittedPayload,
    UncertainSubmissionPayload,
    Unsatisfiable,
    UpdateItem,
    Valid,
    parse_tool_call,
    protocol_schema,
)


MAX_TOOL_CALLS_PER_TURN = 8
MAX_MALFORMED_CORRECTIONS = 2
SAFE_FALLBACK = (
    "I couldn't finish that request safely. Your draft is unchanged. Please try again."
)
REVIEW_CONFIRMATION = "Please confirm: submit this exact order?"
DRAFT_MUTATION_TOOL_NAMES = frozenset({
    "add_item",
    "update_item",
    "change_quantity",
    "remove_item",
    "clear_draft",
    "set_order_instructions",
    "start_new_order",
})


def _tool_specs() -> tuple[ToolSpec, ...]:
    schemas = protocol_schema()["tools"]
    descriptions = {
        "show_menu": "Return the complete Menu or the requested menu items.",
        "show_draft": "Return the complete authoritative Draft order.",
        "add_item": "Add one configured menu item to the Draft order.",
        "update_item": "Customize or replace selected servings in the Draft order.",
        "change_quantity": "Set, increase, or remove a selection's total quantity.",
        "remove_item": "Remove one uniquely identified selection from the Draft order.",
        "clear_draft": "Clear all selections and general instructions from the Draft order.",
        "set_order_instructions": "Set, replace, or clear general Draft-order instructions.",
        "start_new_order": "Start a new Draft after a Submitted order.",
        "propose_submission": "Present the current Draft for customer review.",
        "submit_order": "Submit an unchanged reviewed order after customer confirmation.",
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
    payload: object,
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
    """Run one bounded model/tool/result loop for a customer turn."""

    def __init__(
        self,
        *,
        model: TurnModel,
        menu: Menu,
        session: Session,
        submitter: Submitter | None = None,
        operation_observer: Callable[[ToolCall, Operation], None] | None = None,
        submission_observer: (
            Callable[[dict[str, object], TransportSubmissionResult], None] | None
        ) = None,
    ) -> None:
        self._model = model
        self._menu = menu
        self._session = session
        self._submitter = submitter if submitter is not None else MCPSubmitter()
        self._operation_observer = operation_observer
        self._submission_observer = submission_observer

    def process(self, customer_message: str) -> dict[str, str]:
        self._session.turn_id += 1
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
        customer_input_required = False

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
            if customer_input_required and response.tool_calls:
                current_turn.append(response)
                for call in response.tool_calls:
                    current_turn.append(_aborted_tool_result(
                        call,
                        "customer_input_required",
                    ))
                return self._fallback(current_turn, starting_revision)
            current_turn.append(response)
            messages.append(response)
            if not response.tool_calls:
                if response.content.strip():
                    message = self._authoritative_message(current_turn) or response.content
                    if (
                        self._session.revision != starting_revision
                        and not self._has_draft_result(current_turn)
                    ):
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
            mutation_in_batch = any(
                call.name in DRAFT_MUTATION_TOOL_NAMES
                for call in response.tool_calls
            )
            for call_index, call in enumerate(response.tool_calls):
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
                        if customer_input_required and call.name == "propose_submission":
                            result = _aborted_tool_result(
                                call,
                                "customer_input_required",
                            )
                        elif mutation_in_batch and call.name == "submit_order":
                            result = ToolResultMessage(
                                call.call_id,
                                call.name,
                                AbortedToolResult(
                                    error="turn_aborted",
                                    reason="draft_mutation_in_batch",
                                    resolution=(
                                        "Review the resulting Draft and wait for a new "
                                        "customer turn."
                                    ),
                                ),
                            )
                        else:
                            result = self._dispatch(call)
                        processed_calls[call.call_id] = (call, result)
                    else:
                        result = self._replay_result(call, *previous)
                current_turn.append(result)
                messages.append(result)
                if isinstance(result.payload, SubmittedPayload):
                    for remaining_call in response.tool_calls[call_index + 1:]:
                        current_turn.append(_aborted_tool_result(
                            remaining_call,
                            "submission_completed",
                        ))
                    current_turn.append(AssistantMessage(content=result.payload.receipt))
                    self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
                    return {"message": result.payload.receipt}
                if isinstance(result.payload, Malformed):
                    malformed_calls += 1
                fingerprint = _violation_fingerprint(result.payload)
                if fingerprint is not None:
                    customer_input_required = True
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
        if self._session.status == "draft":
            self._session.invalidate_review()
        if self._session.status == "submitted" and self._session.receipt_message:
            message = self._session.receipt_message
            current_turn.append(AssistantMessage(content=message))
            self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
            return {"message": message}
        if self._session.status == "uncertain" and isinstance(
            self._session.submission_outcome, UncertainSubmissionPayload,
        ):
            uncertain_submission = self._session.submission_outcome
            message = uncertain_submission.reason + " " + uncertain_submission.resolution
            current_turn.append(AssistantMessage(content=message))
            self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
            return {"message": message}
        if self._session.status in {"rejected", "application_error", "not_sent"} and isinstance(
            self._session.submission_outcome,
            (RejectedSubmissionPayload, ApplicationErrorPayload, NotSentPayload),
        ):
            submission_outcome = self._session.submission_outcome
            message = submission_outcome.reason + " " + submission_outcome.resolution
            current_turn.append(AssistantMessage(content=message))
            self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
            return {"message": message}
        message = SAFE_FALLBACK
        if self._session.revision != starting_revision:
            outstanding = "\n".join(self._outstanding_questions(current_turn))
            message = (
                "I couldn't finish that request safely, but some changes were applied.\n"
                + (outstanding + "\n" if outstanding else "")
                + render_draft(self._session.lines, self._session.instructions)
            )
        current_turn.append(AssistantMessage(content=message))
        self._session.transcript.append(TranscriptTurn(tuple(current_turn)))
        return {"message": message}

    def _authoritative_message(self, current_turn: list[ModelMessage]) -> str | None:
        for message in reversed(current_turn):
            if isinstance(message, ToolResultMessage):
                if isinstance(message.payload, ReviewedPayload):
                    return message.payload.review
                if isinstance(
                    message.payload,
                    (
                        RejectedSubmissionPayload,
                        ApplicationErrorPayload,
                        NotSentPayload,
                        UncertainSubmissionPayload,
                    ),
                ):
                    return message.payload.reason + " " + message.payload.resolution
        read_responses: list[str] = []
        for message in current_turn:
            if not isinstance(message, ToolResultMessage) or not isinstance(
                message.payload, ResultPayload,
            ):
                continue
            if isinstance(message.payload.result, MenuSnapshot):
                read_responses.append(render_menu(
                    self._menu,
                    [item.item_id for item in message.payload.result.items],
                ))
            elif self._draft_result(message) is not None:
                read_responses.append(render_draft(
                    self._session.lines,
                    self._session.instructions,
                ))
        if not read_responses:
            return None
        return "\n\n".join([
            *read_responses,
            *self._outstanding_questions(current_turn),
        ])

    @staticmethod
    def _outstanding_questions(current_turn: list[ModelMessage]) -> list[str]:
        questions: list[str] = []
        for message in current_turn:
            if not isinstance(message, ToolResultMessage) or not isinstance(
                message.payload, Incomplete,
            ):
                continue
            payload = message.payload
            choices = (
                " Choices: " + ", ".join(choice.label for choice in payload.alternatives) + "."
                if payload.alternatives else ""
            )
            questions.append(
                "Outstanding question: " + payload.reason + " "
                + payload.resolution + choices
            )
        return questions

    @staticmethod
    def _has_draft_result(current_turn: list[ModelMessage]) -> bool:
        return any(TurnProcessor._draft_result(message) is not None for message in current_turn)

    @staticmethod
    def _draft_result(message: ModelMessage) -> DraftSnapshot | None:
        if (
            isinstance(message, ToolResultMessage)
            and isinstance(message.payload, ResultPayload)
            and isinstance(message.payload.result, DraftSnapshot)
        ):
            return message.payload.result
        return None

    def _replay_result(
        self,
        call: ToolCall,
        previous_call: ToolCall,
        previous_result: ToolResultMessage,
    ) -> ToolResultMessage:
        if call.name != previous_call.name or call.arguments != previous_call.arguments:
            if call.name in DRAFT_MUTATION_TOOL_NAMES:
                self._session.invalidate_review()
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
        if not call.call_id.strip():
            if call.name in DRAFT_MUTATION_TOOL_NAMES:
                self._session.invalidate_review()
            return ToolResultMessage(
                call.call_id,
                call.name,
                Malformed(
                    outcome="MALFORMED",
                    remedy="correct_tool",
                    reason="The tool call is missing its required call ID.",
                    resolution="Return the tool call again with a non-empty call ID.",
                    tool_name=call.name,
                    issues=[SchemaIssue(
                        path=["call_id"],
                        message="Call ID must be a non-empty string",
                    )],
                ),
            )
        operation = parse_tool_call(call.name, call.arguments)
        if isinstance(operation, Malformed):
            if call.name in DRAFT_MUTATION_TOOL_NAMES:
                self._session.invalidate_review()
            return ToolResultMessage(
                call.call_id,
                call.name,
                operation,
            )
        if self._operation_observer is not None:
            try:
                self._operation_observer(call, operation)
            except Exception:
                pass
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
        if self._session.status == "uncertain" and isinstance(
            self._session.submission_outcome, UncertainSubmissionPayload,
        ):
            return ToolResultMessage(
                call.call_id,
                call.name,
                self._session.submission_outcome,
            )
        if isinstance(operation, StartNewOrder):
            if self._session.status != "submitted":
                self._session.invalidate_review()
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    Unsatisfiable(
                        outcome="UNSATISFIABLE",
                        remedy="change_request",
                        reason="A new order can start only after definite acceptance.",
                        resolution="Continue editing the current Draft order.",
                        subject="order_lifecycle",
                    ),
                )
            removed_line_ids = self._session.start_new_order()
            return ToolResultMessage(
                call.call_id,
                call.name,
                AppliedPayload(
                    outcome="APPLIED",
                    effect=AppliedEffect(
                        operation="start_new_order",
                        removed_line_ids=list(removed_line_ids),
                    ),
                    draft=_draft_snapshot(self._session),
                ),
            )
        if isinstance(operation, ProposeSubmission):
            if self._session.status == "submitted" and isinstance(
                self._session.submission_outcome, SubmittedPayload,
            ):
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    self._session.submission_outcome,
                )
            if self._session.status == "application_error" and isinstance(
                self._session.submission_outcome, ApplicationErrorPayload,
            ):
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    self._session.submission_outcome,
                )
            self._session.invalidate_review()
            try:
                raw_payload = submission_payload(
                    self._session.lines,
                    self._menu,
                    self._session.instructions,
                )
            except InvalidSelection as error:
                total_cents = sum(line.total_cents for line in self._session.lines)
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    CartInvalid(
                        outcome="CART_INVALID",
                        remedy="edit_draft",
                        reason=str(error),
                        resolution="Edit the Draft order, then request a new review.",
                        rule="maximum_total" if total_cents > 5000 else "valid_order",
                        current_cents=total_cents,
                        limit_cents=5000 if total_cents > 5000 else None,
                        excess_cents=total_cents - 5000 if total_cents > 5000 else None,
                    ),
                )
            review_text = render_draft(
                self._session.lines,
                self._session.instructions,
            ) + "\n" + REVIEW_CONFIRMATION
            review = ReviewSnapshot(
                review_id=uuid4().hex,
                revision=self._session.revision,
                reviewed_at_turn=self._session.turn_id,
                rendered_order=review_text,
                payload=RestaurantPayload.model_validate(raw_payload),
            )
            self._session.review_snapshot = review
            self._session.reviewed_revision = review.revision
            self._session.status = "draft"
            self._session.submission_outcome = None
            self._session.last_submission_attempt_turn = None
            self._session.rejected_payload = None
            self._session.application_error_payload = None
            return ToolResultMessage(
                call.call_id,
                call.name,
                ReviewedPayload(
                    outcome="REVIEWED",
                    review_id=review.review_id,
                    review=review.rendered_order,
                ),
            )
        if isinstance(operation, SubmitOrder):
            return self._submit_reviewed_order(call)
        if isinstance(
            operation,
            (
                AddItem,
                UpdateItem,
                ChangeQuantity,
                RemoveItem,
                ClearDraft,
                SetOrderInstructions,
            ),
        ):
            if self._session.status == "submitted":
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    Unsatisfiable(
                        outcome="UNSATISFIABLE",
                        remedy="change_request",
                        reason="The current order was already accepted.",
                        resolution="Call start_new_order before adding or editing selections.",
                        subject="order_lifecycle",
                    ),
                )
            draft = DraftState(
                lines=tuple(self._session.lines),
                general_instructions=self._session.instructions,
                next_line_number=self._session.next_line_number,
            )
            if isinstance(operation, AddItem):
                validation = validate_add_item(
                    operation,
                    draft,
                    self._menu,
                )
            elif isinstance(operation, UpdateItem):
                validation = (
                    validate_customize_item(operation, draft, self._menu)
                    if operation.change.type == "customize"
                    else validate_replace_item(operation, draft, self._menu)
                )
            elif isinstance(operation, ChangeQuantity):
                validation = validate_change_quantity(operation, draft)
            elif isinstance(operation, RemoveItem):
                validation = validate_remove_item(operation, draft)
            elif isinstance(operation, ClearDraft):
                validation = validate_clear_draft(operation, draft)
            else:
                validation = validate_set_order_instructions(operation, draft)
            if not isinstance(validation, Valid):
                self._session.invalidate_review()
                return ToolResultMessage(call.call_id, call.name, validation)
            if not validation.changed:
                return ToolResultMessage(
                    call.call_id,
                    call.name,
                    AlreadyAppliedPayload(
                        outcome="ALREADY_APPLIED",
                        operation=call.name,
                        draft=_draft_snapshot(self._session),
                    ),
                )
            self._session.commit_draft(validation.candidate)
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
            reason="That operation is unavailable in this processor version.",
            resolution="Use show_menu, show_draft, or add_item.",
            subject=call.name,
        )
        return ToolResultMessage(call.call_id, call.name, payload)

    def _submit_reviewed_order(self, call: ToolCall) -> ToolResultMessage:
        prior = self._session.submission_outcome
        if self._session.status == "submitted" and isinstance(prior, SubmittedPayload):
            return ToolResultMessage(call.call_id, call.name, prior)
        if self._session.status == "uncertain" and isinstance(
            prior, UncertainSubmissionPayload,
        ):
            return ToolResultMessage(call.call_id, call.name, prior)
        if self._session.status == "application_error" and isinstance(
            prior, ApplicationErrorPayload,
        ):
            return ToolResultMessage(call.call_id, call.name, prior)
        if self._session.status == "not_sent" and isinstance(prior, NotSentPayload):
            return ToolResultMessage(call.call_id, call.name, prior)
        if (
            self._session.status == "rejected"
            and self._session.last_submission_attempt_turn == self._session.turn_id
            and isinstance(prior, RejectedSubmissionPayload)
        ):
            return ToolResultMessage(call.call_id, call.name, prior)
        review = self._session.review_snapshot
        if (
            review is None
            or self._session.reviewed_revision != self._session.revision
            or review.revision != self._session.revision
        ):
            return ToolResultMessage(
                call.call_id,
                call.name,
                Unsatisfiable(
                    outcome="UNSATISFIABLE",
                    remedy="change_request",
                    reason="The current Draft order has no eligible review.",
                    resolution="Call propose_submission and wait for a later customer turn.",
                    subject="confirmation",
                ),
            )
        if review.reviewed_at_turn >= self._session.turn_id:
            return ToolResultMessage(
                call.call_id,
                call.name,
                Unsatisfiable(
                    outcome="UNSATISFIABLE",
                    remedy="change_request",
                    reason="Confirmation must arrive after the Order review.",
                    resolution="Wait for explicit confirmation on a later customer turn.",
                    subject="confirmation",
                ),
            )
        frozen_payload = review.restaurant_payload()
        self._session.last_submission_attempt_turn = self._session.turn_id
        self._session.status = "uncertain"
        uncertain = UncertainSubmissionPayload(
            outcome="UNCERTAIN",
            remedy="block_resubmission",
            reason="The order may have been accepted, but its outcome is uncertain.",
            resolution=(
                "Check with the restaurant; this Session will not mutate or submit again."
            ),
        )
        self._session.submission_outcome = uncertain
        try:
            outcome = self._submitter.submit(frozen_payload)
        except Exception:
            outcome = TransportSubmissionResult(
                status="uncertain",
                invoked=True,
                result={"client_error": "submission_failed", "outcome": "uncertain"},
            )
        if self._submission_observer is not None:
            try:
                self._submission_observer(frozen_payload, outcome)
            except Exception:
                pass
        reviewed_total = sum(line.total_cents for line in self._session.lines)
        if outcome.status == "rejected":
            explanation = _customer_safe_rejection_explanation(outcome.result.get("error"))
            rejected = RejectedSubmissionPayload(
                outcome="REJECTED",
                remedy="ask_customer_before_retry",
                reason="The restaurant rejected the order. " + explanation,
                resolution=(
                    "Your selections are preserved; a later explicit customer request may "
                    "retry this exact reviewed order."
                ),
            )
            self._session.status = "rejected"
            self._session.submission_outcome = rejected
            self._session.rejected_payload = frozen_payload
            return ToolResultMessage(call.call_id, call.name, rejected)
        if outcome.status == "application_error":
            application_error = ApplicationErrorPayload(
                outcome="APPLICATION_ERROR",
                remedy="require_changed_draft",
                reason="The restaurant reported an application error.",
                resolution="Edit the Draft order and request a new review before submitting.",
            )
            self._session.status = "application_error"
            self._session.submission_outcome = application_error
            self._session.application_error_payload = frozen_payload
            self._session.invalidate_review()
            return ToolResultMessage(call.call_id, call.name, application_error)
        if outcome.status == "not_sent":
            not_sent = NotSentPayload(
                outcome="NOT_SENT",
                remedy="require_new_review",
                reason="The order was definitely not sent.",
                resolution="Check the connection or configuration, then request a new review.",
            )
            self._session.status = "not_sent"
            self._session.submission_outcome = not_sent
            self._session.invalidate_review()
            return ToolResultMessage(call.call_id, call.name, not_sent)
        if outcome.status == "uncertain":
            self._session.status = "uncertain"
            return ToolResultMessage(call.call_id, call.name, uncertain)
        receipt = render_receipt(outcome.result, reviewed_total)
        restaurant_total_cents = _restaurant_total_cents(outcome.result.get("total"))
        payload = SubmittedPayload(
            outcome="SUBMITTED",
            receipt=receipt,
            order_id=(str(outcome.result["order_id"])
                      if outcome.result.get("order_id") is not None else None),
            reviewed_total_cents=reviewed_total,
            restaurant_total_cents=restaurant_total_cents,
            estimated_time=(str(outcome.result["estimated_time"])
                            if outcome.result.get("estimated_time") is not None else None),
        )
        self._session.status = "submitted"
        self._session.receipt_message = receipt
        self._session.submission_outcome = payload
        return ToolResultMessage(call.call_id, call.name, payload)


def _restaurant_total_cents(value: object) -> int | None:
    try:
        cents = Decimal(str(value)) * 100
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not cents.is_finite() or cents != cents.to_integral_value():
        return None
    return int(cents)


def _customer_safe_rejection_explanation(value: object) -> str:
    """Map untrusted restaurant diagnostics to a small customer-safe vocabulary."""

    if not isinstance(value, str):
        return "No customer-safe explanation was provided."
    normalized = value.casefold()
    if "busy" in normalized or "capacity" in normalized:
        return "The restaurant is currently busy."
    if "closed" in normalized:
        return "The restaurant is currently closed."
    if "sold out" in normalized or "unavailable" in normalized:
        return "A requested selection is currently unavailable."
    return "No customer-safe explanation was provided."
