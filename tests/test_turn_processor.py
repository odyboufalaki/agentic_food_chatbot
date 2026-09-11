from collections.abc import Sequence
from typing import Any

import pytest

from food_ordering.menu import load_menu
from food_ordering.model_adapter import (
    AssistantMessage,
    CustomerMessage,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
)
from food_ordering.order import OrderLine, render_draft, render_menu
from food_ordering.session import Session
from food_ordering.submission import SubmissionResult
from food_ordering.turn_processor import TurnProcessor
from model_fakes import RecordingSubmitter


EXPECTED_TOOL_NAMES = [
    "show_menu",
    "show_draft",
    "add_item",
    "update_item",
    "change_quantity",
    "remove_item",
    "clear_draft",
    "set_order_instructions",
    "start_new_order",
    "propose_submission",
    "submit_order",
]


class ScriptedModel:
    def __init__(self, *responses: AssistantMessage) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []
        self.tool_specs: list[tuple[ToolSpec, ...]] = []

    def complete(
        self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
    ) -> AssistantMessage:
        self.requests.append(tuple(messages))
        self.tool_specs.append(tuple(tools))
        return next(self._responses)


def _payload(message: ToolResultMessage) -> Any:
    return message.payload.model_dump(mode="json", exclude_none=True)


def test_review_then_later_confirmation_submits_one_frozen_python_payload() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add-burger",
            name="add_item",
            arguments={
                "item_id": "classic_burger",
                "quantity": 1,
                "instructions": "no onions",
            },
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="general-note",
            name="set_order_instructions",
            arguments={"instructions": "ring the bell"},
        ),)),
        AssistantMessage(content="Noted."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review",
            name="propose_submission",
            arguments={},
        ),)),
        AssistantMessage(content="Untrusted model review text."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirmation",
            name="submit_order",
            arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted",
        invoked=True,
        result={
            "success": True,
            "order_id": "ORD-16",
            "total": 8.5,
            "estimated_time": "10 minutes",
        },
    ))
    session = Session()
    processor = TurnProcessor(
        model=model,
        menu=load_menu(),
        session=session,
        submitter=submitter,
    )

    processor.process("Add a burger without onions")
    processor.process("Please ring the bell")
    review = processor.process("Review my order")

    assert review == {"message": (
        "Draft order:\n"
        "1 × Classic Burger (size: regular, patty: beef); instructions: no onions "
        "— $8.50\n"
        "General instructions: ring the bell\n"
        "Total: $8.50\n"
        "Please confirm: submit this exact order?"
    )}
    assert submitter.calls == []

    receipt = processor.process("Yes, submit it")

    expected_payload = {
        "items": [{
            "item_id": "classic_burger",
            "quantity": 1,
            "options": {"size": "regular", "patty": "beef"},
            "extras": [],
        }],
        "special_instructions": (
            "Item 1: 1 × Classic Burger (size: regular, patty: beef): no onions\n"
            "General: ring the bell"
        ),
    }
    assert submitter.calls == [expected_payload]
    assert receipt == {"message": (
        "Order accepted and submitted!\n"
        "Order number: ORD-16\n"
        "Reviewed total: $8.50\n"
        "Restaurant total: $8.50\n"
        "Estimated time: 10 minutes"
    )}
    assert session.status == "submitted"
    assert session.reviewed_revision == session.revision == 2


@pytest.mark.parametrize("submit_first", [False, True])
def test_mutation_batch_preflight_blocks_submission_in_either_call_order(
    submit_first: bool,
) -> None:
    review_call = ToolCall(call_id="review", name="propose_submission", arguments={})
    edit_call = ToolCall(
        call_id="edit",
        name="change_quantity",
        arguments={
            "target": {"type": "line", "line_id": "L1"},
            "mode": "set",
            "quantity": 2,
        },
    )
    submit_call = ToolCall(call_id="submit", name="submit_order", arguments={})
    calls = (submit_call, edit_call) if submit_first else (edit_call, submit_call)
    model = ScriptedModel(
        AssistantMessage(tool_calls=(review_call,)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=calls),
        AssistantMessage(content="The order was changed and needs another review."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted",
        invoked=True,
        result={"success": True, "order_id": "MUST-NOT-HAPPEN"},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1",
            item_id="classic_burger",
            name="Classic Burger",
            quantity=1,
            options=(("size", "regular"), ("patty", "beef")),
            extras=(),
            instructions="",
            unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model,
        menu=load_menu(),
        session=session,
        submitter=submitter,
    )
    processor.process("Review it")

    processor.process("Yes, but make that two burgers")

    assert submitter.calls == []
    assert session.lines[0].quantity == 2
    assert session.revision == 2
    assert session.reviewed_revision is None
    second_turn_results = [
        message for message in session.transcript[1].messages
        if isinstance(message, ToolResultMessage)
    ]
    submit_result = next(result for result in second_turn_results if result.name == "submit_order")
    assert _payload(submit_result) == {
        "error": "turn_aborted",
        "reason": "draft_mutation_in_batch",
        "resolution": "Review the resulting Draft and wait for a new customer turn.",
    }


@pytest.mark.parametrize(
    "edit_call",
    [
        ToolCall(
            call_id="invalid-option",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "change": {"type": "customize", "options": {"size": "giant"}},
            },
        ),
        ToolCall(
            call_id="malformed-quantity",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 0,
            },
        ),
    ],
)
def test_rejected_mutation_attempt_invalidates_review_eligibility(
    edit_call: ToolCall,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review",
            name="propose_submission",
            arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(edit_call,)),
        AssistantMessage(content="That edit could not be applied."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="stale-confirmation",
            name="submit_order",
            arguments={},
        ),)),
        AssistantMessage(content="A new review is required."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True, result={"success": True},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger",
            quantity=1, options=(("size", "regular"), ("patty", "beef")),
            extras=(), instructions="", unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review it")

    processor.process("Change it using an invalid value")

    assert session.revision == 1
    assert session.reviewed_revision is None
    assert session.review_snapshot is None
    processor.process("Yes")
    assert submitter.calls == []


def test_already_applied_mutation_preserves_an_eligible_review() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="same-quantity",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 1,
            },
        ),)),
        AssistantMessage(content="It is already one burger."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True,
        result={"success": True, "order_id": "ORD-NOOP", "total": 8.5},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger",
            quantity=1, options=(("size", "regular"), ("patty", "beef")),
            extras=(), instructions="", unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review it")

    processor.process("Keep it at one")

    assert session.reviewed_revision == 1
    assert session.review_snapshot is not None
    processor.process("Yes")
    assert len(submitter.calls) == 1


def test_rejection_allows_only_a_later_explicit_retry_of_the_frozen_payload() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="automatic-retry", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted rejection wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="customer-retry", name="submit_order", arguments={},
        ),)),
    )
    rejected = SubmissionResult(
        status="rejected",
        invoked=True,
        result={"success": False, "error": "Kitchen is busy"},
    )
    accepted = SubmissionResult(
        status="submitted",
        invoked=True,
        result={"success": True, "order_id": "ORD-RETRY", "total": 8.5},
    )
    submitter = RecordingSubmitter(rejected, accepted)
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger",
            quantity=1, options=(("size", "regular"), ("patty", "beef")),
            extras=(), instructions="", unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review it")

    response = processor.process("Yes")

    assert response["message"].startswith("The restaurant rejected the order.")
    assert "currently busy" in response["message"]
    assert session.status == "rejected"
    assert len(submitter.calls) == 1
    frozen_payload = submitter.calls[0]

    receipt = processor.process("Please retry the same order")

    assert "ORD-RETRY" in receipt["message"]
    assert submitter.calls == [frozen_payload, frozen_payload]


def test_not_sent_requires_a_new_review_before_another_attempt() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review-1", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-1", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted not-sent wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="premature-retry", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Still untrusted."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review-2", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown again."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-2", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(
        SubmissionResult(
            status="not_sent", invoked=False,
            result={"client_error": "configuration", "outcome": "not_sent"},
        ),
        SubmissionResult(
            status="submitted", invoked=True,
            result={"success": True, "order_id": "ORD-AFTER-REVIEW"},
        ),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")
    failed = processor.process("Yes")

    assert "definitely not sent" in failed["message"]
    assert session.status == "not_sent"
    assert session.review_snapshot is None
    processor.process("Try it now")
    assert len(submitter.calls) == 1

    processor.process("Review again")
    receipt = processor.process("Yes")

    assert "ORD-AFTER-REVIEW" in receipt["message"]
    assert len(submitter.calls) == 2


def test_uncertain_submission_persistently_blocks_mutation_reset_and_resubmission() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted uncertainty wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="mutate",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 2,
            },
        ),)),
        AssistantMessage(content="Untrusted mutation wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="reset", name="start_new_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted reset wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="resubmit", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted retry wording."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="uncertain", invoked=True,
        result={"client_error": "submission_failed", "outcome": "uncertain"},
    ))
    original = OrderLine(
        line_id="L1", item_id="fries", name="French Fries", quantity=1,
        options=(("size", "medium"),), extras=(), instructions="",
        unit_cents=350,
    )
    session = Session(lines=[original], next_line_number=2, revision=1)
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")

    uncertain = processor.process("Yes")
    mutation = processor.process("Make it two")
    reset = processor.process("Start over")
    retry = processor.process("Try submitting again")

    assert "uncertain" in uncertain["message"]
    assert "uncertain" in mutation["message"]
    assert "uncertain" in reset["message"]
    assert "uncertain" in retry["message"]
    assert session.status == "uncertain"
    assert session.lines == [original]
    assert session.revision == 1
    assert len(submitter.calls) == 1


def test_review_and_confirmation_in_one_customer_turn_never_submit() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="review", name="propose_submission", arguments={}),
            ToolCall(call_id="confirm", name="submit_order", arguments={}),
        )),
        AssistantMessage(content="Wait for later confirmation."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True, result={"success": True},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )

    response = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    ).process("Review and submit it")

    assert "Please confirm" in response["message"]
    assert submitter.calls == []
    assert session.review_snapshot is not None
    assert session.review_snapshot.reviewed_at_turn == session.turn_id == 1


def test_stale_review_revision_never_authorizes_submission() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="A new review is required."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True, result={"success": True},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")
    session.revision = 2

    processor.process("Yes")

    assert submitter.calls == []


def test_total_mismatch_is_reported_and_repeated_confirmation_does_not_call_again() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-again", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True,
        result={"success": True, "order_id": "ORD-ONCE", "total": 4.0},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")
    receipt = processor.process("Yes")
    assert "Restaurant total: $4.00 (differs from reviewed total)" in receipt["message"]

    repeated = processor.process("Yes again")

    assert repeated == receipt
    assert len(submitter.calls) == 1


def test_application_error_requires_a_changed_draft_before_a_new_review() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review-1", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-1", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted application-error wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="unchanged-review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Untrusted retry wording."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="edit",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 2,
            },
        ),)),
        AssistantMessage(content="Changed."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review-2", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-2", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(
        SubmissionResult(
            status="application_error", invoked=True,
            result={"code": -32602, "message": "invalid arguments"},
        ),
        SubmissionResult(
            status="submitted", invoked=True,
            result={"success": True, "order_id": "ORD-CORRECTED"},
        ),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")
    processor.process("Yes")

    blocked = processor.process("Review the same order")

    assert "application error" in blocked["message"]
    assert len(submitter.calls) == 1
    processor.process("Make it two")
    processor.process("Review the corrected order")
    receipt = processor.process("Yes")
    assert "ORD-CORRECTED" in receipt["message"]
    assert len(submitter.calls) == 2


def test_start_new_order_after_submission_clears_the_entire_lifecycle() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm", name="submit_order", arguments={},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="new-order", name="start_new_order", arguments={},
        ),)),
        AssistantMessage(content="Started a new order."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True,
        result={"success": True, "order_id": "ORD-DONE"},
    ))
    session = Session(
        lines=[OrderLine(
            line_id="L7", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="crispy",
            unit_cents=350,
        )],
        instructions="no cutlery",
        next_line_number=8,
        revision=4,
    )
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )
    processor.process("Review")
    processor.process("Yes")

    processor.process("Start a new order")

    assert session.status == "draft"
    assert session.lines == []
    assert session.instructions == ""
    assert session.next_line_number == 8
    assert session.revision == 5
    assert session.reviewed_revision is None
    assert session.review_snapshot is None
    assert session.submission_outcome is None


@pytest.mark.parametrize(
    ("session", "expected_rule"),
    [
        (Session(), "valid_order"),
        (
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="classic_burger", name="Classic Burger",
                    quantity=6,
                    options=(("size", "regular"), ("patty", "beef")),
                    extras=(), instructions="", unit_cents=850,
                )],
                next_line_number=2,
                revision=1,
            ),
            "maximum_total",
        ),
    ],
)
def test_empty_or_over_limit_draft_cannot_create_a_review_or_submit(
    session: Session,
    expected_rule: str,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="The Draft cannot be reviewed yet."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="confirm-without-review", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="There is no eligible review."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True, result={"success": True},
    ))
    processor = TurnProcessor(
        model=model, menu=load_menu(), session=session, submitter=submitter,
    )

    processor.process("Review")

    result = next(
        message for message in session.transcript[0].messages
        if isinstance(message, ToolResultMessage)
    )
    assert _payload(result)["outcome"] == "CART_INVALID"
    assert _payload(result)["rule"] == expected_rule
    assert session.review_snapshot is None
    processor.process("Yes")
    assert submitter.calls == []


def test_failed_model_turn_after_review_invalidates_confirmation() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(model=model, menu=load_menu(), session=session)
    processor.process("Review")
    assert session.review_snapshot is not None

    processor.process("A request the model fails to interpret")

    assert session.reviewed_revision is None
    assert session.review_snapshot is None


def test_invalid_new_order_attempt_after_review_invalidates_confirmation() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-reset", name="start_new_order", arguments={},
        ),)),
        AssistantMessage(content="The current Draft remains active."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )
    processor = TurnProcessor(model=model, menu=load_menu(), session=session)
    processor.process("Review")

    processor.process("Start a new order")

    assert session.status == "draft"
    assert session.reviewed_revision is None
    assert session.review_snapshot is None


def test_reused_call_id_on_a_malformed_mutation_invalidates_the_new_review() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="reused", name="propose_submission", arguments={}),
            ToolCall(
                call_id="reused",
                name="change_quantity",
                arguments={
                    "target": {"type": "line", "line_id": "L1"},
                    "mode": "set",
                    "quantity": 2,
                },
            ),
        )),
        AssistantMessage(content="The malformed edit needs a new review."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), instructions="",
            unit_cents=350,
        )],
        next_line_number=2,
        revision=1,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Review it and make a malformed edit",
    )

    assert session.revision == 1
    assert session.reviewed_revision is None
    assert session.review_snapshot is None


@pytest.mark.parametrize(
    ("operation", "session"),
    [
        (
            ToolCall(
                call_id="add",
                name="add_item",
                arguments={"item_id": "fries", "quantity": 1},
            ),
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="classic_burger", name="Classic Burger",
                    quantity=1,
                    options=(("size", "regular"), ("patty", "beef")),
                    extras=(), instructions="", unit_cents=850,
                )],
                next_line_number=2,
                revision=1,
            ),
        ),
        (
            ToolCall(
                call_id="update",
                name="update_item",
                arguments={
                    "target": {"type": "line", "line_id": "L1"},
                    "change": {"type": "customize", "add_extras": ["cheese"]},
                },
            ),
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="classic_burger", name="Classic Burger",
                    quantity=1,
                    options=(("size", "regular"), ("patty", "beef")),
                    extras=(), instructions="", unit_cents=850,
                )],
                next_line_number=2,
                revision=1,
            ),
        ),
        (
            ToolCall(
                call_id="remove",
                name="remove_item",
                arguments={"target": {"type": "line", "line_id": "L1"}},
            ),
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="fries", name="French Fries", quantity=1,
                    options=(("size", "medium"),), extras=(), instructions="",
                    unit_cents=350,
                )],
                next_line_number=2,
                revision=1,
            ),
        ),
        (
            ToolCall(call_id="clear", name="clear_draft", arguments={}),
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="fries", name="French Fries", quantity=1,
                    options=(("size", "medium"),), extras=(), instructions="",
                    unit_cents=350,
                )],
                next_line_number=2,
                revision=1,
            ),
        ),
        (
            ToolCall(
                call_id="instructions",
                name="set_order_instructions",
                arguments={"instructions": "no cutlery"},
            ),
            Session(
                lines=[OrderLine(
                    line_id="L1", item_id="fries", name="French Fries", quantity=1,
                    options=(("size", "medium"),), extras=(), instructions="",
                    unit_cents=350,
                )],
                next_line_number=2,
                revision=1,
            ),
        ),
    ],
)
def test_every_changed_draft_operation_invalidates_the_review(
    operation: ToolCall,
    session: Session,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review shown."),
        AssistantMessage(tool_calls=(operation,)),
        AssistantMessage(content="Changed."),
    )
    processor = TurnProcessor(model=model, menu=load_menu(), session=session)
    processor.process("Review")
    assert session.review_snapshot is not None

    processor.process("Change the Draft")

    assert session.revision == 2
    assert session.reviewed_revision is None
    assert session.review_snapshot is None


def test_model_can_read_menu_result_then_complete_a_natural_response() -> None:
    model = ScriptedModel(
        AssistantMessage(
            content="I will check the menu.",
            tool_calls=(ToolCall(call_id="call-1", name="show_menu", arguments={}),),
        ),
        AssistantMessage(content="We have burgers, pizza, sides, drinks, and desserts."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is on the menu?",
    )

    assert response == {"message": render_menu(load_menu(), [])}
    assert [tool.name for tool in model.tool_specs[0]] == EXPECTED_TOOL_NAMES
    update_spec = next(tool for tool in model.tool_specs[0] if tool.name == "update_item")
    assert "ReplaceServings" in update_spec.parameters["$defs"]
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert result.call_id == "call-1"
    assert _payload(result)["outcome"] == "RESULT"
    assert _payload(result)["result"]["items"][0] == {
        "item_id": "classic_burger",
        "name": "Classic Burger",
        "base_price_cents": 850,
        "options": [
            {
                "name": "size",
                "required": True,
                "default": "regular",
                "choices": [
                    {"value": "regular", "price_delta_cents": 0},
                    {"value": "large", "price_delta_cents": 200},
                ],
            },
            {
                "name": "patty",
                "required": False,
                "default": "beef",
                "choices": [
                    {"value": "beef", "price_delta_cents": 0},
                    {"value": "chicken", "price_delta_cents": 0},
                    {"value": "veggie", "price_delta_cents": 0},
                ],
            },
        ],
        "extras": [
            {"value": "cheese", "price_cents": 100},
            {"value": "bacon", "price_cents": 150},
            {"value": "avocado", "price_cents": 200},
            {"value": "extra_patty", "price_cents": 300},
        ],
    }


@pytest.mark.parametrize("item_id", ["lobster", "burger"])
def test_unknown_requested_menu_item_returns_a_deterministic_rejection(
    item_id: str,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="missing-menu-item",
            name="show_menu",
            arguments={"item_ids": [item_id]},
        ),)),
        AssistantMessage(content="Lobster is not on the menu."),
    )

    response = TurnProcessor(model=model, menu=load_menu(), session=Session()).process(
        "Do you have lobster?",
    )

    assert response == {"message": "Lobster is not on the menu."}
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result) == {
        "outcome": "UNSATISFIABLE",
        "remedy": "change_request",
        "reason": "The requested item is not on the Menu.",
        "resolution": "Choose an item listed in the Menu.",
        "subject": item_id,
    }


def test_model_can_read_the_authoritative_draft_from_an_existing_session() -> None:
    session = Session(
        lines=[
            OrderLine(
                item_id="fries",
                name="French Fries",
                quantity=2,
                options=(("size", "large"),),
                extras=("parmesan",),
                unit_cents=600,
                line_id="L4",
                instructions="extra crispy",
            ),
        ],
        instructions="no cutlery",
        next_line_number=5,
        revision=3,
    )
    model = ScriptedModel(
        AssistantMessage(
            tool_calls=(ToolCall(call_id="draft-1", name="show_draft", arguments={}),),
        ),
        AssistantMessage(content="You have two large fries with parmesan."),
    )

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is in my order?",
    )

    assert response == {"message": render_draft(session.lines, session.instructions)}
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result) == {
        "outcome": "RESULT",
        "result": {
            "revision": 3,
            "lines": [{
                "line_id": "L4",
                "item_id": "fries",
                "name": "French Fries",
                "quantity": 2,
                "options": {"size": "large"},
                "extras": ["parmesan"],
                "instructions": "extra crispy",
                "unit_cents": 600,
                "total_cents": 1200,
            }],
            "display_groups": [{
                "line_ids": ["L4"],
                "item_id": "fries",
                "name": "French Fries",
                "quantity": 2,
                "options": {"size": "large"},
                "extras": ["parmesan"],
                "instructions": "extra crispy",
                "unit_cents": 600,
                "total_cents": 1200,
            }],
            "general_instructions": "no cutlery",
            "total_cents": 1200,
            "checkout_state": "draft",
        },
    }


def test_malformed_call_is_returned_for_one_blind_correction() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="bad-1", name="show_draft", arguments={"unexpected": True}),
        )),
        AssistantMessage(tool_calls=(
            ToolCall(call_id="fixed-1", name="show_draft", arguments={}),
        )),
        AssistantMessage(content="Your draft is empty."),
    )

    response = TurnProcessor(model=model, menu=load_menu(), session=Session()).process(
        "What is in my order?",
    )

    assert response == {"message": render_draft([], "")}
    malformed = model.requests[1][-1]
    assert isinstance(malformed, ToolResultMessage)
    assert malformed.call_id == "bad-1"
    assert _payload(malformed)["outcome"] == "MALFORMED"
    assert _payload(malformed)["remedy"] == "correct_tool"
    corrected = model.requests[2][-1]
    assert isinstance(corrected, ToolResultMessage)
    assert corrected.call_id == "fixed-1"
    assert _payload(corrected)["outcome"] == "RESULT"


def test_tool_call_without_an_id_is_rejected_before_draft_mutation() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="",
            name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Please let me try that again."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process("Add fries")

    assert session.lines == []
    assert session.revision == 0
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result) == {
        "outcome": "MALFORMED",
        "remedy": "correct_tool",
        "reason": "The tool call is missing its required call ID.",
        "resolution": "Return the tool call again with a non-empty call ID.",
        "tool_name": "add_item",
        "issues": [{"path": ["call_id"], "message": "Call ID must be a non-empty string"}],
    }


def test_validated_operation_is_observable_before_draft_mutation() -> None:
    session = Session()
    observed: list[tuple[ToolCall, object, int, list[OrderLine]]] = []

    def observe(call: ToolCall, operation: object) -> None:
        observed.append((call, operation, session.revision, list(session.lines)))

    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add-fries",
            name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="I added the fries."),
    )

    TurnProcessor(
        model=model,
        menu=load_menu(),
        session=session,
        operation_observer=observe,
    ).process("Add fries")

    assert len(observed) == 1
    call, operation, revision_at_observation, lines_at_observation = observed[0]
    assert call.arguments == {"item_id": "fries", "quantity": 1}
    assert getattr(operation, "item_id") == "fries"
    assert revision_at_observation == 0
    assert lines_at_observation == []
    assert session.revision == 1


def test_exhausted_malformed_correction_budget_uses_customer_safe_fallback() -> None:
    model = ScriptedModel(
        *(AssistantMessage(tool_calls=(
            ToolCall(
                call_id=f"bad-{number}",
                name="show_draft",
                arguments={"private_bad_argument": number},
            ),
        )) for number in range(1, 4)),
        AssistantMessage(content="This response must not become customer-facing."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is in my order?",
    )

    assert response == {
        "message": "I couldn't finish that request safely. Your draft is unchanged. Please try again.",
    }
    assert len(model.requests) == 3
    assert "private_bad_argument" not in response["message"]
    messages = session.transcript[0].messages
    assert isinstance(messages[-2], ToolResultMessage)
    assert messages[-2].call_id == "bad-3"
    assert _payload(messages[-2])["outcome"] == "MALFORMED"
    assert messages[-1] == AssistantMessage(content=response["message"])


def test_tool_call_budget_pairs_the_over_budget_call_with_an_aborted_result() -> None:
    model = ScriptedModel(
        *(AssistantMessage(tool_calls=(
            ToolCall(call_id=f"draft-{number}", name="show_draft", arguments={}),
        )) for number in range(1, 10)),
        AssistantMessage(content="This response must not be requested."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Keep checking my draft",
    )

    assert response["message"] == (
        "I couldn't finish that request safely. Your draft is unchanged. Please try again."
    )
    assert len(model.requests) == 9
    results = [
        message for message in session.transcript[0].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 9
    assert [_payload(result)["outcome"] for result in results[:8]] == ["RESULT"] * 8
    assert results[8].call_id == "draft-9"
    assert _payload(results[8]) == {
        "error": "turn_aborted",
        "reason": "tool_call_budget_exhausted",
        "resolution": "Wait for a new customer turn before using another tool.",
    }


def test_partial_replacement_uses_clean_defaults_and_preserves_unaffected_servings() -> None:
    original = OrderLine(
        item_id="classic_burger",
        name="Classic Burger",
        quantity=2,
        options=(("size", "large"), ("patty", "chicken")),
        extras=("cheese",),
        unit_cents=1150,
        line_id="L1",
        instructions="no onions",
    )
    session = Session(
        lines=[original],
        instructions="ring the bell",
        next_line_number=2,
        revision=4,
    )
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="replace-one",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "servings": 1,
                "change": {"type": "replace", "item_id": "fries"},
            },
        ),)),
        AssistantMessage(content="I replaced one burger with fries."),
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Replace one burger with fries",
    )

    assert session.lines == [
        OrderLine(
            item_id="classic_burger", name="Classic Burger", quantity=1,
            options=(("size", "large"), ("patty", "chicken")),
            extras=("cheese",), unit_cents=1150, line_id="L1",
            instructions="no onions",
        ),
        OrderLine(
            item_id="fries", name="French Fries", quantity=1,
            options=(("size", "medium"),), extras=(), unit_cents=350,
            line_id="L2", instructions="",
        ),
    ]
    assert session.instructions == "ring the bell"
    assert session.next_line_number == 3
    assert session.revision == 5
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "APPLIED"
    assert payload["effect"] == {
        "operation": "update_item",
        "affected_line_ids": ["L1"],
        "created_line_ids": ["L2"],
        "removed_line_ids": [],
    }
    assert payload["draft"]["total_cents"] == 1500


def test_replacement_missing_a_required_destination_option_changes_nothing() -> None:
    originals = [
        OrderLine(
            item_id="classic_burger", name="Classic Burger", quantity=1,
            options=(("size", "regular"), ("patty", "beef")), extras=(),
            unit_cents=850, line_id="L1", instructions="",
        ),
        OrderLine(
            item_id="classic_burger", name="Classic Burger", quantity=2,
            options=(("size", "large"), ("patty", "veggie")), extras=("cheese",),
            unit_cents=1150, line_id="L2", instructions="well done",
        ),
    ]
    session = Session(lines=list(originals), next_line_number=3, revision=2)
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="missing-flavor",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": "all",
                "change": {"type": "replace", "item_id": "milkshake"},
            },
        ),)),
        AssistantMessage(content="Which milkshake flavor would you like?"),
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Replace all burgers with milkshakes",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "INCOMPLETE"
    assert payload["key"] == "flavor"
    assert [choice["value"] for choice in payload["alternatives"]] == [
        "vanilla", "chocolate", "strawberry", "oreo",
    ]
    assert session.lines == originals
    assert session.revision == 2
    assert session.next_line_number == 3


def test_explicit_replacement_values_span_earliest_lines_and_group_new_servings() -> None:
    session = Session(
        lines=[
            OrderLine(
                item_id="classic_burger", name="Classic Burger", quantity=1,
                options=(("size", "regular"), ("patty", "beef")), extras=(),
                unit_cents=850, line_id="L1", instructions="no salt",
            ),
            OrderLine(
                item_id="classic_burger", name="Classic Burger", quantity=2,
                options=(("size", "regular"), ("patty", "beef")), extras=("cheese",),
                unit_cents=950, line_id="L2", instructions="well done",
            ),
        ],
        next_line_number=3,
        revision=2,
    )
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="replace-earliest-two",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": 2,
                "change": {
                    "type": "replace",
                    "item_id": "fries",
                    "options": {"size": "large"},
                    "extras": ["parmesan"],
                    "instructions": "extra crispy",
                },
            },
        ),)),
        AssistantMessage(content="I replaced the first two burgers."),
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Replace two burgers with large parmesan fries, extra crispy",
    )

    assert [
        (
            line.line_id, line.item_id, line.quantity, dict(line.options),
            line.extras, line.instructions, line.unit_cents,
        )
        for line in session.lines
    ] == [
        ("L1", "fries", 1, {"size": "large"}, ("parmesan",), "extra crispy", 600),
        (
            "L2", "classic_burger", 1,
            {"size": "regular", "patty": "beef"},
            ("cheese",), "well done", 950,
        ),
        ("L3", "fries", 1, {"size": "large"}, ("parmesan",), "extra crispy", 600),
    ]
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["effect"]["affected_line_ids"] == ["L1", "L2"]
    assert payload["effect"]["created_line_ids"] == ["L3"]
    assert payload["draft"]["total_cents"] == 2150
    assert [
        (group["line_ids"], group["quantity"], group["total_cents"])
        for group in payload["draft"]["display_groups"]
    ] == [(["L1", "L3"], 2, 1200), (["L2"], 1, 950)]


@pytest.mark.parametrize(
    ("change", "expected_key"),
    [
        ({"options": {"temperature": "hot"}}, "temperature"),
        ({"options": {"size": "family"}}, "size"),
        ({"extras": ["cheese"]}, "extras"),
    ],
)
def test_invalid_destination_values_leave_replacement_unapplied(
    change: dict[str, Any],
    expected_key: str,
) -> None:
    original = OrderLine(
        item_id="classic_burger", name="Classic Burger", quantity=2,
        options=(("size", "large"), ("patty", "chicken")), extras=("cheese",),
        unit_cents=1150, line_id="L1", instructions="no onions",
    )
    session = Session(lines=[original], next_line_number=2, revision=1)
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-destination",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "servings": 1,
                "change": {"type": "replace", "item_id": "fries", **change},
            },
        ),)),
        AssistantMessage(content="That replacement is unavailable."),
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Replace one burger using an unavailable fries configuration",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "UNSATISFIABLE"
    assert payload["key"] == expected_key
    assert session.lines == [original]
    assert session.revision == 1
    assert session.next_line_number == 2


def test_complete_tool_pairs_are_retained_by_turn_for_later_reconstruction() -> None:
    model = ScriptedModel(
        AssistantMessage(
            content="Preliminary text is transcript-only.",
            tool_calls=(
                ToolCall(call_id="menu", name="show_menu", arguments={"item_ids": ["fries"]}),
                ToolCall(call_id="draft", name="show_draft", arguments={}),
            ),
        ),
        AssistantMessage(content="Fries are available and your draft is empty."),
        AssistantMessage(content="What size fries would you like?"),
    )
    session = Session()

    first = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Do you have fries, and what is in my draft?",
    )
    second = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "I want fries.",
    )

    assert first == {"message": (
        render_menu(load_menu(), ["fries"]) + "\n\n" + render_draft([], "")
    )}
    assert second == {"message": "What size fries would you like?"}
    first_turn = session.transcript[0].messages
    assert first_turn[0] == CustomerMessage("Do you have fries, and what is in my draft?")
    assert isinstance(first_turn[1], AssistantMessage)
    assert first_turn[1].content == "Preliminary text is transcript-only."
    assert [message.call_id for message in first_turn[2:4]
            if isinstance(message, ToolResultMessage)] == ["menu", "draft"]
    assert first_turn[-1] == AssistantMessage(content=first["message"])
    assert model.requests[2] == first_turn + (CustomerMessage("I want fries."),)


def test_empty_model_completion_uses_deterministic_fallback() -> None:
    session = Session()
    model = ScriptedModel(AssistantMessage(content="   "))

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is available?",
    )

    assert response == {
        "message": "I couldn't finish that request safely. Your draft is unchanged. Please try again.",
    }
    assert session.transcript[0].messages[-1] == AssistantMessage(content=response["message"])


def test_truncated_nonempty_model_response_is_not_customer_facing() -> None:
    session = Session()
    model = ScriptedModel(AssistantMessage(
        content="A partial internal response with validation details",
        completion_status="truncated",
    ))

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is available?",
    )

    assert response["message"] == (
        "I couldn't finish that request safely. Your draft is unchanged. Please try again."
    )
    assert "validation" not in response["message"]


def test_truncated_tool_calls_are_not_dispatched_or_added_without_results() -> None:
    session = Session()
    model = ScriptedModel(AssistantMessage(
        content="I only decoded part of this call.",
        tool_calls=(ToolCall(call_id="partial", name="show_draft", arguments={}),),
        completion_status="truncated",
    ))

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "What is in my draft?",
    )

    assert response["message"] == (
        "I couldn't finish that request safely. Your draft is unchanged. Please try again."
    )
    messages = session.transcript[0].messages
    assert messages[0] == CustomerMessage("What is in my draft?")
    assert messages[1] == AssistantMessage(
        content="I only decoded part of this call.",
        tool_calls=(ToolCall(call_id="partial", name="show_draft", arguments={}),),
        completion_status="truncated",
    )
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].call_id == "partial"
    assert _payload(messages[2])["reason"] == "model_response_truncated"
    assert messages[3] == AssistantMessage(content=response["message"])


def test_model_failure_after_a_read_result_preserves_pair_and_uses_fallback() -> None:
    class FailingAfterResultModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
        ) -> AssistantMessage:
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(tool_calls=(
                    ToolCall(call_id="draft-before-failure", name="show_draft", arguments={}),
                ))
            raise RuntimeError("private provider failure")

    model = FailingAfterResultModel()
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Check my draft",
    )

    assert response["message"] == (
        "I couldn't finish that request safely. Your draft is unchanged. Please try again."
    )
    assert "provider" not in response["message"]
    tool_results = [
        message for message in session.transcript[0].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [result.call_id for result in tool_results] == ["draft-before-failure"]


def test_repeated_non_malformed_violation_stops_before_another_model_request() -> None:
    missing_item = {"item_ids": ["lobster"]}
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="missing-1", name="show_menu", arguments=missing_item),
        )),
        AssistantMessage(tool_calls=(
            ToolCall(call_id="missing-2", name="show_menu", arguments=missing_item),
        )),
        AssistantMessage(content="This response must not be requested."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a burger",
    )

    assert response["message"] == (
        "I couldn't finish that request safely. Your draft is unchanged. Please try again."
    )
    assert len(model.requests) == 2
    results = [
        message for message in session.transcript[0].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [result.call_id for result in results] == ["missing-1", "missing-2"]


def test_valid_burgers_commit_while_incomplete_milkshake_asks_for_flavor() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="burgers",
                name="add_item",
                arguments={"item_id": "classic_burger", "quantity": 2},
            ),
            ToolCall(
                call_id="milkshake",
                name="add_item",
                arguments={"item_id": "milkshake", "quantity": 1},
            ),
        )),
        AssistantMessage(content="I added two burgers. Which milkshake flavor would you like?"),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Two burgers and a milkshake",
    )

    assert response == {"message": (
        "I added two burgers. Which milkshake flavor would you like?\n\n"
        "Draft order:\n"
        "2 × Classic Burger (size: regular, patty: beef) — $17.00\n"
        "Total: $17.00"
    )}
    assert [tool.name for tool in model.tool_specs[0]] == EXPECTED_TOOL_NAMES
    results = model.requests[1][-2:]
    assert all(isinstance(result, ToolResultMessage) for result in results)
    assert _payload(results[0]) == {
        "outcome": "APPLIED",
        "effect": {
            "operation": "add_item",
            "affected_line_ids": [],
            "created_line_ids": ["L1"],
            "removed_line_ids": [],
        },
        "draft": {
            "revision": 1,
            "lines": [{
                "line_id": "L1",
                "item_id": "classic_burger",
                "name": "Classic Burger",
                "quantity": 2,
                "options": {"size": "regular", "patty": "beef"},
                "extras": [],
                "instructions": "",
                "unit_cents": 850,
                "total_cents": 1700,
            }],
            "display_groups": [{
                "line_ids": ["L1"],
                "item_id": "classic_burger",
                "name": "Classic Burger",
                "quantity": 2,
                "options": {"size": "regular", "patty": "beef"},
                "extras": [],
                "instructions": "",
                "unit_cents": 850,
                "total_cents": 1700,
            }],
            "general_instructions": "",
            "total_cents": 1700,
            "checkout_state": "draft",
        },
    }
    assert _payload(results[1]) == {
        "outcome": "INCOMPLETE",
        "remedy": "ask_customer",
        "reason": "Milkshake requires a flavor.",
        "resolution": "Ask the customer to choose a flavor.",
        "subject": "Milkshake",
        "key": "flavor",
        "alternatives": [
            {"value": "vanilla", "label": "vanilla"},
            {"value": "chocolate", "label": "chocolate"},
            {"value": "strawberry", "label": "strawberry"},
            {"value": "oreo", "label": "oreo"},
        ],
    }
    assert [(line.line_id, line.item_id, line.quantity, line.total_cents)
            for line in session.lines] == [("L1", "classic_burger", 2, 1700)]
    assert session.next_line_number == 2
    assert session.revision == 1


def test_flavor_answer_reconstructs_a_fresh_add_without_duplicating_prior_success() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="burger-first-turn",
                name="add_item",
                arguments={"item_id": "classic_burger", "quantity": 1},
            ),
            ToolCall(
                call_id="incomplete-shake",
                name="add_item",
                arguments={"item_id": "milkshake", "quantity": 1},
            ),
        )),
        AssistantMessage(content="Which milkshake flavor would you like?"),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="fresh-shake",
            name="add_item",
            arguments={
                "item_id": "milkshake",
                "quantity": 1,
                "options": {"flavor": "chocolate"},
            },
        ),)),
        AssistantMessage(content="I added one chocolate milkshake."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "A burger and a milkshake",
    )
    first_turn = session.transcript[0].messages
    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Chocolate",
    )

    assert response == {"message": (
        "I added one chocolate milkshake.\n\n"
        "Draft order:\n"
        "1 × Classic Burger (size: regular, patty: beef) — $8.50\n"
        "1 × Milkshake (size: regular, flavor: chocolate) — $5.50\n"
        "Total: $14.00"
    )}
    assert model.requests[2] == first_turn + (CustomerMessage("Chocolate"),)
    assert [(line.line_id, line.item_id, line.quantity, dict(line.options), line.total_cents)
            for line in session.lines] == [
        ("L1", "classic_burger", 1, {"size": "regular", "patty": "beef"}, 850),
        ("L2", "milkshake", 1, {"size": "regular", "flavor": "chocolate"}, 550),
    ]
    assert session.revision == 2
    assert session.next_line_number == 3


@pytest.mark.parametrize(
    ("arguments", "expected_outcome", "expected_key", "expected_alternatives"),
    [
        (
            {"item_id": "lobster", "quantity": 1},
            "UNSATISFIABLE",
            "item_id",
            ["classic_burger", "spicy_burger", "margherita"],
        ),
        (
            {
                "item_id": "classic_burger",
                "quantity": 1,
                "options": {"temperature": "hot"},
            },
            "UNSATISFIABLE",
            "temperature",
            ["size", "patty"],
        ),
        (
            {
                "item_id": "milkshake",
                "quantity": 1,
                "options": {"flavor": "mint"},
            },
            "INCOMPLETE",
            "flavor",
            ["vanilla", "chocolate", "strawberry", "oreo"],
        ),
        (
            {
                "item_id": "classic_burger",
                "quantity": 1,
                "extras": ["pickles"],
            },
            "UNSATISFIABLE",
            "extras",
            ["cheese", "bacon", "avocado", "extra_patty"],
        ),
    ],
)
def test_invalid_additions_return_structured_remedies_without_mutating_draft(
    arguments: dict[str, Any],
    expected_outcome: str,
    expected_key: str,
    expected_alternatives: list[str],
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-add",
            name="add_item",
            arguments=arguments,
        ),)),
        AssistantMessage(content="Please choose from the available Menu choices."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add this item",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == expected_outcome
    assert payload["key"] == expected_key
    assert [alternative["value"] for alternative in payload["alternatives"]][
        :len(expected_alternatives)
    ] == expected_alternatives
    assert payload["reason"]
    assert payload["resolution"]
    assert session.lines == []
    assert session.revision == 0
    assert session.next_line_number == 1


def test_addition_applies_defaults_deduplicates_extras_and_prices_in_python() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="configured-burgers",
            name="add_item",
            arguments={
                "item_id": "burger",
                "quantity": 2,
                "options": {"size": "large", "patty": "veggie"},
                "extras": ["cheese", "bacon", "cheese"],
                "instructions": "  well done  ",
            },
        ),)),
        AssistantMessage(content="I added the configured burgers."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Two large veggie burgers with cheese, bacon, and cheese, well done",
    )

    assert len(session.lines) == 1
    line = session.lines[0]
    assert (
        line.line_id,
        line.item_id,
        line.quantity,
        dict(line.options),
        line.extras,
        line.instructions,
        line.unit_cents,
        line.total_cents,
    ) == (
        "L1",
        "classic_burger",
        2,
        {"size": "large", "patty": "veggie"},
        ("bacon", "cheese"),
        "well done",
        1300,
        2600,
    )
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result)["draft"]["total_cents"] == 2600


def test_valid_additions_survive_malformed_and_unsatisfiable_siblings() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="valid-before",
                name="add_item",
                arguments={"item_id": "classic_burger", "quantity": 1},
            ),
            ToolCall(
                call_id="malformed-middle",
                name="add_item",
                arguments={"item_id": "milkshake"},
            ),
            ToolCall(
                call_id="valid-after",
                name="add_item",
                arguments={"item_id": "fries", "quantity": 1},
            ),
            ToolCall(
                call_id="unsupported-last",
                name="add_item",
                arguments={"item_id": "lobster", "quantity": 1},
            ),
        )),
        AssistantMessage(content="I added the burger and fries; the other requests need changes."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a burger, milkshake, fries, and lobster",
    )

    assert response == {"message": (
        "I added the burger and fries; the other requests need changes.\n\n"
        "Draft order:\n"
        "1 × Classic Burger (size: regular, patty: beef) — $8.50\n"
        "1 × French Fries (size: medium) — $3.50\n"
        "Total: $12.00"
    )}
    results = model.requests[1][-4:]
    assert all(isinstance(result, ToolResultMessage) for result in results)
    assert [_payload(result)["outcome"] for result in results] == [
        "APPLIED", "MALFORMED", "APPLIED", "UNSATISFIABLE",
    ]
    assert [(line.line_id, line.item_id, line.total_cents) for line in session.lines] == [
        ("L1", "classic_burger", 850),
        ("L2", "fries", 350),
    ]
    assert session.revision == 2
    assert session.next_line_number == 3


def test_replayed_applied_call_is_idempotent_within_the_customer_turn() -> None:
    call = ToolCall(
        call_id="same-add-call",
        name="add_item",
        arguments={"item_id": "fries", "quantity": 1},
    )
    model = ScriptedModel(
        AssistantMessage(tool_calls=(call,)),
        AssistantMessage(tool_calls=(call,)),
        AssistantMessage(content="The fries are in your draft."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process("Add fries")

    first_result = model.requests[1][-1]
    second_result = model.requests[2][-1]
    assert isinstance(first_result, ToolResultMessage)
    assert isinstance(second_result, ToolResultMessage)
    assert _payload(first_result)["outcome"] == "APPLIED"
    assert _payload(second_result) == {
        "outcome": "ALREADY_APPLIED",
        "operation": "add_item",
        "draft": _payload(first_result)["draft"],
    }
    assert [(line.line_id, line.item_id, line.quantity) for line in session.lines] == [
        ("L1", "fries", 1),
    ]
    assert session.revision == 1
    assert session.next_line_number == 2


def test_model_failure_after_addition_reports_partial_success_and_canonical_draft() -> None:
    class FailingAfterAdditionModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
        ) -> AssistantMessage:
            self.calls += 1
            if self.calls == 1:
                return AssistantMessage(tool_calls=(ToolCall(
                    call_id="fries-before-failure",
                    name="add_item",
                    arguments={"item_id": "fries", "quantity": 1},
                ),))
            raise RuntimeError("private provider failure")

    session = Session()

    response = TurnProcessor(
        model=FailingAfterAdditionModel(),
        menu=load_menu(),
        session=session,
    ).process("Add fries")

    assert response == {"message": (
        "I couldn't finish that request safely, but some changes were applied.\n"
        "Draft order:\n"
        "1 × French Fries (size: medium) — $3.50\n"
        "Total: $3.50"
    )}
    assert [(line.line_id, line.item_id, line.total_cents) for line in session.lines] == [
        ("L1", "fries", 350),
    ]
    assert session.revision == 1


def test_customer_remedy_blocks_guessed_tool_retry_until_another_customer_turn() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="missing-flavor",
            name="add_item",
            arguments={"item_id": "milkshake", "quantity": 1},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="guessed-flavor",
            name="add_item",
            arguments={
                "item_id": "milkshake",
                "quantity": 1,
                "options": {"flavor": "vanilla"},
            },
        ),)),
        AssistantMessage(content="This response must not be requested."),
    )
    session = Session()

    response = TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a milkshake",
    )

    assert response == {
        "message": "I couldn't finish that request safely. Your draft is unchanged. Please try again.",
    }
    assert len(model.requests) == 2
    results = [
        message for message in session.transcript[0].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert _payload(results[0])["outcome"] == "INCOMPLETE"
    assert _payload(results[1]) == {
        "error": "turn_aborted",
        "reason": "customer_input_required",
        "resolution": "Wait for a new customer turn before using another tool.",
    }
    assert session.lines == []
    assert session.revision == 0


def test_menu_extra_cannot_bypass_validation_through_item_instructions() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="hidden-extra",
            name="add_item",
            arguments={
                "item_id": "classic_burger",
                "quantity": 1,
                "instructions": "add bacon",
            },
        ),)),
        AssistantMessage(content="Bacon needs to be selected as an Extra."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a burger and put bacon on it",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "UNSATISFIABLE"
    assert payload["key"] == "instructions"
    assert [alternative["value"] for alternative in payload["alternatives"]] == [
        "cheese", "bacon", "avocado", "extra_patty",
    ]
    assert session.lines == []
    assert session.revision == 0
    assert session.next_line_number == 1


def test_customizing_one_of_three_burgers_splits_the_selected_serving() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="one-large-burger",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": 1,
                "change": {
                    "type": "customize",
                    "options": {"size": "large"},
                },
            },
        ),)),
        AssistantMessage(content="I made one burger large."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1",
            item_id="classic_burger",
            name="Classic Burger",
            quantity=3,
            options=(("size", "regular"), ("patty", "beef")),
            extras=(),
            instructions="",
            unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make one burger large",
    )

    assert [
        (line.line_id, line.quantity, dict(line.options), line.unit_cents)
        for line in session.lines
    ] == [
        ("L1", 2, {"size": "regular", "patty": "beef"}, 850),
        ("L2", 1, {"size": "large", "patty": "beef"}, 1050),
    ]
    assert sum(line.quantity for line in session.lines) == 3
    assert session.next_line_number == 3
    assert session.revision == 2


def test_ambiguous_milkshake_customization_returns_matching_lines_unchanged() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="ambiguous-shake",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "milkshake"},
                "change": {"type": "customize", "options": {"size": "large"}},
            },
        ),)),
        AssistantMessage(content="Which milkshake should I make large?"),
    )
    session = Session(
        lines=[
            OrderLine(
                line_id="L1", item_id="milkshake", name="Milkshake", quantity=1,
                options=(("size", "regular"), ("flavor", "chocolate")),
                extras=(), instructions="", unit_cents=550,
            ),
            OrderLine(
                line_id="L2", item_id="milkshake", name="Milkshake", quantity=1,
                options=(("size", "regular"), ("flavor", "strawberry")),
                extras=(), instructions="", unit_cents=550,
            ),
        ],
        next_line_number=3,
        revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make the milkshake large",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "INCOMPLETE"
    assert payload["key"] == "target"
    assert [choice["value"] for choice in payload["alternatives"]] == ["L1", "L2"]
    assert "chocolate" in payload["alternatives"][0]["label"]
    assert "strawberry" in payload["alternatives"][1]["label"]
    assert [(line.line_id, dict(line.options)) for line in session.lines] == [
        ("L1", {"size": "regular", "flavor": "chocolate"}),
        ("L2", {"size": "regular", "flavor": "strawberry"}),
    ]
    assert session.revision == 2
    assert session.next_line_number == 3


def test_one_serving_customizes_only_the_earliest_of_multiple_matching_lines() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="earliest-burgers",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": 1,
                "change": {"type": "customize", "add_extras": ["cheese"]},
            },
        ),)),
        AssistantMessage(content="I added cheese to the first burger."),
    )
    session = Session(
        lines=[
            OrderLine(
                line_id="L1", item_id="classic_burger", name="Classic Burger",
                quantity=1, options=(("size", "regular"), ("patty", "beef")),
                extras=(), instructions="", unit_cents=850,
            ),
            OrderLine(
                line_id="L2", item_id="classic_burger", name="Classic Burger",
                quantity=2, options=(("size", "regular"), ("patty", "beef")),
                extras=(), instructions="", unit_cents=850,
            ),
        ],
        next_line_number=3,
        revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add cheese to one burger",
    )

    assert [
        (line.line_id, line.quantity, line.extras, line.unit_cents)
        for line in session.lines
    ] == [
        ("L1", 1, ("cheese",), 950),
        ("L2", 2, (), 850),
    ]
    assert session.revision == 3
    assert session.next_line_number == 3


def test_customization_rejects_more_servings_than_match_without_mutation() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="too-many-burgers",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": 4,
                "change": {"type": "customize", "options": {"size": "large"}},
            },
        ),)),
        AssistantMessage(content="Only three matching burgers are selected."),
    )
    original = OrderLine(
        line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=3,
        options=(("size", "regular"), ("patty", "beef")), extras=(),
        instructions="", unit_cents=850,
    )
    session = Session(lines=[original], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make four burgers large",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result) == {
        "outcome": "INCOMPLETE",
        "remedy": "ask_customer",
        "reason": "Only 3 matching servings are selected.",
        "resolution": "Ask the customer to choose from 1 to 3 matching servings.",
        "subject": "Classic Burger",
        "key": "servings",
        "constraint": {"minimum": 1, "maximum": 3},
    }
    assert session.lines == [original]
    assert session.revision == 1
    assert session.next_line_number == 2


def test_all_customization_preserves_stored_ids_and_groups_only_for_display() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="all-burgers",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": "all",
                "change": {"type": "customize", "options": {"size": "large"}},
            },
        ),)),
        AssistantMessage(content="I made all burgers large."),
    )
    session = Session(
        lines=[
            OrderLine(
                line_id="L1", item_id="classic_burger", name="Classic Burger",
                quantity=1, options=(("size", "regular"), ("patty", "beef")),
                extras=(), instructions="", unit_cents=850,
            ),
            OrderLine(
                line_id="L2", item_id="classic_burger", name="Classic Burger",
                quantity=2, options=(("size", "regular"), ("patty", "beef")),
                extras=(), instructions="", unit_cents=850,
            ),
        ],
        next_line_number=3,
        revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make all burgers large",
    )

    assert [
        (line.line_id, line.quantity, dict(line.options), line.unit_cents)
        for line in session.lines
    ] == [
        ("L1", 1, {"size": "large", "patty": "beef"}, 1050),
        ("L2", 2, {"size": "large", "patty": "beef"}, 1050),
    ]
    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    group = _payload(result)["draft"]["display_groups"]
    assert [(entry["line_ids"], entry["quantity"], entry["total_cents"]) for entry in group] == [
        (["L1", "L2"], 3, 3150),
    ]
    assert session.next_line_number == 3
    assert session.revision == 3


@pytest.mark.parametrize(
    ("change", "selected_instructions"),
    [
        ({"options": {"size": "large"}}, "no onions"),
        ({"instructions": ""}, ""),
        ({"instructions": "cut in half"}, "cut in half"),
    ],
)
def test_split_customization_preserves_clears_or_replaces_selected_instructions(
    change: dict[str, Any],
    selected_instructions: str,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="instruction-patch",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "servings": 1,
                "change": {"type": "customize", **change},
            },
        ),)),
        AssistantMessage(content="I updated one burger."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=2,
            options=(("size", "regular"), ("patty", "beef")), extras=(),
            instructions="no onions", unit_cents=850,
        )],
        next_line_number=2,
        revision=1,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Customize one burger",
    )

    assert [(line.line_id, line.quantity, line.instructions) for line in session.lines] == [
        ("L1", 1, "no onions"),
        ("L2", 1, selected_instructions),
    ]
    snapshot = _payload(model.requests[1][-1])
    assert [
        group["instructions"] for group in snapshot["draft"]["display_groups"]
    ] == ["no onions", selected_instructions]


def test_idempotent_customization_is_already_applied_without_a_new_revision() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="same-cheese",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "change": {
                    "type": "customize",
                    "add_extras": ["cheese", "cheese"],
                },
            },
        ),)),
        AssistantMessage(content="That burger already has cheese."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=1,
            options=(("size", "regular"), ("patty", "beef")), extras=("cheese",),
            instructions="", unit_cents=950,
        )],
        next_line_number=2,
        revision=4,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add cheese to that burger",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result)["outcome"] == "ALREADY_APPLIED"
    assert _payload(result)["operation"] == "update_item"
    assert [(line.line_id, line.extras, line.unit_cents) for line in session.lines] == [
        ("L1", ("cheese",), 950),
    ]
    assert session.revision == 4
    assert session.next_line_number == 2


def test_customization_rejects_an_invalid_recognized_option_value() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-patty",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "change": {"type": "customize", "options": {"patty": "fish"}},
            },
        ),)),
        AssistantMessage(content="Which supported patty would you like?"),
    )
    original = OrderLine(
        line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=1,
        options=(("size", "regular"), ("patty", "beef")), extras=(),
        instructions="", unit_cents=850,
    )
    session = Session(lines=[original], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make the burger a fish patty",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "INCOMPLETE"
    assert payload["key"] == "patty"
    assert [choice["value"] for choice in payload["alternatives"]] == [
        "beef", "chicken", "veggie",
    ]
    assert session.lines == [original]
    assert session.revision == 1


@pytest.mark.parametrize(
    ("change", "expected_key"),
    [
        ({"options": {"temperature": "hot"}}, "temperature"),
        ({"add_extras": ["pickles"]}, "extras"),
        ({"remove_extras": ["bacon"]}, "extras"),
        ({"add_extras": ["cheese"], "remove_extras": ["cheese"]}, "extras"),
        ({"instructions": "add bacon"}, "instructions"),
    ],
)
def test_invalid_customizations_are_rejected_without_mutating_the_draft(
    change: dict[str, Any],
    expected_key: str,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="invalid-customization",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "change": {"type": "customize", **change},
            },
        ),)),
        AssistantMessage(content="That customization is not available."),
    )
    original = OrderLine(
        line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=1,
        options=(("size", "regular"), ("patty", "beef")), extras=(),
        instructions="", unit_cents=850,
    )
    session = Session(lines=[original], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Apply an unavailable customization",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "UNSATISFIABLE"
    assert payload["key"] == expected_key
    assert session.lines == [original]
    assert session.revision == 1
    assert session.next_line_number == 2


def test_customization_uses_extra_set_semantics_and_reprices_the_line() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="change-extras",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "change": {
                    "type": "customize",
                    "add_extras": ["cheese", "cheese"],
                    "remove_extras": ["bacon"],
                },
            },
        ),)),
        AssistantMessage(content="I replaced bacon with cheese."),
    )
    session = Session(
        lines=[OrderLine(
            line_id="L1", item_id="classic_burger", name="Classic Burger", quantity=1,
            options=(("size", "regular"), ("patty", "beef")), extras=("bacon",),
            instructions="", unit_cents=1000,
        )],
        next_line_number=2,
        revision=1,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Replace the bacon with cheese",
    )

    assert [
        (line.line_id, line.extras, line.unit_cents, line.total_cents)
        for line in session.lines
    ] == [
        ("L1", ("cheese",), 950, 950),
    ]
    assert session.revision == 2


def test_customization_rejects_a_missing_line_id_target() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="missing-line",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L99"},
                "change": {"type": "customize", "options": {"size": "large"}},
            },
        ),)),
        AssistantMessage(content="That line is not in the draft."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make line 99 large",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert _payload(result)["outcome"] == "UNSATISFIABLE"
    assert _payload(result)["key"] == "target"
    assert session.lines == []
    assert session.revision == 0


def test_unlisted_addition_cannot_bypass_validation_through_item_instructions() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="hidden-unlisted-extra",
            name="add_item",
            arguments={
                "item_id": "classic_burger",
                "quantity": 1,
                "instructions": "add pickles",
            },
        ),)),
        AssistantMessage(content="Pickles are not a supported Extra."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a burger with pickles",
    )

    result = model.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    payload = _payload(result)
    assert payload["outcome"] == "UNSATISFIABLE"
    assert payload["key"] == "instructions"
    assert payload["note"] == "Instruction requested the unselected addition pickles."
    assert session.lines == []
    assert session.revision == 0


def test_negative_item_instruction_is_preserved_without_selecting_the_extra() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="negative-instruction",
            name="add_item",
            arguments={
                "item_id": "classic_burger",
                "quantity": 1,
                "instructions": "no bacon",
            },
        ),)),
        AssistantMessage(content="Added the burger without bacon."),
    )
    session = Session()

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Add a burger, no bacon",
    )

    assert [(line.item_id, line.instructions) for line in session.lines] == [
        ("classic_burger", "no bacon"),
    ]
    assert session.revision == 1
