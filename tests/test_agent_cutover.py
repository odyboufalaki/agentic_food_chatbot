import json

import pytest

from agent import FoodOrderAgent
from food_ordering.model_adapter import (
    AbortedToolResult,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
)
from food_ordering.session import Session
from food_ordering.submission import SubmissionResult
from model_fakes import FailingAfterScriptModel, RecordingSubmitter, ScriptedModel


def test_send_uses_the_outcome_driven_turn_processor(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="burger",
                name="add_item",
                arguments={"item_id": "classic_burger", "quantity": 1},
            ),
            ToolCall(
                call_id="shake",
                name="add_item",
                arguments={"item_id": "milkshake", "quantity": 1},
            ),
        )),
        AssistantMessage(content="I added the burger. Which milkshake flavor would you like?"),
    )
    session = Session()
    agent = FoodOrderAgent(
        model=model,
        session=session,
        log_path=tmp_path / "turns.jsonl",
    )

    response = agent.send("A burger and a milkshake")

    assert "Which milkshake flavor" in response["message"]
    assert "1 × Classic Burger" in response["message"]
    assert [line.item_id for line in session.lines] == ["classic_burger"]

    record = json.loads((tmp_path / "turns.jsonl").read_text())
    assert record["tool_protocol_version"] == 1
    assert [call["name"] for call in record["model_calls"]] == [
        "add_item", "add_item",
    ]
    assert [outcome["outcome"] for outcome in record["outcomes"]] == [
        "APPLIED", "INCOMPLETE",
    ]


def test_send_renders_menu_and_draft_reads_from_authoritative_python_state(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="menu", name="show_menu", arguments={"item_ids": ["milkshake"]},
        ),)),
        AssistantMessage(content="Lobster costs $1."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="draft", name="show_draft", arguments={},
        ),)),
        AssistantMessage(content="Your draft contains lobster."),
    )
    agent = FoodOrderAgent(model=model, log_path=tmp_path / "turns.jsonl")

    menu = agent.send("What milkshakes are available?")["message"]
    draft = agent.send("What is in my draft?")["message"]

    assert "Milkshake" in menu and "vanilla" in menu and "lobster" not in menu.lower()
    assert draft == "Draft order:\n\nTotal: $0.00"


def test_mixed_read_and_incomplete_operation_disclose_the_outstanding_choice(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="menu", name="show_menu",
                arguments={"item_ids": ["milkshake"]},
            ),
            ToolCall(
                call_id="shake", name="add_item",
                arguments={"item_id": "milkshake", "quantity": 1},
            ),
        )),
        AssistantMessage(content="Which flavor would you like?"),
    )
    agent = FoodOrderAgent(model=model, log_path=tmp_path / "turns.jsonl")

    response = agent.send("Show shakes and add one")

    assert "Menu" in response["message"]
    assert "Outstanding question:" in response["message"]
    assert "flavor" in response["message"]
    assert "vanilla" in response["message"]


def test_mutation_with_draft_read_renders_the_authoritative_draft_once(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="add", name="add_item",
                arguments={"item_id": "fries", "quantity": 1},
            ),
            ToolCall(call_id="draft", name="show_draft", arguments={}),
        )),
        AssistantMessage(content="Added fries."),
    )
    agent = FoodOrderAgent(model=model, log_path=tmp_path / "turns.jsonl")

    response = agent.send("Add fries and show the draft")

    assert response["message"].count("Draft order:") == 1
    assert "French Fries" in response["message"]


def test_send_reconstructs_clarification_from_the_transcript(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="incomplete",
            name="add_item",
            arguments={"item_id": "milkshake", "quantity": 1},
        ),)),
        AssistantMessage(content="Which flavor would you like?"),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="fresh-call",
            name="add_item",
            arguments={
                "item_id": "milkshake",
                "quantity": 1,
                "options": {"flavor": "oreo"},
            },
        ),)),
        AssistantMessage(content="I added an Oreo milkshake."),
    )
    session = Session()
    agent = FoodOrderAgent(model=model, session=session, log_path=tmp_path / "turns.jsonl")

    first = agent.send("Add a milkshake")
    second = agent.send("Oreo")

    assert first == {"message": "Which flavor would you like?"}
    assert "Milkshake" in second["message"] and "flavor: oreo" in second["message"]
    assert len(model.requests[2]) == 5


def test_send_handles_serving_split_ambiguity_and_clean_replacement(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add-two",
            name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 2},
        ),)),
        AssistantMessage(content="Added two burgers."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="split",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "servings": 1,
                "change": {"type": "customize", "options": {"size": "large"}},
            },
        ),)),
        AssistantMessage(content="Made one large."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="ambiguous",
            name="update_item",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "change": {"type": "customize", "add_extras": ["cheese"]},
            },
        ),)),
        AssistantMessage(content="Which burger should get cheese?"),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="replace",
            name="update_item",
            arguments={
                "target": {"type": "line", "line_id": "L2"},
                "change": {"type": "replace", "item_id": "spicy_burger"},
            },
        ),)),
        AssistantMessage(content="Replaced the large burger."),
    )
    session = Session()
    agent = FoodOrderAgent(model=model, session=session, log_path=tmp_path / "turns.jsonl")

    agent.send("Two burgers")
    agent.send("Make one large")
    ambiguous = agent.send("Add cheese to the burger")
    replaced = agent.send("Replace the large one with a spicy burger")

    assert ambiguous == {"message": "Which burger should get cheese?"}
    assert [(line.line_id, line.item_id, dict(line.options)) for line in session.lines] == [
        ("L1", "classic_burger", {"size": "regular", "patty": "beef"}),
        ("L2", "spicy_burger", {"size": "regular", "spice_level": "medium"}),
    ]
    assert "Spicy Jalapeño Burger" in replaced["message"]


def test_send_supports_quantity_removal_instructions_and_clear(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 2},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="quantity", name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 3,
            },
        ),)),
        AssistantMessage(content="Updated quantity."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="instructions", name="set_order_instructions",
            arguments={"instructions": "no cutlery"},
        ),)),
        AssistantMessage(content="Noted."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="remove", name="remove_item",
            arguments={"target": {"type": "line", "line_id": "L1"}},
        ),)),
        AssistantMessage(content="Removed."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="clear", name="clear_draft", arguments={},
        ),)),
        AssistantMessage(content="Cleared."),
    )
    session = Session()
    agent = FoodOrderAgent(model=model, session=session, log_path=tmp_path / "turns.jsonl")

    agent.send("Two burgers")
    resized = agent.send("Make that three")
    instructed = agent.send("No cutlery")
    removed = agent.send("Remove the burgers")
    cleared = agent.send("Clear everything")

    assert "3 × Classic Burger" in resized["message"] and "$25.50" in resized["message"]
    assert "General instructions: no cutlery" in instructed["message"]
    assert "Classic Burger" not in removed["message"]
    assert session.lines == [] and session.instructions == ""
    assert "Total: $0.00" in cleared["message"]


def test_over_limit_draft_remains_editable_but_cannot_be_reviewed(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 6},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Reduce the order below the checkout limit."),
    )
    session = Session()
    agent = FoodOrderAgent(model=model, session=session, log_path=tmp_path / "turns.jsonl")

    added = agent.send("Six burgers")
    blocked = agent.send("Review")

    assert "$51.00" in added["message"]
    assert "reduce" in blocked["message"].lower()
    assert session.review_snapshot is None
    assert session.revision == 1


def test_send_reviews_then_submits_exactly_once_and_exposes_only_mcp_call(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add",
            name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Untrusted review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="submit", name="submit_order", arguments={},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="repeat", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted",
        invoked=True,
        result={"success": True, "order_id": "ORD-17", "total": 8.5},
    ))
    agent = FoodOrderAgent(
        model=model,
        submitter=submitter,
        log_path=tmp_path / "turns.jsonl",
    )

    added = agent.send("Add a burger")
    review = agent.send("Review")
    submitted = agent.send("Yes")
    repeated = agent.send("Yes")

    assert "tool_calls" not in added and "tool_calls" not in review
    assert review["message"].endswith("Please confirm: submit this exact order?")
    assert len(submitter.calls) == 1
    assert submitter.calls[0] == {
        "items": [{
            "item_id": "classic_burger",
            "quantity": 1,
            "options": {"size": "regular", "patty": "beef"},
            "extras": [],
        }],
    }
    assert submitted["tool_calls"][0]["arguments"] == submitter.calls[0]
    assert "tool_calls" not in repeated
    assert submitted["message"] == repeated["message"]


def test_accepted_submission_pairs_later_calls_in_the_same_model_batch(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(
            ToolCall(call_id="submit", name="submit_order", arguments={}),
            ToolCall(call_id="after-submit", name="show_draft", arguments={}),
        )),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True,
        result={"success": True, "order_id": "ORD-paired"},
    ))
    session = Session()
    agent = FoodOrderAgent(
        model=model, session=session, submitter=submitter,
        log_path=tmp_path / "turns.jsonl",
    )

    agent.send("Fries")
    agent.send("Review")
    agent.send("Yes")

    results = [
        message for message in session.transcript[-1].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [result.call_id for result in results] == ["submit", "after-submit"]
    assert isinstance(results[1].payload, AbortedToolResult)
    assert results[1].payload.reason == "submission_completed"


def test_send_reports_rejection_and_requires_a_later_explicit_retry(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="first", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Please tell me if you want another attempt."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="retry", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(
        SubmissionResult(
            status="rejected", invoked=True,
            result={
                "success": False,
                "error": "Authorization: Bearer private-backend-token",
            },
        ),
        SubmissionResult(
            status="submitted", invoked=True,
            result={"success": True, "order_id": "ORD-retry"},
        ),
    )
    agent = FoodOrderAgent(model=model, submitter=submitter, log_path=tmp_path / "turns.jsonl")

    agent.send("Fries")
    agent.send("Review")
    rejected = agent.send("Yes")
    retried = agent.send("Try again")

    assert "rejected" in rejected["message"].lower()
    assert "private-backend-token" not in rejected["message"]
    assert len(rejected["tool_calls"]) == 1
    assert len(submitter.calls) == 2
    assert "ORD-retry" in retried["message"]


def test_rejection_stays_customer_visible_when_followup_model_call_fails(tmp_path) -> None:
    model = FailingAfterScriptModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="submit", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="rejected", invoked=True,
        result={"success": False, "error": "Kitchen busy"},
    ))
    agent = FoodOrderAgent(model=model, submitter=submitter, log_path=tmp_path / "turns.jsonl")

    agent.send("Fries")
    agent.send("Review")
    response = agent.send("Yes")

    assert "restaurant rejected" in response["message"].lower()
    assert "currently busy" in response["message"]
    assert len(response["tool_calls"]) == 1


def test_uncertain_submission_locks_later_mutation_and_resubmission(tmp_path) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="submit", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="This text is ignored."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="edit", name="clear_draft", arguments={},
        ),)),
        AssistantMessage(content="I cannot change it while acceptance is uncertain."),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="uncertain", invoked=True,
        result={"client_error": "lost_response", "outcome": "uncertain"},
    ))
    session = Session()
    agent = FoodOrderAgent(
        model=model, session=session, submitter=submitter,
        log_path=tmp_path / "turns.jsonl",
    )

    agent.send("Fries")
    agent.send("Review")
    uncertain = agent.send("Yes")
    locked = agent.send("Clear it")

    assert "may have been accepted" in uncertain["message"]
    assert len(submitter.calls) == 1
    assert "may have been accepted" in locked["message"]
    assert [line.item_id for line in session.lines] == ["fries"]


@pytest.mark.parametrize(
    ("outcome", "expected_status", "public_call"),
    [
        (
            SubmissionResult(
                status="application_error", invoked=True,
                result={"code": -32602, "message": "invalid arguments"},
            ),
            "application_error",
            True,
        ),
        (
            SubmissionResult(
                status="not_sent", invoked=False,
                result={"client_error": "configuration", "outcome": "not_sent"},
            ),
            "not_sent",
            False,
        ),
    ],
)
def test_send_preserves_non_rejection_submission_dispositions(
    tmp_path, outcome: SubmissionResult, expected_status: str, public_call: bool,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="submit", name="submit_order", arguments={},
        ),)),
        AssistantMessage(content="Untrusted submission status."),
    )
    session = Session()
    agent = FoodOrderAgent(
        model=model, session=session, submitter=RecordingSubmitter(outcome),
        log_path=tmp_path / "turns.jsonl",
    )

    agent.send("Fries")
    agent.send("Review")
    response = agent.send("Yes")

    assert session.status == expected_status
    assert ("tool_calls" in response) is public_call
    assert expected_status.replace("_", " ") in response["message"].lower()


def test_acceptance_survives_logging_failure_without_duplicate_submission(
    tmp_path, capsys,
) -> None:
    log_path = tmp_path / "log-directory"
    log_path.mkdir()
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="add", name="add_item",
            arguments={"item_id": "fries", "quantity": 1},
        ),)),
        AssistantMessage(content="Added."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="review", name="propose_submission", arguments={},
        ),)),
        AssistantMessage(content="Review."),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="submit", name="submit_order", arguments={},
        ),)),
        AssistantMessage(tool_calls=(ToolCall(
            call_id="repeat", name="submit_order", arguments={},
        ),)),
    )
    submitter = RecordingSubmitter(SubmissionResult(
        status="submitted", invoked=True,
        result={"success": True, "order_id": "ORD-logged"},
    ))
    agent = FoodOrderAgent(model=model, submitter=submitter, log_path=log_path)

    agent.send("Fries")
    agent.send("Review")
    accepted = agent.send("Yes")
    repeated = agent.send("Yes again")

    assert accepted["message"] == repeated["message"]
    assert len(submitter.calls) == 1
    assert "logging failed" in capsys.readouterr().err.lower()


def test_partial_success_fallback_identifies_the_outstanding_question(tmp_path) -> None:
    model = FailingAfterScriptModel(AssistantMessage(tool_calls=(
        ToolCall(
            call_id="burger", name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 1},
        ),
        ToolCall(
            call_id="shake", name="add_item",
            arguments={"item_id": "milkshake", "quantity": 1},
        ),
    )))
    agent = FoodOrderAgent(model=model, log_path=tmp_path / "turns.jsonl")

    response = agent.send("A burger and milkshake")

    assert "some changes were applied" in response["message"]
    assert "flavor" in response["message"]
    assert "vanilla" in response["message"]


def test_partial_success_fallback_lists_each_outstanding_operation(tmp_path) -> None:
    model = ScriptedModel(AssistantMessage(tool_calls=(
        ToolCall(
            call_id="burger", name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 1},
        ),
        ToolCall(
            call_id="shake-one", name="add_item",
            arguments={"item_id": "milkshake", "quantity": 1},
        ),
        ToolCall(
            call_id="shake-two", name="add_item",
            arguments={"item_id": "milkshake", "quantity": 2},
        ),
    )))
    agent = FoodOrderAgent(model=model, log_path=tmp_path / "turns.jsonl")

    response = agent.send("A burger and two shake selections")

    assert response["message"].count("Outstanding question:") == 2
