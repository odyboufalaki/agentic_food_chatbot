import json

from agent import FoodOrderAgent
from food_ordering.interpretation import ModelFailure
from food_ordering.submission import MCPSubmitter, SubmissionSettings
from test_agent import ScriptedInterpreter, add
from test_submission import RECEIPT, RestaurantTransport


class CapturingInterpreter(ScriptedInterpreter):
    def __init__(self, *proposals):
        super().__init__(*proposals)
        self.contexts = []

    def interpret(self, **context):
        self.contexts.append(context)
        return super().interpret(**context)


def test_missing_required_choice_holds_the_whole_request_until_answered(tmp_path):
    transport = RestaurantTransport()
    interpreter = CapturingInterpreter(
        {"operations": [add("cola"), add("milkshake")]},
        {"operations": [add("cola"), add("milkshake", options={"flavor": "oreo"})]},
        {"operations": [{"type": "submit"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent = FoodOrderAgent(
        interpreter=interpreter,
        submitter=MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                               transport=transport),
        log_path=tmp_path / "turns.jsonl",
    )
    question = agent.send("Add a cola and a milkshake")["message"]
    assert "flavor" in question and "oreo" in question
    assert interpreter.contexts[0]["draft"] == []
    resolved = agent.send("Oreo")["message"]
    assert "Soft Drink" in resolved and "Milkshake" in resolved and "Total: $7.50" in resolved
    pending = interpreter.contexts[1]["pending_clarification"]
    assert pending["original_message"] == "Add a cola and a milkshake"
    assert pending["reason"] == "required_option"
    assert len(pending["proposal"]["operations"]) == 2
    assert interpreter.contexts[1]["draft"] == []
    assert "confirm" in agent.send("Submit")["message"].lower()
    assert "ORD-12345" in agent.send("Yes")["message"]
    assert len(transport.calls) == 1
    assert len(transport.calls[0]["arguments"]["items"]) == 2
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[1]["operations"][0] == {
        "type": "clarification_resolved", "reason": "required_option",
    }


def test_ambiguous_target_and_other_operation_apply_together_after_clarification(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("burger"), add("burger", options={"size": "large"})]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "burger"}}, add("fries")]},
        {"operations": [{"type": "clarify", "reason": "target"}]},
        {"operations": [{"type": "remove_line", "target": {
            "item_id": "burger", "options": {"size": "large"},
        }}, add("fries")]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    original = agent.send("A regular burger and a separate large burger")
    question = agent.send("Remove the burger and add fries")["message"]
    assert "which" in question.lower()
    unclear = agent.send("That one")["message"]
    assert "which" in unclear.lower()
    assert interpreter.contexts[2]["pending_clarification"] is not None
    resolved = agent.send("The large burger")["message"]
    assert "size: large" not in resolved
    assert "size: regular" in resolved and "French Fries" in resolved
    assert "Total: $12.00" in resolved
    assert "French Fries" not in original["message"]


def test_cancel_pending_change_preserves_draft_but_cancel_order_clears_both(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("burger")]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "burger"}}, add("milkshake")]},
        {"operations": [{"type": "cancel_pending"}]},
        {"operations": [{"type": "summary"}]},
        {"operations": [{"type": "clarify", "reason": "quantity"}]},
        {"operations": [{"type": "clear_draft"}]},
        {"operations": [{"type": "summary"}]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    original = agent.send("A burger")
    assert "flavor" in agent.send("Remove the burger and add a milkshake")["message"]
    canceled = agent.send("Cancel that change")["message"]
    assert "canceled" in canceled.lower() and "Classic Burger" in canceled
    assert agent.send("Show draft") == original
    assert "quantity" in agent.send("Remove some burgers")["message"].lower()
    cleared = agent.send("Cancel my whole order")["message"]
    assert "Total: $0.00" in cleared and "Burger" not in cleared
    assert agent.send("Show draft")["message"] == "Draft order:\n\nTotal: $0.00"


def test_abandonment_handles_new_request_and_old_answer_never_revives(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("milkshake")]},
        {"operations": [{"type": "abandon_pending"}, add("fries")]},
        {"operations": [{"type": "unsupported", "reason": "unclear"}]},
        {"operations": [{"type": "summary"}]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    assert "flavor" in agent.send("A milkshake")["message"]
    replacement = agent.send("Never mind; add fries instead")["message"]
    assert "French Fries" in replacement and "Milkshake" not in replacement
    assert interpreter.contexts[1]["pending_clarification"] is not None
    assert "unchanged" in agent.send("Chocolate")["message"]
    summary = agent.send("Show draft")["message"]
    assert "French Fries" in summary and "Milkshake" not in summary
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[1]["operations"][0] == {"type": "abandon_pending"}


def test_invalid_resolution_is_revalidated_and_keeps_the_pending_change(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("cola"), add("milkshake")]},
        {"operations": [add("cola"), add("milkshake", options={"flavor": "vanilla bean"})]},
        {"operations": [add("cola"), add("milkshake", options={"flavor": "chocolate"})]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    agent.send("A cola and milkshake")
    rejected = agent.send("Vanilla bean")["message"]
    assert "supported flavor" in rejected and "No changes have been applied" in rejected
    resolved = agent.send("Chocolate")["message"]
    assert interpreter.contexts[2]["draft"] == []
    assert interpreter.contexts[2]["pending_clarification"] is not None
    assert "Soft Drink" in resolved and "flavor: chocolate" in resolved


def test_successive_answers_retain_values_resolved_before_the_last_question(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("milkshake"), {"type": "clarify", "reason": "quantity"}]},
        {"operations": [
            add("milkshake", options={"flavor": "chocolate"}),
            {"type": "clarify", "reason": "quantity"},
        ]},
        {"operations": [
            add("milkshake", options={"flavor": "chocolate"}), add("fries", quantity=2),
        ]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    assert "quantity" in agent.send("A milkshake and some fries")["message"].lower()
    assert "quantity" in agent.send("Chocolate, but I am unsure how many fries")["message"].lower()
    resolved = agent.send("Two fries")["message"]
    partially_resolved = interpreter.contexts[2]["pending_clarification"]["proposal"]["operations"]
    assert partially_resolved[0]["options"] == {"flavor": "chocolate"}
    assert "flavor: chocolate" in resolved and "2 × French Fries" in resolved


def test_quantity_answer_cannot_invent_an_unanswered_required_option(tmp_path):
    interpreter = ScriptedInterpreter(
        {"operations": [
            {"type": "clarify", "reason": "quantity"},
            add("milkshake", options={"flavor": "vanilla"}, extras=["cherry_on_top"]),
        ]},
        {"operations": [
            add("burger"),
            add("milkshake", options={"flavor": "vanilla"}, extras=["cherry_on_top"]),
        ]},
        {"operations": [
            add("burger"),
            add("milkshake", options={"flavor": "vanilla"}, extras=["cherry_on_top"]),
        ]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    assert "quantity" in agent.send("I want burgers and milkshake with cherry on top")["message"].lower()
    response = agent.send("I want one burger")["message"]
    assert "flavor" in response.lower()
    assert "vanilla, chocolate, strawberry, oreo" in response
    assert "No changes have been applied" in response
    resolved = agent.send("Vanilla")["message"]
    assert "flavor: vanilla" in resolved and "extras: cherry_on_top" in resolved


def test_yes_and_failed_interpretation_during_clarification_never_submit(tmp_path):
    transport = RestaurantTransport()
    interpreter = CapturingInterpreter(
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "clarify", "reason": "quantity"}]},
        {"operations": [{"type": "clarify", "reason": "quantity"}]},
        ModelFailure("model_timeout"),
        {"operations": [{"type": "set_quantity", "target": {"item_id": "burger"}, "quantity": 2}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent = FoodOrderAgent(
        interpreter=interpreter,
        submitter=MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                               transport=transport),
        log_path=tmp_path / "turns.jsonl",
    )
    agent.send("A burger")
    agent.send("Submit")
    agent.send("Remove some burgers")
    assert "quantity" in agent.send("Yes")["message"].lower()
    assert "unchanged" in agent.send("Maybe two")["message"]
    assert transport.calls == []
    changed = agent.send("Two")["message"]
    assert "2 × Classic Burger" in changed
    review = agent.send("Yes")["message"]
    assert "confirm" in review.lower() and transport.calls == []
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_checkout_reply_keeps_pending_proposal_and_abandonment_can_start_review(tmp_path):
    interpreter = CapturingInterpreter(
        {"operations": [add("burger")]},
        {"operations": [add("milkshake")]},
        {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "abandon_pending"}, {"type": "review"}]},
    )
    agent = FoodOrderAgent(interpreter=interpreter, log_path=tmp_path / "turns.jsonl")
    agent.send("A burger")
    agent.send("Add a milkshake")
    assert "flavor" in agent.send("Submit")["message"]
    review = agent.send("Never mind the shake; review my burger")["message"]
    still_pending = interpreter.contexts[3]["pending_clarification"]
    assert still_pending["proposal"]["operations"][0]["item_id"] == "milkshake"
    assert "Classic Burger" in review and "confirm" in review.lower()


def test_canceling_ambiguous_edit_after_rejection_still_requires_fresh_review(tmp_path):
    transport = RestaurantTransport()
    transport.receipt = {"success": False, "error": "busy"}
    interpreter = ScriptedInterpreter(
        {"operations": [add("burger"), add("burger", options={"size": "large"})]},
        {"operations": [{"type": "submit"}]}, {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "burger"}}]},
        {"operations": [{"type": "cancel_pending"}]},
        {"operations": [{"type": "retry_submission"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent = FoodOrderAgent(
        interpreter=interpreter,
        submitter=MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                               transport=transport),
        log_path=tmp_path / "turns.jsonl",
    )
    agent.send("Two different burgers")
    agent.send("Submit")
    assert "rejected" in agent.send("Yes")["message"]
    transport.receipt = dict(RECEIPT)
    assert "which" in agent.send("Remove the burger")["message"].lower()
    agent.send("Cancel that change")
    review = agent.send("Try again")["message"]
    assert "confirm" in review.lower() and len(transport.calls) == 1
    agent.send("Yes")
    assert len(transport.calls) == 2


def test_clarification_lifecycle_is_recorded_without_partial_operations(tmp_path):
    path = tmp_path / "turns.jsonl"
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("cola"), add("milkshake")]},
        {"operations": [{"type": "clarify", "reason": "required_option"}]},
        {"operations": [{"type": "cancel_pending"}]},
    ), log_path=path)
    agent.send("A cola and a milkshake")
    agent.send("Something nice")
    agent.send("Cancel that change")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["operations"] == [{"type": "clarification_requested", "reason": "required_option"}]
    assert records[0]["totals"] == {"before_cents": 0, "after_cents": 0}
    assert records[1]["operations"] == [{"type": "clarification_requested", "reason": "required_option"}]
    assert records[2]["operations"] == [{"type": "cancel_pending"}]
    assert records[2]["state_transition"]["before"]["pending_change"] is True
    assert records[2]["state_transition"]["after"]["pending_change"] is False
