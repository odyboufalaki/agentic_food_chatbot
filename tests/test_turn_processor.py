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
from food_ordering.order import OrderLine
from food_ordering.session import Session
from food_ordering.turn_processor import TurnProcessor


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

    assert response == {"message": "We have burgers, pizza, sides, drinks, and desserts."}
    assert [tool.name for tool in model.tool_specs[0]] == ["show_menu", "show_draft"]
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

    assert response == {"message": "You have two large fries with parmesan."}
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

    assert response == {"message": "Your draft is empty."}
    malformed = model.requests[1][-1]
    assert isinstance(malformed, ToolResultMessage)
    assert malformed.call_id == "bad-1"
    assert _payload(malformed)["outcome"] == "MALFORMED"
    assert _payload(malformed)["remedy"] == "correct_tool"
    corrected = model.requests[2][-1]
    assert isinstance(corrected, ToolResultMessage)
    assert corrected.call_id == "fixed-1"
    assert _payload(corrected)["outcome"] == "RESULT"


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


def test_read_only_processor_rejects_mutation_and_leaves_authoritative_state_unchanged() -> None:
    line = OrderLine(
        item_id="fries",
        name="French Fries",
        quantity=1,
        options=(("size", "medium"),),
        extras=(),
        unit_cents=350,
        line_id="L2",
    )
    session = Session(
        lines=[line],
        instructions="ring the bell",
        next_line_number=3,
        revision=4,
        reviewed_revision=4,
        status="rejected",
        receipt_message="prior receipt text",
        rejected_payload={"error": "kitchen busy"},
        retry_requires_review=True,
    )
    authoritative_before = (
        list(session.lines),
        session.instructions,
        session.next_line_number,
        session.revision,
        session.reviewed_revision,
        session.status,
        session.receipt_message,
        session.rejected_payload,
        session.application_error_payload,
        session.retry_requires_review,
        session.pending_change,
    )
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="mutation-1",
            name="add_item",
            arguments={"item_id": "classic_burger", "quantity": 1},
        ),)),
        AssistantMessage(content="I can't change the order in this read-only flow."),
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process("Add a burger")

    rejection = model.requests[1][-1]
    assert isinstance(rejection, ToolResultMessage)
    assert _payload(rejection)["outcome"] == "UNSATISFIABLE"
    assert (
        list(session.lines),
        session.instructions,
        session.next_line_number,
        session.revision,
        session.reviewed_revision,
        session.status,
        session.receipt_message,
        session.rejected_payload,
        session.application_error_payload,
        session.retry_requires_review,
        session.pending_change,
    ) == authoritative_before


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

    assert first == {"message": "Fries are available and your draft is empty."}
    assert second == {"message": "What size fries would you like?"}
    first_turn = session.transcript[0].messages
    assert first_turn[0] == CustomerMessage("Do you have fries, and what is in my draft?")
    assert isinstance(first_turn[1], AssistantMessage)
    assert first_turn[1].content == "Preliminary text is transcript-only."
    assert [message.call_id for message in first_turn[2:4]
            if isinstance(message, ToolResultMessage)] == ["menu", "draft"]
    assert first_turn[-1] == AssistantMessage(
        content="Fries are available and your draft is empty.",
    )
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
    mutation = {"item_id": "classic_burger", "quantity": 1}
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="mutation-1", name="add_item", arguments=mutation),
        )),
        AssistantMessage(tool_calls=(
            ToolCall(call_id="mutation-2", name="add_item", arguments=mutation),
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
    assert [result.call_id for result in results] == ["mutation-1", "mutation-2"]
