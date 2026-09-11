import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from mistralai.client import Mistral

from food_ordering.interpretation import ModelFailure
from food_ordering.menu import load_menu
from food_ordering.mistral_adapter import MistralToolModel
from food_ordering.model_adapter import (
    AssistantMessage,
    CustomerMessage,
    ToolCall,
    ToolResultMessage,
    TranscriptTurn,
)
from food_ordering.tool_protocol import DraftSnapshot, ResultPayload, protocol_schema
from food_ordering.session import Session
from food_ordering.turn_processor import TurnProcessor, _tool_specs


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "mistral-small-latest",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def _with_sdk(
    handler: Callable[[httpx.Request], httpx.Response],
    action: Callable[[Mistral], None],
) -> None:
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with Mistral(api_key="test-key", client=http_client) as sdk:
            action(sdk)


def test_tool_call_and_paired_result_round_trip_through_mistral() -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _completion(tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {"name": "show_draft", "arguments": "{}"},
            }], finish_reason="tool_calls")
        return _completion(content="Your draft is empty.")

    def exercise(sdk: Mistral) -> None:
        model = MistralToolModel(client=sdk)
        first = model.complete(
            messages=[CustomerMessage("What is in my draft?")],
            tools=_tool_specs(),
        )
        assert first == AssistantMessage(tool_calls=(
            ToolCall(call_id="call-1", name="show_draft", arguments={}),
        ))

        result = ToolResultMessage(
            call_id="call-1",
            name="show_draft",
            payload=ResultPayload(outcome="RESULT", result=DraftSnapshot(
                revision=0,
                lines=[],
                display_groups=[],
                general_instructions="",
                total_cents=0,
                checkout_state="draft",
            )),
        )
        final = model.complete(
            messages=[CustomerMessage("What is in my draft?"), first, result],
            tools=_tool_specs(),
        )
        assert final == AssistantMessage(content="Your draft is empty.")

    _with_sdk(handle, exercise)

    assert {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in requests[0]["tools"]
    } == protocol_schema()["tools"]
    assert requests[1]["messages"][-2]["tool_calls"][0]["id"] == "call-1"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "name": "show_draft",
        "tool_call_id": "call-1",
        "content": (
            '{"outcome":"RESULT","result":{"revision":0,"lines":[],'
            '"display_groups":[],"general_instructions":"","total_cents":0,'
            '"checkout_state":"draft"}}'
        ),
    }


def test_multiple_unknown_and_malformed_calls_remain_observable_in_emitted_order() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return _completion(tool_calls=[
            {
                "id": "valid",
                "type": "function",
                "function": {"name": "show_menu", "arguments": '{"item_ids":[]}'},
            },
            {
                "id": "unknown",
                "type": "function",
                "function": {"name": "make_coffee", "arguments": "{}"},
            },
            {
                "id": "malformed",
                "type": "function",
                "function": {"name": "add_item", "arguments": "{not-json"},
            },
        ], finish_reason="tool_calls")

    observed: list[AssistantMessage] = []

    def exercise(sdk: Mistral) -> None:
        observed.append(MistralToolModel(client=sdk).complete(
            messages=[CustomerMessage("Coffee and the menu")],
            tools=_tool_specs(),
        ))

    _with_sdk(handle, exercise)

    assert observed == [AssistantMessage(tool_calls=(
        ToolCall(call_id="valid", name="show_menu", arguments={"item_ids": []}),
        ToolCall(call_id="unknown", name="make_coffee", arguments={}),
        ToolCall(call_id="malformed", name="add_item", arguments="{not-json"),
    ))]


@pytest.mark.parametrize(
    ("outcomes", "expected_calls", "category"),
    [
        ([429, 429], 2, "rate_limit"),
        ([503, 503], 2, "model_unavailable"),
        (["timeout", "timeout"], 2, "model_unavailable"),
        ([401], 1, "authentication"),
        ([422], 1, "configuration"),
    ],
)
def test_only_transient_provider_failures_receive_one_bounded_retry(
    outcomes: list[int | str],
    expected_calls: int,
    category: str,
) -> None:
    remaining = iter(outcomes)
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        outcome = next(remaining)
        if outcome == "timeout":
            raise httpx.ReadTimeout("private provider detail", request=request)
        return httpx.Response(outcome, json={"message": "private provider detail"})

    def exercise(sdk: Mistral) -> None:
        with pytest.raises(ModelFailure) as error:
            MistralToolModel(client=sdk).complete(
                messages=[CustomerMessage("Show my draft")],
                tools=_tool_specs(),
            )
        assert error.value.category == category

    _with_sdk(handle, exercise)
    assert calls == expected_calls


def test_length_limited_response_is_reported_as_truncated_with_its_calls() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return _completion(
            content="I only decoded part of this.",
            tool_calls=[{
                "id": "partial",
                "type": "function",
                "function": {"name": "add_item", "arguments": '{"item_id":"fries"}'},
            }],
            finish_reason="length",
        )

    observed: list[AssistantMessage] = []

    def exercise(sdk: Mistral) -> None:
        observed.append(MistralToolModel(client=sdk).complete(
            messages=[CustomerMessage("Add fries")],
            tools=_tool_specs(),
        ))

    _with_sdk(handle, exercise)

    assert observed == [AssistantMessage(
        content="I only decoded part of this.",
        tool_calls=(ToolCall(
            call_id="partial",
            name="add_item",
            arguments={"item_id": "fries"},
        ),),
        completion_status="truncated",
    )]


def test_missing_provider_call_id_cannot_mutate_the_draft() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completion(tool_calls=[{
                "type": "function",
                "function": {
                    "name": "add_item",
                    "arguments": '{"item_id":"fries","quantity":1}',
                },
            }], finish_reason="tool_calls")
        return _completion(content="I could not apply that call.")

    session = Session()

    def exercise(sdk: Mistral) -> None:
        response = TurnProcessor(
            model=MistralToolModel(client=sdk),
            menu=load_menu(),
            session=session,
        ).process("Add fries")
        assert response == {"message": "I could not apply that call."}

    _with_sdk(handle, exercise)

    assert calls == 2
    assert session.lines == []
    assert session.revision == 0


def test_adapter_receives_only_complete_turns_retained_by_session_context_policy() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _completion(content="Ready for the next request.")

    session = Session()
    for number in range(13):
        session.transcript.append(TranscriptTurn((
            CustomerMessage(f"customer-{number}"),
            AssistantMessage(content=f"assistant-{number}"),
        )))

    def exercise(sdk: Mistral) -> None:
        TurnProcessor(
            model=MistralToolModel(client=sdk),
            menu=load_menu(),
            session=session,
        ).process("Current request")

    _with_sdk(handle, exercise)

    contents = [message.get("content") for message in captured[0]["messages"]]
    assert "customer-0" not in contents
    assert contents[1:5] == [
        "customer-1",
        "assistant-1",
        "customer-2",
        "assistant-2",
    ]
    assert contents[-3:] == ["customer-12", "assistant-12", "Current request"]


def test_controlled_mistral_tool_loop_applies_result_then_completes() -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return _completion(tool_calls=[{
                "id": "add-fries",
                "type": "function",
                "function": {
                    "name": "add_item",
                    "arguments": '{"item_id":"fries","quantity":1}',
                },
            }], finish_reason="tool_calls")
        return _completion(content="I added the fries.")

    session = Session()

    def exercise(sdk: Mistral) -> None:
        response = TurnProcessor(
            model=MistralToolModel(client=sdk),
            menu=load_menu(),
            session=session,
        ).process("Add fries")
        assert response["message"].startswith("I added the fries.")

    _with_sdk(handle, exercise)

    assert [(line.item_id, line.quantity) for line in session.lines] == [("fries", 1)]
    assert session.revision == 1
    assert requests[1]["messages"][-1]["tool_call_id"] == "add-fries"
    result = json.loads(requests[1]["messages"][-1]["content"])
    assert result["outcome"] == "APPLIED"
    assert result["draft"]["total_cents"] == 350
