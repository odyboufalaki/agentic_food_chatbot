import json

import httpx
import pytest
from mistralai.client import Mistral

from agent import FoodOrderAgent
from food_ordering.interpretation import MistralInterpreter, Settings
from food_ordering.proposals import Proposal


def completion(content):
    return httpx.Response(200, json={
        "id": "test-completion", "object": "chat.completion", "created": 0,
        "model": "mistral-small-latest",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_mistral_sends_schema_and_context_and_returns_a_typed_proposal():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return completion('{"operations":[{"type":"summary"}]}')

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with Mistral(api_key="test-key", client=http_client) as sdk:
            interpreter = MistralInterpreter(client=sdk)
            proposal = interpreter.interpret(
                message="What is in my draft?", menu={"items": []},
                draft=[{"item_id": "soda", "quantity": 1}],
                history=[{"role": "user", "content": "A cola"}],
            )
    assert isinstance(proposal, Proposal)
    assert proposal.operations[0].type == "summary"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "mistral-small-latest"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert request["messages"][-1]["content"] == "What is in my draft?"
    context = request["messages"][0]["content"]
    assert '"item_id": "soda"' in context
    assert '"pending_clarification": null' in context
    assert request["messages"][-2]["content"] == "A cola"


@pytest.mark.parametrize("failures,expected_calls", [
    ([429, 429], 2),
    ([503, 503], 2),
    (["timeout", "timeout"], 2),
    (["invalid", "invalid"], 2),
    (["timeout", "invalid"], 2),
    (["invalid", 429], 2),
    ([401], 1),
    ([403], 1),
    ([400], 1),
    ([422], 1),
])
def test_failed_turn_preserves_draft_and_caps_actual_http_requests(tmp_path, failures, expected_calls):
    outcomes = iter(["add", *failures, "summary"])
    calls = []

    def handle(request):
        calls.append(request)
        outcome = next(outcomes)
        if outcome == "timeout":
            raise httpx.ReadTimeout("private server detail", request=request)
        if outcome == "invalid":
            return completion('{"operations":[{"type":"add","item_id":"fries","quantity":true}]}')
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"message": "private server detail"})
        if outcome == "add":
            return completion('{"operations":[{"type":"add","item_id":"soda","quantity":1}]}')
        return completion('{"operations":[{"type":"summary"}]}')

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with Mistral(api_key="test-key", client=http_client) as sdk:
            agent = FoodOrderAgent(interpreter=MistralInterpreter(client=sdk), log_path=tmp_path / "turns.jsonl")
            initial = agent.send("A cola")
            failed = agent.send("Add fries")
            assert len(calls) == 1 + expected_calls
            assert "unchanged" in failed["message"]
            assert "private server detail" not in failed["message"]
            assert agent.send("Show draft") == initial
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[1]["error_category"] is not None
    assert records[1]["operations"] == []
    assert records[1]["totals"] == {"before_cents": 200, "after_cents": 200}
    assert "private server detail" not in (tmp_path / "turns.jsonl").read_text()


@pytest.mark.parametrize("first_failure", [429, "timeout", "invalid"])
def test_one_recovery_adds_once(tmp_path, first_failure):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if first_failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if first_failure == "invalid":
                return completion("not JSON")
            return httpx.Response(first_failure, json={"message": "rate limit"})
        return completion('{"operations":[{"type":"add","item_id":"fries","quantity":1}]}')

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with Mistral(api_key="test-key", client=http_client) as sdk:
            agent = FoodOrderAgent(interpreter=MistralInterpreter(client=sdk), log_path=tmp_path / "turns.jsonl")
            response = agent.send("Fries")
    assert len(requests) == 2
    assert "Total: $3.50" in response["message"]
    if first_failure == "invalid":
        assert "schema" in requests[1]["messages"][-1]["content"]


def test_missing_credentials_is_a_customer_response_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = FoodOrderAgent(log_path=tmp_path / "turns.jsonl")
    response = agent.send("A burger")
    assert "configuration" in response["message"]
    assert "unchanged" in response["message"]


def test_empty_model_does_not_make_a_request(tmp_path):
    def handle(request):
        pytest.fail("Invalid configuration must not make an HTTP request")

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with Mistral(api_key="test-key", client=http_client) as sdk:
            agent = FoodOrderAgent(
                interpreter=MistralInterpreter(client=sdk, settings=Settings(model="")),
                log_path=tmp_path / "turns.jsonl",
            )
            assert "configuration" in agent.send("A burger")["message"]
