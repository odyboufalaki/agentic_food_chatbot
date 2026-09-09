import json
from decimal import Decimal

import httpx
import pytest

from agent import FoodOrderAgent
from food_ordering.interpretation import ModelFailure
from food_ordering.menu import Menu
from food_ordering.order import InvalidSelection, normalize, submission_payload
from food_ordering.proposals import Add
from food_ordering.submission import MCPSubmitter, SubmissionSettings
from test_agent import ScriptedInterpreter, add


INPUT_SCHEMA = {
    "type": "object", "required": ["items"], "additionalProperties": False,
    "properties": {
        "items": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["item_id", "quantity"], "additionalProperties": False,
            "properties": {
                "item_id": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1},
                "options": {"type": "object"}, "extras": {"type": "array", "items": {"type": "string"}},
            },
        }},
        "special_instructions": {"type": "string"},
    },
}
RECEIPT = {"success": True, "order_id": "ORD-12345", "total": 13.00, "estimated_time": "15-20 minutes"}


class RestaurantTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, sse=False, text_only=False):
        self.requests = []
        self.calls = []
        self.closed = 0
        self.sse = sse
        self.text_only = text_only
        self.receipt = dict(RECEIPT)
        self.schema = INPUT_SCHEMA
        self.is_error = False
        self.raw_text = None

    async def handle_async_request(self, request):
        self.requests.append(request)
        if request.method in {"GET", "DELETE"}:
            return httpx.Response(405 if request.method == "GET" else 200)
        body = json.loads(request.content)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "controlled-restaurant", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "submit_order", "inputSchema": self.schema,
                                 "outputSchema": {"type": "object", "required": ["success"]}}]}
        elif method == "tools/call":
            self.calls.append(body["params"])
            text = self.raw_text if self.raw_text is not None else json.dumps(self.receipt)
            result = {"isError": self.is_error, "content": [{"type": "text", "text": text}]}
            if not self.text_only:
                result["structuredContent"] = self.receipt
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        response = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        headers = {"mcp-session-id": "test-session"}
        if self.sse and method == "tools/call":
            return httpx.Response(200, headers={**headers, "content-type": "text/event-stream"},
                                  content="event: message\ndata: " + json.dumps(response) + "\n\n")
        return httpx.Response(200, headers=headers, json=response)

    async def aclose(self):
        self.closed += 1


def restaurant_agent(tmp_path, transport, *proposals):
    return FoodOrderAgent(
        interpreter=ScriptedInterpreter(*proposals),
        submitter=MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                               transport=transport),
        log_path=tmp_path / "turns.jsonl",
    )


def test_review_then_confirmation_submits_once_and_returns_receipt(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", options={"size": "large"}, extras=["cheese", "bacon"])]},
        {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("A large classic burger with cheese and bacon")
    review = agent.send("Submit my order")
    assert "Total: $13.00" in review["message"] and "confirm" in review["message"].lower()
    assert "Classic Burger" in review["message"] and "bacon" in review["message"]
    assert transport.calls == [] and not review.get("tool_calls")
    response = agent.send("Yes")
    assert "ORD-12345" in response["message"] and "15-20 minutes" in response["message"]
    assert "$13.00" in response["message"]
    expected = {"items": [{"item_id": "classic_burger", "quantity": 1,
                           "options": {"size": "large", "patty": "beef"}, "extras": ["bacon", "cheese"]}]}
    assert transport.calls == [{"name": "submit_order", "arguments": expected}]
    assert response["tool_calls"] == [{"name": "submit_order", "arguments": expected, "result": RECEIPT}]
    repeated = agent.send("Yes again")
    assert repeated["message"] == response["message"] and not repeated.get("tool_calls")
    assert len(transport.calls) == 1
    assert transport.closed == 1


def test_accepted_order_requires_explicit_new_order_before_adding_food(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]},
        {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
        {"operations": [add("fries")]},
        {"operations": [{"type": "clear_draft"}]},
        {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "new_order"}, add("fries")]},
        {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    receipt = agent.send("Yes")
    assert "new order" in agent.send("Add fries")["message"].lower()
    assert "new order" in agent.send("Cancel my order")["message"].lower()
    assert agent.send("Yes")["message"] == receipt["message"]
    new = agent.send("Start a new order with fries")
    assert "Total: $3.50" in new["message"] and "Burger" not in new["message"]
    agent.send("Submit")
    assert len(transport.calls) == 1
    agent.send("Yes")
    assert len(transport.calls) == 2
    assert transport.calls[1]["arguments"] == {"items": [
        {"item_id": "fries", "quantity": 1, "options": {"size": "medium"}, "extras": []},
    ]}


def test_approval_plus_edit_requires_confirmation_of_the_updated_review(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", options={"size": "large"}, extras=["cheese", "bacon"])]},
        {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}, {"type": "edit", "target": {"item_id": "burger"}, "remove_extras": ["bacon"]}]},
        {"operations": [{"type": "submit"}]},
    )
    agent.send("A large burger with cheese and bacon")
    agent.send("That's it")
    updated = agent.send("Yes, but remove the bacon")
    assert "Total: $11.50" in updated["message"] and "confirm" in updated["message"].lower()
    assert transport.calls == []
    agent.send("Submit")
    assert len(transport.calls) == 1
    assert transport.calls[0]["arguments"]["items"][0]["extras"] == ["cheese"]


@pytest.mark.parametrize("invalid", [
    {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "options": {"size": "giant"}}]},
    {"operations": [{"type": "unsupported", "reason": "unclear"}]},
    {"operations": [{"type": "remove_units", "target": {"item_id": "burger"}, "quantity": True}]},
    {"operations": [{"type": "confirm"}, add("milkshake", options={"flavor": "vanilla bean"})]},
])
def test_invalid_or_ambiguous_edit_invalidates_review_even_when_draft_unchanged(tmp_path, invalid):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]}, invalid,
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    assert "unchanged" in agent.send("Change my order")["message"]
    review = agent.send("Yes")
    assert "Total: $8.50" in review["message"]
    assert transport.calls == []
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_informational_question_retains_review_eligibility(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "menu", "item_ids": ["milkshake"]}]},
        {"operations": [{"type": "confirm"}, {"type": "submit"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    assert "vanilla" in agent.send("What milkshake flavors are available?")["message"]
    agent.send("Yes, submit")
    assert len(transport.calls) == 1


@pytest.mark.parametrize("sse,text_only", [(False, False), (True, False), (False, True), (True, True)])
def test_real_adapter_initializes_with_applicant_headers_and_reads_json_or_sse(tmp_path, sse, text_only):
    transport = RestaurantTransport(sse=sse, text_only=text_only)
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert response["tool_calls"][0]["result"] == RECEIPT
    assert transport.closed == 1
    assert any(request.method == "DELETE" for request in transport.requests)
    assert all(request.headers["X-Applicant-Email"] == "applicant@example.test" for request in transport.requests)
    methods = [json.loads(request.content)["method"] for request in transport.requests if request.method == "POST"]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[0]["tool_calls"] == records[1]["tool_calls"] == []
    assert records[2]["tool_calls"] == response["tool_calls"]
    assert records[2]["state_transition"]["before"]["status"] == "draft"
    assert records[2]["state_transition"]["after"]["status"] == "submitted"
    assert "applicant@example.test" not in (tmp_path / "turns.jsonl").read_text()


def test_exactly_fifty_dollars_can_be_submitted(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("soda", quantity=25)]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("Twenty-five colas")
    assert "Total: $50.00" in agent.send("Submit")["message"]
    agent.send("Yes")
    assert transport.calls[0]["arguments"]["items"][0]["quantity"] == 25


def test_empty_or_incomplete_order_never_reaches_restaurant(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{"type": "submit"}]},
        {"operations": [add("milkshake")]},
        {"operations": [{"type": "confirm"}]},
    )
    assert "empty" in agent.send("Submit")["message"]
    assert "flavor" in agent.send("Add a milkshake")["message"]
    assert "flavor" in agent.send("Yes")["message"]
    assert transport.requests == []


def test_mixed_add_and_submit_above_limit_keeps_editable_draft_and_reports_it_honestly(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("soda", quantity=25, extras=[]), add("milkshake", options={"flavor": "oreo"}, extras=["cherry_on_top"]), {"type": "submit"}]},
        {"operations": [{"type": "summary"}]},
        {"operations": [{"type": "confirm"}]},
    )
    response = agent.send("Twenty-five colas and an Oreo shake with a cherry, submit")
    assert "$55.75" in response["message"] and "$50.00" in response["message"]
    assert "unchanged" not in response["message"]
    assert "Total: $55.75" in agent.send("Show draft")["message"]
    agent.send("Yes")
    assert transport.requests == []


@pytest.mark.parametrize("failure", [ModelFailure("model_timeout"), RuntimeError("private integration detail")])
def test_failed_interpretation_requires_a_fresh_review(tmp_path, failure):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]}, failure,
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    assert "unchanged" in agent.send("Change the burger")["message"]
    assert "confirm" in agent.send("Yes")["message"].lower()
    assert transport.calls == []
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_deterministic_submission_boundary_rejects_fifty_dollars_and_one_cent():
    # The assignment menu only has multiples of 25 cents. Exercise the same
    # public validator with a one-cent menu boundary without changing that menu.
    menu = Menu.model_validate({"menu": [{"id": "boundary_item", "name": "Boundary item",
                                        "base_price": Decimal("50.01"), "options": {}}]})
    line = normalize(Add(type="add", item_id="boundary_item", quantity=1), menu)
    with pytest.raises(InvalidSelection, match=r"\$50\.01"):
        submission_payload([line], menu)


def test_above_limit_order_can_be_edited_then_reviewed_and_submitted(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("soda", quantity=24), add("milkshake", options={"flavor": "oreo"}, extras=["cherry_on_top"])]},
        {"operations": [{"type": "submit"}]}, {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "milkshake"}}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("Twenty-four colas and an Oreo shake with a cherry")
    assert "$50.00" in agent.send("Submit")["message"]
    agent.send("Yes")
    assert transport.requests == []
    agent.send("Remove the shake")
    assert "Total: $48.00" in agent.send("Yes")["message"]
    assert transport.calls == []
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_assignment_modification_conversation_submits_exact_final_selections(tmp_path):
    transport = RestaurantTransport()
    transport.receipt = {"success": True, "order_id": "ORD-55102", "total": 22.25}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("margherita", options={"size": "medium"}, extras=["olives"]),
                        add("fries", options={"size": "large"}, extras=["parmesan"]), add("cola")]},
        {"operations": [{"type": "edit", "target": {"item_id": "cola"}, "options": {"size": "large"}}]},
        {"operations": [{"type": "submit"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A medium margherita with olives, large fries with parmesan, and a cola")
    assert "Total: $22.25" in agent.send("Make the cola large")["message"]
    assert "Total: $22.25" in agent.send("Submit")["message"]
    assert transport.calls == []
    assert "ORD-55102" in agent.send("Yes")["message"]
    assert transport.calls == [{"name": "submit_order", "arguments": {"items": [
        {"item_id": "margherita", "quantity": 1, "options": {"size": "medium", "crust": "regular"}, "extras": ["olives"]},
        {"item_id": "fries", "quantity": 1, "options": {"size": "large"}, "extras": ["parmesan"]},
        {"item_id": "soda", "quantity": 1, "options": {"size": "large", "flavor": "cola"}, "extras": []},
    ]}}]


def test_advertised_input_schema_is_validated_before_invocation(tmp_path):
    transport = RestaurantTransport()
    transport.schema = {"type": "object", "required": ["unavailable_field"]}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert "not sent" in response["message"] and not response.get("tool_calls")
    assert transport.calls == [] and transport.closed == 1


def test_lost_submission_response_blocks_confirmation_and_new_order(tmp_path):
    class LostResponse(RestaurantTransport):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            if request.method == "POST" and json.loads(request.content)["method"] == "tools/call":
                raise httpx.ReadError("private connection detail", request=request)
            return response

    transport = LostResponse()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "new_order"}, add("fries")]},
        {"operations": [{"type": "submit"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    failed = agent.send("Yes")
    assert "uncertain" in failed["message"]
    assert failed["tool_calls"][0]["result"] == {"client_error": "submission_failed", "outcome": "uncertain"}
    agent.send("Yes again")
    agent.send("Start a new order with fries")
    agent.send("Submit")
    assert len(transport.calls) == 1 and transport.closed == 1


def test_receipt_survives_transport_cleanup_failure(tmp_path):
    class CleanupFailure(RestaurantTransport):
        async def aclose(self):
            await super().aclose()
            raise RuntimeError("private cleanup detail")

    transport = CleanupFailure()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert "ORD-12345" in response["message"]
    assert agent.send("Yes again")["message"] == response["message"]
    assert len(transport.calls) == 1


def test_unexpected_submitter_failure_cannot_authorize_another_attempt(tmp_path):
    class BrokenSubmitter:
        def __init__(self):
            self.calls = 0

        def submit(self, payload):
            self.calls += 1
            raise RuntimeError("Unknown integration failure after possible dispatch")

    submitter = BrokenSubmitter()
    agent = FoodOrderAgent(submitter=submitter, interpreter=ScriptedInterpreter(
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "confirm"}]},
    ), log_path=tmp_path / "turns.jsonl")
    agent.send("A burger")
    agent.send("Submit")
    assert "uncertain" in agent.send("Yes")["message"]
    agent.send("Yes")
    agent.send("Yes")
    assert submitter.calls == 1


def test_cli_submission_reports_the_same_receipt_and_tool_record(tmp_path):
    from io import StringIO
    from main import main

    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    output = StringIO()
    main(agent=agent, input_stream=StringIO("A burger\nSubmit\nYes\nYes\nquit\n"), output_stream=output)
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert len(transport.calls) == 1
    assert records[2]["response"]["message"] in output.getvalue()
    assert records[2]["tool_calls"] == records[2]["response"]["tool_calls"]
    assert records[3]["tool_calls"] == []
    assert records[1]["state_transition"]["after"]["reviewed_revision"] == 1


def test_submission_timeout_is_bounded_and_never_reinvokes(tmp_path):
    import asyncio

    class SlowRestaurant(RestaurantTransport):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            if request.method == "POST" and json.loads(request.content)["method"] == "tools/call":
                await asyncio.sleep(10)
            return response

    transport = SlowRestaurant()
    agent = FoodOrderAgent(
        submitter=MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test", timeout_seconds=0.1),
                               transport=transport),
        interpreter=ScriptedInterpreter(
            {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
            {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
        ), log_path=tmp_path / "turns.jsonl",
    )
    agent.send("A burger")
    agent.send("Submit")
    assert "uncertain" in agent.send("Yes")["message"]
    agent.send("Yes")
    assert len(transport.calls) == 1 and transport.closed == 1


def test_missing_applicant_configuration_never_opens_a_connection(tmp_path):
    transport = RestaurantTransport()
    agent = FoodOrderAgent(
        submitter=MCPSubmitter(transport=transport),
        interpreter=ScriptedInterpreter(
            {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
            {"operations": [{"type": "confirm"}]},
        ), log_path=tmp_path / "turns.jsonl",
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert "configuration" in response["message"] and not response.get("tool_calls")
    assert transport.requests == []


def test_repeated_checkout_review_never_counts_as_approval(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "review"}]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    first = agent.send("That's it")
    assert "confirm" in first["message"].lower()
    assert agent.send("Review for checkout again") == first
    assert transport.requests == []
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_accepted_order_summary_shows_receipt_and_selections_as_submitted(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "summary"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    agent.send("Yes")
    summary = agent.send("Show my order")["message"]
    assert "ORD-12345" in summary and "Classic Burger" in summary
    assert "Draft order" not in summary
    assert len(transport.calls) == 1


def test_unclassified_rejection_retries_only_when_customer_requests_it(tmp_path):
    transport = RestaurantTransport()
    outcomes = iter([
        {"success": False, "error": "Restaurant could not accept the order."},
        {"success": True, "order_id": "ORD-RETRIED", "estimated_time": "20 minutes"},
    ])

    class SequencedInterpreter(ScriptedInterpreter):
        def interpret(self, **context):
            proposal = super().interpret(**context)
            if proposal == "advance_receipt":
                transport.receipt = next(outcomes)
                return {"operations": [{"type": "confirm"}]}
            return proposal

    agent = FoodOrderAgent(
        interpreter=SequencedInterpreter(
            {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
            "advance_receipt", {"operations": [{"type": "summary"}]},
            {"operations": [{"type": "retry_submission"}]},
        ), submitter=MCPSubmitter(
            settings=SubmissionSettings(applicant_email="applicant@example.test"), transport=transport,
        ), log_path=tmp_path / "turns.jsonl",
    )
    agent.send("A burger")
    agent.send("Submit")
    rejected = agent.send("Yes")
    assert "Restaurant could not accept" in rejected["message"]
    assert "retry" in rejected["message"].lower()
    assert len(transport.calls) == 1
    assert "Classic Burger" in agent.send("Show my order")["message"]
    assert len(transport.calls) == 1
    transport.receipt = next(outcomes)
    retried = agent.send("Please try submitting again")
    assert "ORD-RETRIED" in retried["message"]
    assert len(transport.calls) == 2
    assert transport.calls[0]["arguments"] == transport.calls[1]["arguments"]
    methods = [json.loads(request.content)["method"] for request in transport.requests if request.method == "POST"]
    assert methods.count("initialize") == methods.count("tools/call") == 2
    assert all(request.headers["X-Applicant-Email"] == "applicant@example.test" for request in transport.requests)
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[2]["state_transition"]["after"]["status"] == "rejected"
    assert records[2]["tool_calls"][0]["result"]["success"] is False
    assert records[3]["tool_calls"] == []
    assert records[4]["operations"] == [{"type": "retry_submission"}]
    assert records[4]["state_transition"]["before"]["status"] == "rejected"
    assert records[4]["state_transition"]["after"]["status"] == "submitted"


@pytest.mark.parametrize("is_error", [False, True])
def test_success_false_is_an_explicit_rejection_even_when_is_error(tmp_path, is_error):
    transport = RestaurantTransport()
    transport.receipt = {"success": False, "error": "Controlled refusal"}
    transport.is_error = is_error
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert "rejected" in response["message"] and "Controlled refusal" in response["message"]
    assert response["tool_calls"][0]["result"] == transport.receipt
    assert json.loads((tmp_path / "turns.jsonl").read_text().splitlines()[-1])["error_category"] == "submission_rejected"


def test_schema_error_from_text_is_not_retryable_unchanged(tmp_path):
    transport = RestaurantTransport(text_only=True)
    transport.is_error = True
    transport.raw_text = json.dumps({"jsonrpc": "2.0", "error": {
        "code": -32602, "message": "Controlled invalid arguments",
    }})
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "retry_submission"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    failed = agent.send("Yes")
    assert "application error" in failed["message"].lower()
    assert failed["tool_calls"][0]["result"]["error"]["code"] == -32602
    assert "cannot retry" in agent.send("Try again")["message"].lower()
    assert len(transport.calls) == 1


def test_schema_error_can_be_corrected_then_reviewed_before_a_new_attempt(tmp_path):
    transport = RestaurantTransport(text_only=True)
    transport.is_error = True
    transport.raw_text = "JSON-RPC request failed with code -32602"
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", extras=["cheese"])]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "remove_extras": ["cheese"]}]},
        {"operations": [{"type": "retry_submission"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A cheeseburger")
    agent.send("Submit")
    agent.send("Yes")
    agent.send("Remove cheese")
    transport.text_only = False
    transport.is_error = False
    transport.raw_text = None
    transport.receipt = RECEIPT
    review = agent.send("Try again")
    assert "confirm" in review["message"].lower()
    assert len(transport.calls) == 1
    agent.send("Yes")
    assert len(transport.calls) == 2


def test_edit_after_rejection_requires_new_review_and_confirmation(tmp_path):
    transport = RestaurantTransport()
    transport.receipt = {"success": False, "error": "Controlled refusal"}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", extras=["cheese"])]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]},
        {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "remove_extras": ["cheese"]}]},
        {"operations": [{"type": "retry_submission"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A cheeseburger")
    agent.send("Submit")
    agent.send("Yes")
    agent.send("Remove cheese")
    transport.receipt = RECEIPT
    review = agent.send("Try submission again")
    assert "confirm" in review["message"].lower() and "cheese" not in review["message"]
    assert len(transport.calls) == 1
    agent.send("Yes")
    assert len(transport.calls) == 2
    assert transport.calls[1]["arguments"]["items"][0]["extras"] == []


@pytest.mark.parametrize("payload,is_error", [
    ({"success": True, "order_id": "ORD-CONTRADICTORY"}, True),
    ({"success": True, "order_id": "ORD-CONTRADICTORY", "error": {"code": -32602}}, True),
    ({}, False),
    ({"order_id": "ORD-WITHOUT-SUCCESS"}, False),
])
def test_contradictory_or_malformed_result_is_uncertain_and_blocks_all_resubmission(tmp_path, payload, is_error):
    transport = RestaurantTransport()
    transport.receipt = payload
    transport.is_error = is_error
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "retry_submission"}]},
        {"operations": [{"type": "new_order"}, add("fries")]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    response = agent.send("Yes")
    assert "uncertain" in response["message"]
    assert response["tool_calls"][0]["result"] == payload
    agent.send("Retry")
    agent.send("Start a new order with fries")
    agent.send("Yes")
    assert len(transport.calls) == 1


def test_accepted_incomplete_receipt_reports_missing_details_without_reinvoking(tmp_path):
    transport = RestaurantTransport()
    transport.receipt = {"success": True}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    receipt = agent.send("Yes")
    assert "accepted" in receipt["message"].lower()
    assert "order number" in receipt["message"].lower() and "not provided" in receipt["message"].lower()
    assert "Reviewed total: $8.50" in receipt["message"] and "restaurant total" in receipt["message"].lower()
    assert agent.send("Yes again")["message"] == receipt["message"]
    assert len(transport.calls) == 1


def test_accepted_different_total_exposes_both_amounts_and_preserves_reviewed_price(tmp_path):
    transport = RestaurantTransport()
    transport.receipt = {"success": True, "order_id": "ORD-DIFFERENT", "total": 9.25}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "summary"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    receipt = agent.send("Yes")
    assert "Reviewed total: $8.50" in receipt["message"]
    assert "Restaurant total: $9.25" in receipt["message"] and "differs" in receipt["message"]
    summary = agent.send("Show my submitted order")["message"]
    assert "Total: $8.50" in summary and "$9.25" in summary
    assert len(transport.calls) == 1


def test_force_failure_header_is_scoped_to_one_tool_call_and_never_drives_classification():
    class ForceAwareRestaurant(RestaurantTransport):
        async def handle_async_request(self, request):
            if request.method == "POST" and json.loads(request.content).get("method") == "tools/call":
                forced = request.headers.get("X-Demo-Force-Failure")
                self.receipt = ({"success": False, "error": "Controlled busy response"}
                                if forced else RECEIPT)
                self.is_error = forced == "server_error"
            return await super().handle_async_request(request)

    transport = ForceAwareRestaurant()
    submitter = MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                             transport=transport)
    payload = {"items": [{"item_id": "classic_burger", "quantity": 1}]}
    forced = submitter.submit(payload, force_failure="kitchen_busy")
    normal = submitter.submit(payload)
    assert forced.status == "rejected" and normal.status == "submitted"
    calls = [request for request in transport.requests
             if request.method == "POST" and json.loads(request.content).get("method") == "tools/call"]
    assert calls[0].headers["X-Demo-Force-Failure"] == "kitchen_busy"
    assert "X-Demo-Force-Failure" not in calls[1].headers
    assert all("X-Demo-Force-Failure" not in request.headers for request in transport.requests if request not in calls)


def test_server_error_forced_shape_is_still_an_explicit_rejection():
    transport = RestaurantTransport()
    transport.receipt = {"success": False, "error": "Controlled server failure"}
    transport.is_error = True
    submitter = MCPSubmitter(settings=SubmissionSettings(applicant_email="applicant@example.test"),
                             transport=transport)
    result = submitter.submit({"items": [{"item_id": "classic_burger", "quantity": 1}]},
                              force_failure="server_error")
    assert result.status == "rejected"
    call = next(request for request in transport.requests
                if request.method == "POST" and json.loads(request.content).get("method") == "tools/call")
    assert call.headers["X-Demo-Force-Failure"] == "server_error"


def test_logging_failure_after_acceptance_preserves_receipt_and_prevents_duplicate(tmp_path, capsys):
    transport = RestaurantTransport()
    failed_path = tmp_path / "log-target-is-a-directory"
    failed_path.mkdir()
    agent = FoodOrderAgent(
        interpreter=ScriptedInterpreter(
            {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
            {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
        ), submitter=MCPSubmitter(
            settings=SubmissionSettings(applicant_email="applicant@example.test"), transport=transport,
        ), log_path=failed_path,
    )
    agent.send("A burger")
    agent.send("Submit")
    receipt = agent.send("Yes")
    repeated = agent.send("Yes again")
    assert "ORD-12345" in receipt["message"] and repeated["message"] == receipt["message"]
    assert len(transport.calls) == 1
    assert "logging failed" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("noop", [
    {"type": "set_quantity", "target": {"item_id": "burger"}, "quantity": 1},
    {"type": "edit", "target": {"item_id": "burger"}, "add_extras": ["cheese"]},
])
def test_noop_edit_cannot_resubmit_a_schema_rejected_payload(tmp_path, noop):
    transport = RestaurantTransport(text_only=True)
    transport.is_error = True
    transport.raw_text = '{"error":{"code":-32602,"message":"invalid"}}'
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", extras=["cheese"])]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [noop]},
        {"operations": [{"type": "retry_submission"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A cheeseburger")
    agent.send("Submit")
    agent.send("Yes")
    agent.send("Make no effective change")
    agent.send("Try again")
    blocked = agent.send("Yes")
    assert "unchanged" in blocked["message"] and "application error" in blocked["message"]
    assert len(transport.calls) == 1


@pytest.mark.parametrize("failed_change", [
    {"operations": [{"type": "unsupported", "reason": "unclear"}]},
    {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "options": {"size": "giant"}}]},
    ModelFailure("model_timeout"),
])
def test_failed_change_after_rejection_requires_review_before_retry(tmp_path, failed_change):
    transport = RestaurantTransport()
    transport.receipt = {"success": False, "error": "Controlled refusal"}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]}, {"operations": [{"type": "submit"}]},
        {"operations": [{"type": "confirm"}]}, failed_change,
        {"operations": [{"type": "retry_submission"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    agent.send("Submit")
    agent.send("Yes")
    agent.send("Change the burger")
    transport.receipt = RECEIPT
    review = agent.send("Try again")
    assert "confirm" in review["message"].lower()
    assert len(transport.calls) == 1
    agent.send("Yes")
    assert len(transport.calls) == 2
