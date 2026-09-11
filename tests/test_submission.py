import json
from decimal import Decimal

import httpx
import pytest

from food_ordering.menu import Menu
from food_ordering.order import InvalidSelection, OrderLine, submission_payload
from food_ordering.submission import MCPSubmitter, SubmissionSettings


INPUT_SCHEMA = {
    "type": "object",
    "required": ["items"],
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["item_id", "quantity"],
                "additionalProperties": False,
                "properties": {
                    "item_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "options": {"type": "object"},
                    "extras": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "special_instructions": {"type": "string"},
    },
}
RECEIPT = {
    "success": True,
    "order_id": "ORD-12345",
    "total": 8.5,
    "estimated_time": "15-20 minutes",
}


class RestaurantTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, sse: bool = False, text_only: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.calls: list[dict[str, object]] = []
        self.closed = 0
        self.sse = sse
        self.text_only = text_only
        self.receipt = dict(RECEIPT)
        self.is_error = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method in {"GET", "DELETE"}:
            return httpx.Response(405 if request.method == "GET" else 200)
        body = json.loads(request.content)
        method = body["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result: dict[str, object] = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "controlled-restaurant", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "submit_order",
                "inputSchema": INPUT_SCHEMA,
                "outputSchema": {"type": "object", "required": ["success"]},
            }]}
        elif method == "tools/call":
            self.calls.append(body["params"])
            result = {
                "isError": self.is_error,
                "content": [{"type": "text", "text": json.dumps(self.receipt)}],
            }
            if not self.text_only:
                result["structuredContent"] = self.receipt
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        response = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        headers = {"mcp-session-id": "test-session"}
        if self.sse and method == "tools/call":
            return httpx.Response(
                200,
                headers={**headers, "content-type": "text/event-stream"},
                content="event: message\ndata: " + json.dumps(response) + "\n\n",
            )
        return httpx.Response(200, headers=headers, json=response)

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.parametrize(
    ("sse", "text_only"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_mcp_adapter_submits_once_with_required_header_and_normalizes_result(
    sse: bool, text_only: bool,
) -> None:
    transport = RestaurantTransport(sse=sse, text_only=text_only)
    submitter = MCPSubmitter(
        settings=SubmissionSettings(applicant_email="applicant@example.test"),
        transport=transport,
    )

    outcome = submitter.submit({
        "items": [{"item_id": "classic_burger", "quantity": 1}],
    })

    assert outcome.status == "submitted"
    assert outcome.result == RECEIPT
    assert len(transport.calls) == 1
    assert transport.closed == 1
    assert all(
        request.headers["X-Applicant-Email"] == "applicant@example.test"
        for request in transport.requests
    )


def test_force_failure_header_is_scoped_to_one_attempt() -> None:
    class ForceAwareRestaurant(RestaurantTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if (
                request.method == "POST"
                and json.loads(request.content).get("method") == "tools/call"
            ):
                forced = request.headers.get("X-Demo-Force-Failure")
                self.receipt = (
                    {"success": False, "error": "Controlled busy response"}
                    if forced else dict(RECEIPT)
                )
            return await super().handle_async_request(request)

    transport = ForceAwareRestaurant()
    submitter = MCPSubmitter(
        settings=SubmissionSettings(applicant_email="applicant@example.test"),
        transport=transport,
    )
    payload = {"items": [{"item_id": "classic_burger", "quantity": 1}]}

    assert submitter.submit(payload, force_failure="kitchen_busy").status == "rejected"
    assert submitter.submit(payload).status == "submitted"
    calls = [
        request
        for request in transport.requests
        if request.method == "POST"
        and json.loads(request.content).get("method") == "tools/call"
    ]
    assert calls[0].headers["X-Demo-Force-Failure"] == "kitchen_busy"
    assert "X-Demo-Force-Failure" not in calls[1].headers


def test_submission_payload_rejects_fifty_dollars_and_one_cent() -> None:
    menu = Menu.model_validate({
        "menu": [{
            "id": "boundary_item",
            "name": "Boundary item",
            "base_price": Decimal("50.01"),
            "options": {},
        }],
    })
    line = OrderLine(
        item_id="boundary_item",
        name="Boundary item",
        quantity=1,
        options=(),
        extras=(),
        unit_cents=5001,
        line_id="L1",
    )

    with pytest.raises(InvalidSelection, match=r"\$50\.01"):
        submission_payload([line], menu)
