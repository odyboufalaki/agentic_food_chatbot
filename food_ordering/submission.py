import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

import httpx
from jsonschema.validators import validator_for
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from referencing import Registry

from food_ordering.menu import money


ENDPOINT = "https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp"


@dataclass(frozen=True)
class SubmissionSettings:
    applicant_email: str = ""
    timeout_seconds: float = 20

    @classmethod
    def from_env(cls) -> "SubmissionSettings":
        return cls(applicant_email=os.getenv("APPLICANT_EMAIL", ""))


@dataclass(frozen=True)
class SubmissionResult:
    status: Literal["submitted", "rejected", "application_error", "uncertain", "not_sent"]
    invoked: bool
    result: dict[str, Any]


class Submitter(Protocol):
    def submit(self, payload: dict[str, Any]) -> SubmissionResult: ...


def render_receipt(result: dict[str, Any], reviewed_total_cents: int) -> str:
    details = ["Order accepted and submitted!"]
    details.append(f"Order number: {result.get('order_id', 'not provided')}")
    details.append(f"Reviewed total: {money(reviewed_total_cents)}")
    returned_total = result.get("total")
    if returned_total is None:
        details.append("Restaurant total: not provided")
    else:
        try:
            amount = Decimal(str(returned_total)) * 100
            returned_cents = int(amount) if amount.is_finite() and amount == amount.to_integral_value() else None
        except (InvalidOperation, ValueError, TypeError):
            returned_cents = None
        formatted = (f"${returned_total:.2f}"
                     if isinstance(returned_total, (int, float)) and not isinstance(returned_total, bool)
                     else str(returned_total))
        difference = " (differs from reviewed total)" if returned_cents != reviewed_total_cents else ""
        details.append(f"Restaurant total: {formatted}{difference}")
    if "estimated_time" in result:
        details.append(f"Estimated time: {result['estimated_time']}")
    return "\n".join(details)


def read_result(result: types.CallToolResult) -> SubmissionResult:
    is_error = result.isError
    payload = result.structuredContent
    text_blocks = [block.text for block in result.content if isinstance(block, types.TextContent)]
    if payload is None:
        for text in text_blocks:
            try:
                decoded = json.loads(text)
            except ValueError:
                continue
            if isinstance(decoded, dict):
                payload = decoded
                break
    if payload is not None and payload.get("success") is True:
        status: Literal["submitted", "uncertain"] = "uncertain" if is_error else "submitted"
        return SubmissionResult(status, True, payload)
    if is_error and (_contains_code(payload, -32602)
                     or any(re.search(r"(?<!\d)-32602(?!\d)", text) for text in text_blocks)):
        preserved = payload if payload is not None else {"code": -32602, "message": "\n".join(text_blocks)}
        return SubmissionResult("application_error", True, preserved)
    if payload is None:
        return SubmissionResult("uncertain", True, {
            "client_error": "unrecognized_result", "outcome": "uncertain",
            "text": "\n".join(text_blocks),
        })
    if payload.get("success") is False:
        return SubmissionResult("rejected", True, payload)
    return SubmissionResult("uncertain", True, payload)


def _contains_code(value: object, code: int) -> bool:
    if isinstance(value, dict):
        return value.get("code") == code or any(_contains_code(item, code) for item in value.values())
    if isinstance(value, list):
        return any(_contains_code(item, code) for item in value)
    return False


class MCPSubmitter:
    """One bounded session per call. The adapter owns and closes its HTTP transport."""

    def __init__(self, *, settings: SubmissionSettings | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings if settings is not None else SubmissionSettings.from_env()
        self._transport = transport

    def submit(self, payload: dict[str, Any], *,
               force_failure: Literal["kitchen_busy", "server_error"] | None = None) -> SubmissionResult:
        if (not self._settings.applicant_email.strip()
                or not math.isfinite(self._settings.timeout_seconds) or self._settings.timeout_seconds <= 0
                or force_failure not in {None, "kitchen_busy", "server_error"}):
            return SubmissionResult("not_sent", False, {"client_error": "configuration", "outcome": "not_sent"})
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._submit(payload, force_failure=force_failure))
        return SubmissionResult("not_sent", False, {"client_error": "use_worker_thread", "outcome": "not_sent"})

    async def _submit(self, payload: dict[str, Any], *,
                      force_failure: Literal["kitchen_busy", "server_error"] | None) -> SubmissionResult:
        invoked = False
        outcome = None

        async def track_request(request: httpx.Request) -> None:
            nonlocal invoked
            is_tool_call = request.method == "POST" and json.loads(request.content).get("method") == "tools/call"
            if is_tool_call:
                invoked = True
                if force_failure is not None:
                    request.headers["X-Demo-Force-Failure"] = force_failure

        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                async with httpx.AsyncClient(
                    headers={"X-Applicant-Email": self._settings.applicant_email},
                    timeout=httpx.Timeout(self._settings.timeout_seconds),
                    transport=self._transport, follow_redirects=False,
                    event_hooks={"request": [track_request]},
                ) as client:
                    async with streamable_http_client(ENDPOINT, http_client=client) as (read, write, _):
                        async with ClientSession(read, write, read_timeout_seconds=timedelta(
                            seconds=self._settings.timeout_seconds,
                        )) as session:
                            await session.initialize()
                            listed = await session.list_tools()
                            tool = next((tool for tool in listed.tools if tool.name == "submit_order"), None)
                            if tool is None:
                                raise ValueError("Missing submit_order tool")
                            validator = validator_for(tool.inputSchema)
                            validator.check_schema(tool.inputSchema)
                            validator(tool.inputSchema, registry=Registry()).validate(payload)
                            # The convenience call_tool validates outputSchema before returning,
                            # losing text-only or incomplete receipts. Normalize the raw SDK result.
                            result = await session.send_request(types.ClientRequest(types.CallToolRequest(
                                params=types.CallToolRequestParams(name="submit_order", arguments=payload),
                            )), types.CallToolResult)
                            outcome = read_result(result)
        except Exception:
            if outcome is None:
                status: Literal["uncertain", "not_sent"] = "uncertain" if invoked else "not_sent"
                outcome = SubmissionResult(status, invoked, {"client_error": "submission_failed", "outcome": status})
        assert outcome is not None
        return outcome
