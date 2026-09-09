import asyncio
import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, Protocol

import httpx
from jsonschema.validators import validator_for
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from referencing import Registry


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
    status: Literal["submitted", "rejected", "uncertain", "not_sent"]
    invoked: bool
    result: dict[str, Any]


class Submitter(Protocol):
    def submit(self, payload: dict[str, Any]) -> SubmissionResult: ...


def render_receipt(result: dict[str, Any]) -> str:
    details = ["Order submitted!"]
    for label, key in [("Order number", "order_id"), ("Restaurant total", "total"),
                       ("Estimated time", "estimated_time")]:
        if key in result:
            value = result[key]
            if key == "total" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = f"${value:.2f}"
            details.append(f"{label}: {value}")
    return "\n".join(details)


def read_result(result: types.CallToolResult) -> SubmissionResult:
    is_error = result.isError
    payload = result.structuredContent
    if payload is None:
        for block in result.content:
            if isinstance(block, types.TextContent):
                try:
                    decoded = json.loads(block.text)
                except ValueError:
                    continue
                if isinstance(decoded, dict):
                    payload = decoded
                    break
    if payload is None:
        return SubmissionResult("uncertain", True, {"client_error": "unrecognized_result", "outcome": "uncertain"})
    if payload.get("success") is True and not is_error:
        return SubmissionResult("submitted", True, payload)
    if payload.get("success") is False:
        return SubmissionResult("rejected", True, payload)
    return SubmissionResult("uncertain", True, payload)


class MCPSubmitter:
    """One bounded session per call. The adapter owns and closes its HTTP transport."""

    def __init__(self, *, settings: SubmissionSettings | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings if settings is not None else SubmissionSettings.from_env()
        self._transport = transport

    def submit(self, payload: dict[str, Any]) -> SubmissionResult:
        if (not self._settings.applicant_email.strip()
                or not math.isfinite(self._settings.timeout_seconds) or self._settings.timeout_seconds <= 0):
            return SubmissionResult("not_sent", False, {"client_error": "configuration", "outcome": "not_sent"})
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._submit(payload))
        return SubmissionResult("not_sent", False, {"client_error": "use_worker_thread", "outcome": "not_sent"})

    async def _submit(self, payload: dict[str, Any]) -> SubmissionResult:
        invoked = False
        outcome = None

        async def track_request(request: httpx.Request) -> None:
            nonlocal invoked
            if request.method == "POST" and json.loads(request.content).get("method") == "tools/call":
                invoked = True

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
