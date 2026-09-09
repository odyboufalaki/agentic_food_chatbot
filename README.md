# Food-ordering chatbot

Ask about the assignment menu and build, edit, or clear an in-memory, priced draft
through natural language. Python validates and prices the order; Mistral
interprets requests into typed proposals. Review a valid order, explicitly confirm
it, and receive a restaurant receipt through MCP (tickets 01–04).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```sh
uv python install 3.13
uv sync --locked
export MISTRAL_API_KEY='your-studio-api-key'
export APPLICANT_EMAIL='your-cv-email@example.com'
# Optional; the default is mistral-small-latest:
export MISTRAL_MODEL='mistral-small-latest'
# Optional; the default is logs/turns.jsonl:
export FOOD_ORDER_LOG_PATH='logs/turns.jsonl'
uv run --locked python main.py
```

The Python requirement is 3.13+, `.python-version` selects 3.13, and `uv.lock`
pins runtime and development dependencies. `uv sync --locked` installs pytest
and mypy as development dependencies. Environment variables are read when the
agent is constructed; `.env` files are not loaded automatically.

The direct assignment commands also work inside the prepared environment:

```sh
source .venv/bin/activate
python main.py
```

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()
response = agent.send("I'd like a large classic burger with cheese and bacon")
print(response["message"])  # Normalized draft, total $13.00
print(agent.send("Remove the bacon from my burger")["message"])  # $11.50
print(agent.send("What milkshake flavors do you have?")["message"])
print(agent.send("Show my draft")["message"])
print(agent.send("Submit")["message"])  # Review and confirmation question; no restaurant call
print(agent.send("Yes")["message"])  # One restaurant call and its receipt
```

Every synchronous `send(str)` returns a dictionary containing `message`.
An actual restaurant invocation also returns `tool_calls`, containing its `name`,
`arguments`, and dictionary `result`. Other turns omit that field, including
repeated approval after acceptance.
The CLI uses this same method; `quit`, `exit`, EOF, and Ctrl-C end the session.

## Demo

With valid Mistral configuration, enter:

```text
I'd like a large classic burger with cheese and bacon
```

The expected draft has a large beef Classic Burger with unique cheese and bacon
extras, total **$13.00**. Inspect the latest record in `logs/turns.jsonl`.

Continue with these requests, checking the updated draft and JSONL record each time:

| Request | Expected total |
| --- | --- |
| Remove the bacon from my burger | $11.50 |
| Make the burger regular size | $9.50 |
| Add cheese to the burger again | $9.50 |
| Make that six burgers | $57.00 |
| Remove two burgers | $38.00 |
| Two more of that burger | $57.00 |
| Remove the burger line | $0.00 |
| Add fries | $3.50 |
| Cancel my entire order | $0.00 |

For manual language evaluation, also try "take off the bacon", "make that two",
an invalid mixed request ("make the burger large and remove the milkshake" when
no shake is selected), and "remove the burger" with two separately added burger
lines. Invalid or ambiguous requests must leave the entire draft unchanged.
These are suggested live checks; automated tests use controlled model responses.

For the assignment's multiple-item price, use a fresh agent and ask for a medium
margherita with olives, large fries with parmesan, and a **large** cola: **$22.25**.
You can also request a default cola first and then ask to make it large.

To finish an order, say "That's it" or "Submit". The agent presents the full
normalized order and asks for confirmation. Reply "Yes" or "Submit" to send it.
A second "Yes" repeats the receipt without another invocation. After acceptance,
say "Start a new order with fries" to begin another draft explicitly.

Ordinary add/edit summaries and "Show my draft" are informational. Checkout
starts with a submit request. If approval includes an edit, the updated order
is reviewed again before it can be submitted. An invalid edit or failed model
interpretation also clears the previous approval eligibility. Informational menu
questions can retain an unchanged review.
A request such as "review for checkout" always displays a review and never counts
as approval, even if a prior review was eligible. The separate `review` operation
enforces this distinction; only `submit` or `confirm` can authorize an invocation.

## Design

- `agent.py`: one conversation, active draft, atomic turn coordination and public API.
- `food_ordering/menu.yaml`: the complete menu copied from the assignment.
  `menu.py` loads and validates it once per process. Decimal YAML parsing converts
  prices exactly to integer cents, including negative option modifiers.
- `proposals.py`: strict, discriminated action schemas. Unknown fields, malformed
  types, boolean/fractional/nonpositive quantities, and unknown actions are rejected.
  Python revalidates typed instances as well as dictionaries.
- `order.py`: validates item-specific options/extras, applies menu defaults,
  calculates totals, and renders normalized selections and menu answers.
  A message runs against a temporary draft in operation order and is fully validated
  before any change is committed. Repeated extras
  charge once; separately added lines keep distinct stable IDs. Stored lines reserve
  an empty item-instruction field for later work.
- Draft changes use separate `edit`, `set_quantity`, `increase_quantity`,
  `remove_units`, `remove_line`, and `clear_draft` operations. Targets match a
  current line ID or item ID with optional current options/extras. Exactly one
  line must match; nonexistent or ambiguous targets reject the whole message.
  Edits preserve surviving line IDs and reprice supported choices from the menu.
  A set quantity must be positive; zero remaining servings uses explicit line
  removal. Removing more servings than exist is rejected.
- `interpretation.py`: direct synchronous Mistral SDK integration using
  [custom structured output](https://docs.mistral.ai/studio/conversations/structured-output/custom).
  It receives menu data, a fresh draft snapshot, submission status, current and
  reviewed revisions, any pending clarification, and at most six recent turns.
  A pending proposal remains separate from the draft until a direct answer
  completes it; Python then revalidates and applies the entire proposal atomically.
  SDK retries are explicitly
  disabled; each turn permits an initial request and one transient retry or one
  schema repair, with a 20-second timeout per request. Authentication/configuration
  failures are not retried. The
  [SDK source](https://github.com/mistralai/client-python) and installed 2.9.4 code
  were checked for the retry and structured-output interfaces.
- `turn_logging.py`: append-only local JSONL, isolated from the customer outcome.
- `submission.py`: one synchronous adapter around the official
  [MCP Python SDK v1](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x),
  locked to 1.30.0. Each authorized attempt creates an HTTP client and MCP session,
  initializes, inspects `submit_order`, validates the payload against its advertised
  input schema, invokes once, and closes. `APPLICANT_EMAIL` supplies the required
  `X-Applicant-Email` header on every request to the assignment endpoint.
  A 20-second overall deadline and HTTP/session timeouts bound the attempt.
  Schema references resolve locally; they cannot fetch another URL.
  JSON and SSE are handled by the SDK's Streamable HTTP transport. The public
  `ClientSession.send_request` API retains raw tool results so text-only receipts
  are available even when a tool advertises an output schema. Structured content
  takes precedence, with JSON text as fallback. No tool invocation is retried.

Before submission, Python revalidates the menu selections and total, rejects an
empty order or a total above $50, and freezes the outgoing payload. Exactly $50
is permitted. Payloads contain normalized item IDs, positive quantities, options,
and unique extras. Internal line IDs and empty `special_instructions` are omitted.
Over-limit drafts remain editable, including a valid edit accompanied by a blocked
submit request.

The explicit aliases are `burger` → `classic_burger`, `cola` → `soda`, and
`pizza` → `margherita`. The interpreter also maps menu names and natural-language
option wording to menu IDs. Model-generated prices, prose, and submission claims
are never used as authoritative output.

External interpretation is injectable with `FoodOrderAgent(interpreter=...)`.
Its `interpret` method accepts keyword arguments `message`, `menu`, `draft`,
`history`, `order_state`, and `pending_clarification`, returning a `Proposal` or
an equivalent dictionary that Python validates.
To test the real adapter without network access, pass a Mistral SDK client with a
controlled HTTP transport to `MistralInterpreter(client=...)`. Injected clients
are owned by the caller; default clients are closed after each interpretation.

Submission is injectable with `FoodOrderAgent(submitter=...)`. The synchronous
`submit(payload)` boundary returns a `SubmissionResult` describing acceptance,
rejection, uncertainty, or failure before invocation. `MCPSubmitter(transport=...)`
accepts a controlled HTTPX async transport for tests; the adapter closes it.
`SubmissionSettings` can configure the applicant identity and timeout in code.
For the assignment's controlled failure demonstration only,
`MCPSubmitter.submit(payload, force_failure="kitchen_busy")` or `"server_error"`
adds `X-Demo-Force-Failure` to that tool-call request. It is never retained on
the client, sent during initialization, or inferred from customer language.

Async hosts must run `send` in a worker thread (for example,
`await asyncio.to_thread(agent.send, message)`) and serialize calls per agent.
The adapter uses `asyncio.run` for each attempt, with no persistent event loop or
connection service. It refuses direct execution inside an already running loop.

## Logging and privacy

Every send attempt writes session/turn IDs, timestamp, input, response, validated
operations, before/after totals in cents, current and reviewed revisions/status
transition, elapsed milliseconds, and an error category. `tool_calls` reports only
actual attempts and their arguments/results. Failed draft validation records no
applied operations; a blocked checkout can retain valid edits made in that turn.

Logs retain customer conversation text locally. They contain no SDK response
objects, HTTP headers, exception bodies, or hidden reasoning. The configured
`MISTRAL_API_KEY` is redacted if pasted into text. Logs, JSONL files, and `.env`
are ignored by Git. A write failure emits a fixed message to stderr without paths,
customer text, or credentials and leaves the conversation result unchanged.
Logging is best effort: there is no rotation or durable audit guarantee.

## Tests

```sh
uv run --locked pytest
uv run --locked mypy
# Focused behavioral or transport tests:
uv run --locked pytest tests/test_agent.py
uv run --locked pytest tests/test_mistral.py
uv run --locked pytest tests/test_submission.py
```

Tests use scripted interpretation or the real Mistral SDK with a controlled HTTP
transport, plus the real MCP client against controlled JSON/SSE HTTP responses.
The default suite blocks network connections and removes live provider
configuration. It exercises pricing, atomic rejection, model request counts,
sanitized failure logging, and both required entry points without live credentials.
Scripted results do not establish live natural-language accuracy; no successful
live-model evaluation or restaurant call is included in this ticket. Since the
supplied menu prices are multiples of 25 cents, the exact $50.01 rejection is
tested at the public deterministic submission validator using a boundary menu.

## Current limitations

This slice adds, edits, and removes selections, clears unsubmitted drafts,
answers menu questions, resolves missing choices and ambiguous changes, reviews,
and submits confirmed orders. Splitting some servings into another configuration,
changing multiple matching lines as a group, grouping identical displayed lines,
product replacement, and preparation instructions are later tickets.
If a required choice is missing, such as milkshake flavor, the whole proposal is
held until the customer answers or cancels it. An over-$50 draft
stays open to additions and edits, but cannot be submitted until within the limit.

Explicit rejection preserves the selections and server explanation without
claiming whether retry will help. It never triggers an automatic call. An explicit
"try again" authorizes one call against the same frozen payload. Editing after a
rejection returns the order to draft state, so retry intent presents a fresh review
and requires confirmation. A schema-validation response carrying JSON-RPC code
`-32602` is an application error and cannot retry unchanged.

The adapter inspects `isError` before interpreting content, prefers structured
content, and falls back to JSON text. `success: false` is an explicit rejection
with or without `isError`. Contradictory success/error flags and malformed results
are uncertain. A lost or unrecognized result may mean the order was accepted, so
uncertainty blocks resubmission and new-order reset within the session.

Clear acceptance remains accepted when receipt fields are absent or the returned
total differs. The receipt displays the reviewed total, any restaurant total and
the discrepancy; the locally reviewed price remains unchanged. Receipt details
already received survive cleanup or logging failures and no second call is made.

Instances are in-memory and calls per instance must be serialized. Restarting
loses the draft and duplicate-submission protection; there is no cross-restart
deduplication or server reconciliation. Menu data cannot establish ingredient or allergy guarantees.
Model credentials, account access, quota availability, and live interpretation
quality must be checked separately before a live demonstration.
