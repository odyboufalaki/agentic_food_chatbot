# Food-ordering chatbot

Ticket 01: ask about the assignment menu and add fully specified selections to
an in-memory, priced draft through natural language. Python validates and prices
the order; Mistral interprets requests into typed proposals.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```sh
uv python install 3.13
uv sync --locked
export MISTRAL_API_KEY='your-studio-api-key'
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
print(agent.send("What milkshake flavors do you have?")["message"])
print(agent.send("Show my draft")["message"])
```

Every synchronous `send(str)` returns a dictionary containing `message`.
There are no restaurant calls in this slice, so responses omit `tool_calls`.
The CLI uses this same method; `quit`, `exit`, EOF, and Ctrl-C end the session.

## Demo

With valid Mistral configuration, enter:

```text
I'd like a large classic burger with cheese and bacon
```

The expected draft has a large beef Classic Burger with unique cheese and bacon
extras, total **$13.00**. Inspect the latest record in `logs/turns.jsonl`.

For the assignment's multiple-item price, use a fresh agent and ask for a medium
margherita with olives, large fries with parmesan, and a **large** cola: **$22.25**.
The original example reaches that amount after a cola edit; edits belong to
ticket 02, so this demo requests the final configuration directly.

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
  A message is fully validated before any addition is committed. Repeated extras
  charge once; separately added lines keep distinct stable IDs. Stored lines reserve
  an empty item-instruction field for later work.
- `interpretation.py`: direct synchronous Mistral SDK integration using
  [custom structured output](https://docs.mistral.ai/studio/conversations/structured-output/custom).
  It receives menu data, a fresh draft snapshot, and at most six recent turns.
  There is no pending clarification in this slice. SDK retries are explicitly
  disabled; each turn permits an initial request and one transient retry or one
  schema repair, with a 20-second timeout per request. Authentication/configuration
  failures are not retried. The
  [SDK source](https://github.com/mistralai/client-python) and installed 2.9.4 code
  were checked for the retry and structured-output interfaces.
- `turn_logging.py`: append-only local JSONL, isolated from the customer outcome.

The explicit aliases are `burger` → `classic_burger`, `cola` → `soda`, and
`pizza` → `margherita`. The interpreter also maps menu names and natural-language
option wording to menu IDs. Model-generated prices, prose, and submission claims
are never used as authoritative output.

External interpretation is injectable with `FoodOrderAgent(interpreter=...)`.
Its `interpret` method accepts keyword arguments `message`, `menu`, `draft`, and
`history`, returning a `Proposal` or an equivalent dictionary that Python validates.
To test the real adapter without network access, pass a Mistral SDK client with a
controlled HTTP transport to `MistralInterpreter(client=...)`. Injected clients
are owned by the caller; default clients are closed after each interpretation.

## Logging and privacy

Every send attempt writes session/turn IDs, timestamp, input, response, validated
operations, before/after totals in cents, draft revision/status transition, elapsed
milliseconds, and an error category. `tool_calls` is empty until submission is
implemented. Failed validation records no applied operations.

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
```

Tests use scripted interpretation or the real Mistral SDK with a controlled HTTP
transport. The default suite blocks network connections and removes live provider
configuration. It exercises pricing, atomic rejection, model request counts,
sanitized failure logging, and both required entry points without live credentials.
Scripted results do not establish live natural-language accuracy; no successful
live-model evaluation is included in this ticket.

## Current limitations

This slice adds items, answers menu questions, and displays drafts. Editing,
removal, grouping identical displayed lines, preparation instructions, pending
clarification continuation, confirmation, and MCP submission are later tickets.
If a required choice is missing, such as milkshake flavor, the whole message is
rejected with choices and a request to restate it completely. An over-$50 draft
stays open to further additions; later tickets add edits and enforce the limit
at submission. No food is ordered from a restaurant in this version.

Instances are in-memory and calls per instance must be serialized. Restarting
loses the draft. Menu data cannot establish ingredient or allergy guarantees.
Model credentials, account access, quota availability, and live interpretation
quality must be checked separately before a live demonstration.
