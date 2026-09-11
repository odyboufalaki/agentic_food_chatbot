# Food-ordering chatbot

Build, edit, review, and submit an in-memory food order through natural language.
Mistral translates customer requests into protocol-version-1 tool calls; Python
validates every operation, owns the authoritative Draft order, calculates prices,
and constructs the restaurant payload.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv python install 3.13
uv sync --locked
export MISTRAL_API_KEY='your-studio-api-key'
export APPLICANT_EMAIL='your-cv-email@example.com'
# Optional:
export MISTRAL_MODEL='mistral-small-latest'
export FOOD_ORDER_LOG_PATH='logs/turns.jsonl'
uv run --locked python main.py
```

The project requires Python 3.13 or newer. Environment variables are read when
the agent is constructed; `.env` files are not loaded automatically.

The programmatic API uses the same synchronous path as the CLI:

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()
print(agent.send("I'd like a large classic burger with cheese")["message"])
print(agent.send("Review my order")["message"])
result = agent.send("Yes")
print(result["message"])
```

Every `send(str)` result contains a nonempty `message`. Only an actual MCP
`submit_order` invocation adds `tool_calls`, with the exact arguments and
dictionary result.

## Conversation behavior

Operations emitted together are validated and applied in order. A valid operation
remains committed when an independent sibling is incomplete or invalid. For
example, asking for a Classic Burger, Spicy Jalapeño Burger, and flavorless
Milkshake adds both burgers and asks which Milkshake flavor is wanted.

There is no stored pending change. Tool calls, structured Outcomes, and customer
messages are retained in a bounded provider-neutral transcript. A later answer is
interpreted by a fresh model call, which must reconstruct a new complete operation.

Order lines have stable session-local IDs. Explicit serving edits can split a line;
identical lines group only for display. Replacements start from the destination
item's Menu defaults and do not inherit source options, extras, or instructions.

Checkout requires an exact Python-rendered Order review and explicit Confirmation
on a later customer turn. Any attempted edit invalidates eligibility. Python freezes
the reviewed restaurant payload and permits one invocation per authorized attempt.
Accepted, rejected, application-error, definitely-not-sent, and uncertain outcomes
remain distinct. An uncertain possible dispatch permanently blocks mutation and
resubmission in that Session.

## Architecture

- `agent.py` is the public synchronous facade. It owns one `Session`, creates a
  fresh `TurnProcessor` for every valid message, projects actual MCP attempts into
  the public response, and records exactly one JSONL turn.
- `session.py` owns persistent Draft, transcript, review, receipt, submission
  disposition, revision, and stable line identity.
- `turn_processor.py` owns the bounded per-turn model/tool/Outcome loop, commits
  valid Draft candidates independently, and enforces review and submission policy.
- `tool_protocol.py` is the strict version-1 contract used to generate advertised
  schemas and parse model calls. Unknown tools, extra fields, and invalid types are
  `MALFORMED` before deterministic handlers execute.
- `draft_operations.py` contains pure candidate validators and typed remedies.
- `mistral_adapter.py` translates application-owned messages and schemas to the
  Mistral tool-calling API. Provider objects never enter `Session`.
- `submission.py` makes one bounded MCP attempt using a fresh session, validates
  the advertised schema, includes `X-Applicant-Email`, and normalizes JSON or SSE
  results without automatic retry.
- `turn_logging.py` writes best-effort, append-only, redacted JSONL.

The model cannot mutate the Draft, price an order, construct the restaurant
payload, or claim a submission succeeded. Customer-visible deterministic review,
receipt, rejection, and uncertainty messages override model prose.

For deterministic tests, inject an application-owned `TurnModel`:

```python
agent = FoodOrderAgent(model=scripted_model, submitter=controlled_submitter)
```

`TurnModel.complete(messages=..., tools=...)` receives the bounded transcript and
versioned tool specifications and returns an `AssistantMessage`. Submission
remains independently injectable through `Submitter.submit(payload)`.

## Logging and privacy

Each JSONL record includes protocol version, session and turn IDs, customer input,
customer-visible response, ordered raw model calls, validated operations, parse
failures, structured Outcomes, commit effects, before/after totals and checkout
state, model-call usage, elapsed time, error category, and actual MCP disposition.

Logs exclude provider objects, HTTP headers, hidden reasoning, internal Draft
candidates, and exception text. The configured `MISTRAL_API_KEY` is redacted if
it appears in logged text. Logging failure emits a fixed sanitized stderr message
and never changes a known customer or restaurant outcome.

## Verification

```sh
uv run --locked mypy
uv run --locked pytest
```

The default suite is network-free. It uses scripted model responses and controlled
MCP transports. Live Mistral interpretation is opt-in and is not evidence supplied
by the deterministic suite.
