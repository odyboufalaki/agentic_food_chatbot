# Food-ordering chatbot

A natural-language food ordering agent built with Mistral and MCP.

Mistral interprets customer requests and turns them into structured tool calls.  
The Python application owns the actual order state, validates menu choices, applies edits, calculates prices, handles review and confirmation, and submits the final order to the restaurant MCP server.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv python install 3.13
uv sync --locked

export MISTRAL_API_KEY='your-studio-api-key'
export APPLICANT_EMAIL='your-cv-email@example.com'

# Optional
export MISTRAL_MODEL='mistral-small-latest'
export FOOD_ORDER_LOG_PATH='logs/turns.jsonl'

uv run --locked python main.py
```

Python 3.13+ is required.

Environment variables are read when the agent is created. `.env` files are not loaded automatically.

## Programmatic usage

The same agent can also be used directly from Python:

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()

print(
    agent.send(
        "I'd like a large classic burger with cheese"
    )["message"]
)

print(agent.send("Review my order")["message"])

result = agent.send("Yes")
print(result["message"])
```

Every call to `send()` returns a dictionary containing a non-empty `message`.

If an MCP `submit_order` call was actually made, the response also includes the submitted arguments and returned result under `tool_calls`.

## Conversation behavior

The model is responsible for understanding the customer's language, but it does not directly modify the order.

Instead, it emits structured operations such as:

- `add_item`
- `update_item`
- `change_quantity`
- `remove_item`
- `show_menu`
- `show_draft`
- `propose_submission`
- `submit_order`

These operations are parsed and validated by Python before they can affect the draft.

Operations emitted in the same model response are handled independently. If one operation is incomplete or invalid, other valid operations can still be applied.

For example, if the customer asks for two valid burgers and a milkshake without specifying a required flavor, the burgers can be added while the customer is asked to choose the milkshake flavor.

Order lines have stable IDs within a session. Editing only some servings of a line may split it into separate lines, while equivalent lines are grouped again when the draft is displayed.

Before submission, the current order is rendered for review. Confirmation must arrive on a later customer turn. The restaurant payload created at review time is frozen and reused for submission so that the submitted order is exactly the one the customer reviewed.

Submission failures are handled conservatively. In particular, if the application cannot determine whether the restaurant received the request, the session is marked as uncertain and further submission attempts are blocked to avoid duplicate orders.

## Architecture

- `main.py` — command-line interface.
- `agent.py` — public synchronous API and top-level turn handling.
- `session.py` — persistent draft, transcript, review, and submission state.
- `turn_processor.py` — orchestrates model calls, tool execution, and turn-level policy.
- `tool_protocol.py` — tool schemas, typed operations, outcomes, and protocol parsing.
- `draft_operations.py` — pure validation and draft-edit logic.
- `order.py` — menu validation, pricing, order rendering, and restaurant payload construction.
- `model_adapter.py` — model_adapter.py defines the generic message types, tool descriptions, and model API that the rest of the application uses, without depending on Mistral-specific classes.
- `mistral_adapter.py` — translates between application types and the Mistral SDK.
- `mistral_support.py` — Mistral configuration, client setup, and retry/error handling.
- `submission.py` — MCP `submit_order` integration and submission result handling.
- `turn_logging.py` — redacted JSONL logging.

The main design choice is to use the LLM for language understanding while keeping state changes, pricing, validation, and submission decisions in deterministic Python code.

The model and submission layer are both injectable, which keeps the core behavior easy to test without network access:

```python
agent = FoodOrderAgent(
    model=scripted_model,
    submitter=controlled_submitter,
)
```

## Logging

Turn logs are written to `logs/turns.jsonl` by default.

Each record includes information such as:

- customer input and response
- model tool calls
- parsed operations and outcomes
- state changes and totals
- submission attempts
- timing and error information

Known API keys, authorization-related fields, bearer tokens, and other sensitive values are redacted before the record is written.

Logging is best-effort: a logging failure does not change the result of the conversation.

## Verification

Run the type checker and tests with:

```sh
uv run --locked mypy
uv run --locked pytest
```

The default test suite is network-free. Model responses and MCP transports are replaced with controlled test doubles.
