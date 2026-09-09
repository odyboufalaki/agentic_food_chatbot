import json

from agent import FoodOrderAgent
from test_agent import ScriptedInterpreter, add


def test_success_and_invalid_turns_are_logged_with_normalized_operations(tmp_path):
    path = tmp_path / "nested" / "turns.jsonl"
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", extras=["cheese", "cheese"])]},
        {"operations": [add("burger")]},
        {"operations": [add("fries"), add("milkshake", options={"flavor": "vanilla bean"})]},
        {"operations": [{"type": "summary"}]},
    ), log_path=path)
    responses = [agent.send(message) for message in [
        "Burger with cheese", "Another burger", "Fries and a vanilla bean shake", "Show draft",
    ]]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 4
    assert len({record["session_id"] for record in records}) == 1
    assert [record["turn_id"] for record in records] == [1, 2, 3, 4]
    assert [record["response"] for record in records] == responses
    assert records[0]["input"] == "Burger with cheese"
    assert records[0]["operations"][0]["item_id"] == "classic_burger"
    assert records[0]["operations"][0]["options"]["patty"] == "beef"
    assert records[0]["operations"][0]["extras"] == ["cheese"]
    assert records[0]["operations"][0]["line_id"] != records[1]["operations"][0]["line_id"]
    assert records[0]["totals"] == {"before_cents": 0, "after_cents": 950}
    assert records[2]["totals"] == {"before_cents": 1800, "after_cents": 1800}
    assert records[2]["state_transition"]["before"] == records[2]["state_transition"]["after"]
    assert records[2]["error_category"] == "invalid_selection"
    assert records[2]["operations"] == []
    for record in records:
        assert record["elapsed_ms"] >= 0
        assert record["timestamp"]
        assert record["tool_calls"] == []


def test_unexpected_failed_turn_returns_message_and_logs_no_exception_details(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    secret = "secret-provider-token"
    monkeypatch.setenv("MISTRAL_API_KEY", secret)

    class BrokenInterpreter:
        def interpret(self, **context):
            raise RuntimeError(f"Authorization: Bearer {secret}; hidden reasoning")

    agent = FoodOrderAgent(interpreter=BrokenInterpreter(), log_path=path)
    response = agent.send("A burger")
    assert "unchanged" in response["message"]
    raw = path.read_text()
    assert secret not in raw
    assert "Authorization" not in raw
    assert "hidden reasoning" not in raw
    assert json.loads(raw)["error_category"] == "internal_error"


def test_log_write_failure_is_separate_sanitized_stderr_and_preserves_response(tmp_path, capsys):
    proposal = {"operations": [add("burger")]}
    normal = FoodOrderAgent(interpreter=ScriptedInterpreter(proposal), log_path=tmp_path / "ok.jsonl")
    # A directory cannot be opened as a JSONL file; its name must not leak either.
    failed_path = tmp_path / "private-customer-text-secret-token"
    failed_path.mkdir()
    failing = FoodOrderAgent(interpreter=ScriptedInterpreter(proposal, {"operations": [{"type": "summary"}]}), log_path=failed_path)
    assert failing.send("private customer text") == normal.send("private customer text")
    assert "Total: $8.50" in failing.send("Show draft")["message"]
    captured = capsys.readouterr()
    assert "logging failed" in captured.err.lower()
    assert "private" not in captured.err
    assert "secret" not in captured.err
    assert not captured.out


def test_known_credentials_are_redacted_even_when_pasted_into_input(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-provider-token")
    path = tmp_path / "turns.jsonl"
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter({"operations": [{"type": "summary"}]}), log_path=path)
    agent.send("My token is secret-provider-token")
    assert "secret-provider-token" not in path.read_text()
    assert "[REDACTED]" in path.read_text()


def test_closed_stderr_during_logging_failure_still_returns_the_draft(tmp_path):
    from contextlib import redirect_stderr
    from io import StringIO

    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("soda")]}, {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path)
    stderr = StringIO()
    stderr.close()
    with redirect_stderr(stderr):
        added = agent.send("A cola")
        assert "Total: $2.00" in added["message"]
        assert agent.send("Show draft") == added


def test_targeting_a_line_by_id_preserves_other_lines_and_checks_all_selectors(tmp_path):
    path = tmp_path / "turns.jsonl"
    proposals = [{"operations": [add("burger"), add("burger"), add("soda")]}]

    class ControlledInterpreter:
        def interpret(self, **context):
            return proposals.pop(0)

    agent = FoodOrderAgent(interpreter=ControlledInterpreter(), log_path=path)
    original = agent.send("Add a burger, another separate burger, and a cola")
    lines = json.loads(path.read_text())["operations"]
    proposals.extend([
        {"operations": [{"type": "remove_line", "target": {"line_id": lines[1]["line_id"], "item_id": "soda"}}]},
        {"operations": [{"type": "summary"}]},
        {"operations": [{"type": "edit", "target": {"line_id": lines[1]["line_id"]}, "options": {"size": "large"}}]},
        {"operations": [{"type": "remove_line", "target": {"line_id": lines[0]["line_id"]}}]},
    ])
    assert "unchanged" in agent.send("Remove the cola with an inconsistent identifier")["message"]
    assert agent.send("Show draft") == original
    edited = agent.send("Make the second burger large")["message"]
    assert "size: regular" in edited and "size: large" in edited
    assert "Total: $21.00" in edited
    removed = agent.send("Remove the first burger")["message"]
    assert "Total: $12.50" in removed
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[3]["operations"][0]["line_id"] == lines[1]["line_id"]
    assert records[4]["operations"][0]["line_id"] == lines[0]["line_id"]
