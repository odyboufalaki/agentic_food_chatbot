import json

from agent import FoodOrderAgent
from food_ordering.model_adapter import AssistantMessage, ToolCall
from model_fakes import ScriptedModel


def test_protocol_log_matches_the_authoritative_displayed_state(tmp_path) -> None:
    path = tmp_path / "turns.jsonl"
    agent = FoodOrderAgent(
        model=ScriptedModel(
            AssistantMessage(tool_calls=(ToolCall(
                call_id="add",
                name="add_item",
                arguments={"item_id": "fries", "quantity": 2},
            ),)),
            AssistantMessage(content="Added two fries."),
        ),
        log_path=path,
    )

    response = agent.send("Two fries")
    record = json.loads(path.read_text())

    assert record["tool_protocol_version"] == 1
    assert record["response"] == response
    assert record["totals"] == {"before_cents": 0, "after_cents": 700}
    assert record["state_transition"]["after"]["revision"] == 1
    assert record["validated_operations"][0]["arguments"]["item_id"] == "fries"
    assert record["outcomes"][0]["draft"]["total_cents"] == 700
    assert record["commit_effects"][0]["created_line_ids"] == ["L1"]
    assert record["tool_calls"] == []


def test_logging_redacts_credentials_and_failure_does_not_change_response(
    tmp_path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-provider-token")
    failed_path = tmp_path / "log-target"
    failed_path.mkdir()
    agent = FoodOrderAgent(
        model=ScriptedModel(AssistantMessage(content="Credentials are private.")),
        log_path=failed_path,
    )

    response = agent.send("secret-provider-token")

    assert response == {"message": "Credentials are private."}
    captured = capsys.readouterr()
    assert "logging failed" in captured.err.lower()
    assert "secret-provider-token" not in captured.err


def test_malformed_model_material_redacts_unknown_headers_and_secret_keys(tmp_path) -> None:
    path = tmp_path / "turns.jsonl"
    agent = FoodOrderAgent(
        model=ScriptedModel(
            AssistantMessage(tool_calls=(ToolCall(
                call_id="bad",
                name="show_draft",
                arguments={
                    "headers": {"Authorization": "Bearer private-token"},
                    "api_key": "another-private-value",
                },
            ),)),
            AssistantMessage(content="I could not read the draft."),
        ),
        log_path=path,
    )

    agent.send("Show my draft")

    raw = path.read_text()
    assert "private-token" not in raw
    assert "another-private-value" not in raw
    assert raw.count("[REDACTED]") >= 2
