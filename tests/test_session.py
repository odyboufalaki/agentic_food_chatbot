from agent import FoodOrderAgent
from food_ordering.model_adapter import AssistantMessage, ToolCall
from food_ordering.session import Session
from model_fakes import ScriptedModel


def test_agent_reuses_only_its_own_session_across_fresh_turn_processors(tmp_path) -> None:
    session = Session()
    agent = FoodOrderAgent(
        model=ScriptedModel(
            AssistantMessage(tool_calls=(ToolCall(
                call_id="add",
                name="add_item",
                arguments={"item_id": "fries", "quantity": 1},
            ),)),
            AssistantMessage(content="Added."),
            AssistantMessage(tool_calls=(ToolCall(
                call_id="show", name="show_draft", arguments={},
            ),)),
            AssistantMessage(content="You have fries."),
        ),
        session=session,
        log_path=tmp_path / "turns.jsonl",
    )

    agent.send("Add fries")
    response = agent.send("Show my draft")

    assert "fries" in response["message"].lower()
    assert session.turn_id == 2
    assert session.revision == 1
    assert [line.line_id for line in session.lines] == ["L1"]
