from io import StringIO

from agent import FoodOrderAgent
from food_ordering.model_adapter import AssistantMessage, ToolCall
from main import main
from model_fakes import ScriptedModel


def test_interactive_entry_point_uses_the_same_send_path(tmp_path) -> None:
    agent = FoodOrderAgent(
        model=ScriptedModel(
            AssistantMessage(tool_calls=(ToolCall(
                call_id="add",
                name="add_item",
                arguments={"item_id": "classic_burger", "quantity": 1},
            ),)),
            AssistantMessage(content="Added."),
        ),
        log_path=tmp_path / "turns.jsonl",
    )
    output = StringIO()

    main(
        agent=agent,
        input_stream=StringIO("A burger\nquit\n"),
        output_stream=output,
    )

    assert "Classic Burger" in output.getvalue()
    assert "Total: $8.50" in output.getvalue()
    assert "User: " in output.getvalue()
