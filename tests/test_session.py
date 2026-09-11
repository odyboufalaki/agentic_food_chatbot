from agent import FoodOrderAgent
from food_ordering.session import Session
from test_agent import ScriptedInterpreter, add


def test_agent_uses_its_session_across_turns_without_sharing_state(tmp_path):
    session = Session()
    agent = FoodOrderAgent(
        interpreter=ScriptedInterpreter(
            {"operations": [add("burger")]},
            {"operations": [{"type": "summary"}]},
        ),
        log_path=tmp_path / "first.jsonl",
        session=session,
    )

    added = agent.send("Add a burger")
    summary = agent.send("Show my draft")

    assert summary == added
    assert session.revision == 1
    assert [line.line_id for line in session.lines] == ["L1"]
    assert session.next_line_number == 2
    assert [entry["role"] for entry in session.history] == [
        "user", "assistant", "user", "assistant",
    ]

    separate = FoodOrderAgent(
        interpreter=ScriptedInterpreter({"operations": [{"type": "summary"}]}),
        log_path=tmp_path / "second.jsonl",
    )
    assert "Total: $0.00" in separate.send("Show my draft")["message"]
