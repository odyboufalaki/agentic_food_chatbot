from io import StringIO
import json
import os
import subprocess
import sys
from pathlib import Path

from agent import FoodOrderAgent
from main import main
from test_agent import ScriptedInterpreter, add
from test_customization import edit
from test_submission import RestaurantTransport, restaurant_agent


def test_interactive_entry_point_uses_send_and_logs_the_turn(tmp_path):
    path = tmp_path / "turns.jsonl"
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter({"operations": [
        add("burger", options={"size": "large"}, extras=["cheese", "bacon"])
    ]}), log_path=path)
    output = StringIO()
    main(agent=agent, input_stream=StringIO("A large burger with cheese and bacon\nquit\n"), output_stream=output)
    record = json.loads(path.read_text())
    assert record["response"]["message"] in output.getvalue()
    assert "$13.00" in output.getvalue()
    assert record["input"] == "A large burger with cheese and bacon"


def test_required_python_entry_points_work_without_credentials(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "MISTRAL_API_KEY": "", "FOOD_ORDER_LOG_PATH": str(tmp_path / "turns.jsonl")}
    cli = subprocess.run([sys.executable, "main.py"], input="A burger\nquit\n", text=True,
                         capture_output=True, cwd=root, env=env, timeout=10)
    assert cli.returncode == 0
    assert "configuration" in cli.stdout
    assert not cli.stderr
    api = subprocess.run([sys.executable, "-c",
        "from agent import FoodOrderAgent; print(FoodOrderAgent().send('A burger')['message'])"],
        text=True, capture_output=True, cwd=root, env=env, timeout=10)
    assert api.returncode == 0
    assert "configuration" in api.stdout
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["session_id"] != records[1]["session_id"]


def test_cli_edits_and_removals_match_logged_summaries_and_prices(tmp_path):
    path = tmp_path / "turns.jsonl"
    target = {"item_id": "burger"}
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", options={"size": "large"}, extras=["cheese", "bacon"])]},
        {"operations": [{"type": "edit", "target": target, "remove_extras": ["bacon"]}]},
        {"operations": [{"type": "edit", "target": target, "options": {"size": "regular"}}]},
        {"operations": [{"type": "edit", "target": target, "add_extras": ["cheese", "cheese"]}]},
        {"operations": [{"type": "set_quantity", "target": target, "quantity": 6}]},
        {"operations": [{"type": "remove_units", "target": target, "quantity": 2}]},
        {"operations": [{"type": "increase_quantity", "target": target, "quantity": 1}]},
        {"operations": [
            {"type": "edit", "target": target, "remove_extras": ["cheese"]},
            {"type": "remove_line", "target": {"item_id": "soda"}},
        ]},
        {"operations": [{"type": "remove_line", "target": target}]},
        {"operations": [add("fries")]},
        {"operations": [{"type": "clear_draft"}]},
    ), log_path=path)
    messages = [
        "A large burger with cheese and bacon", "Remove the bacon", "Make the burger regular",
        "Add cheese again", "Make that six burgers", "Remove two burgers", "One more burger",
        "Remove the cheese and the cola", "Remove the burger line", "Add fries", "Cancel my order",
    ]
    output = StringIO()
    main(agent=agent, input_stream=StringIO("\n".join([*messages, "quit", ""])), output_stream=output)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["input"] for record in records] == messages
    assert [record["totals"]["after_cents"] for record in records] == [
        1300, 1150, 950, 950, 5700, 3800, 4750, 4750, 0, 350, 0,
    ]
    for record in records:
        assert record["response"]["message"] in output.getvalue()
        assert record["tool_calls"] == []
    assert records[7]["error_category"] == "invalid_selection"
    assert records[7]["operations"] == []
    assert records[7]["state_transition"]["before"] == records[7]["state_transition"]["after"]
    assert all(record["error_category"] is None for i, record in enumerate(records) if i != 7)
    line_id = records[0]["operations"][0]["line_id"]
    assert all(records[i]["operations"][0]["line_id"] == line_id for i in [1, 2, 3, 4, 5, 6, 8])
    assert records[4]["operations"][0]["after_quantity"] == 6
    assert records[5]["operations"][0]["after_quantity"] == 4
    assert records[6]["operations"][0]["after_quantity"] == 5
    assert records[10]["operations"][0]["line_ids"] == [records[9]["operations"][0]["line_id"]]


def test_cli_customization_demo_reviews_and_submits_the_logged_order(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", quantity=2)]},
        {"operations": [edit(servings=1, options={"patty": "chicken"}, instructions="no onions")]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    output = StringIO()
    main(agent=agent, input_stream=StringIO(
        "Two burgers\nMake one chicken with no onions\nReview\nYes\nquit\n"
    ), output_stream=output)
    text = output.getvalue()
    assert "2 × Classic Burger" in text and "patty: chicken" in text and "no onions" in text
    assert "Total: $17.00" in text and "ORD-12345" in text
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert all(record["response"]["message"] in text for record in records)
    assert all(record["tool_calls"] == [] for record in records[:-1])
    assert len(transport.calls) == 1
    assert records[-1]["tool_calls"][0]["arguments"] == transport.calls[0]["arguments"]
