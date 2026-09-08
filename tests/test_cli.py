from io import StringIO
import json
import os
import subprocess
import sys
from pathlib import Path

from agent import FoodOrderAgent
from main import main
from test_agent import ScriptedInterpreter, add


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
