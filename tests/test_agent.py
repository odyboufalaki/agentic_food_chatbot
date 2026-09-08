from agent import FoodOrderAgent
import pytest


class ScriptedInterpreter:
    def __init__(self, *proposals):
        self.proposals = iter(proposals)

    def interpret(self, **context):
        return next(self.proposals)


def add(item_id, quantity=1, options=None, extras=None):
    return {
        "type": "add",
        "item_id": item_id,
        "quantity": quantity,
        "options": options or {},
        "extras": extras or [],
    }


def test_large_burger_with_cheese_and_bacon_costs_thirteen_dollars(tmp_path):
    agent = FoodOrderAgent(
        interpreter=ScriptedInterpreter({"operations": [
            add("burger", options={"size": "large"}, extras=["cheese", "bacon"])
        ]}),
        log_path=tmp_path / "turns.jsonl",
    )
    response = agent.send("I'd like a large classic burger with cheese and bacon")
    assert isinstance(response["message"], str)
    assert "Classic Burger" in response["message"]
    assert "large" in response["message"]
    assert "beef" in response["message"]
    assert "cheese" in response["message"]
    assert "bacon" in response["message"]
    assert "$13.00" in response["message"]
    assert not response.get("tool_calls")


@pytest.mark.parametrize("invalid", [
    add("not_on_menu"),
    add("fries", options={"patty": "beef"}),
    add("classic_burger", options={"size": "giant"}),
    add("soda", extras=["bacon"]),
    add("milkshake"),
    add("fries", quantity=0),
    add("fries", quantity=-1),
    add("fries", quantity=True),
    add("fries", quantity=1.5),
    add("fries", quantity="2"),
    {**add("fries"), "options": []},
    {**add("fries"), "extras": "parmesan"},
    {**add("fries"), "total": 0},
    {"type": "submit"},
])
def test_invalid_message_leaves_all_existing_selections_unchanged(tmp_path, invalid):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("soda")]},
        {"operations": [add("fries"), invalid]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    initial = agent.send("A cola")
    failed = agent.send("Add fries and another selection")
    assert isinstance(failed["message"], str)
    assert "unchanged" in failed["message"]
    assert agent.send("What is in my draft?") == initial


def test_menu_questions_show_authoritative_choices_without_changing_draft(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger")]},
        {"operations": [{"type": "menu", "item_ids": ["milkshake"]}]},
        {"operations": [{"type": "menu", "item_ids": []}]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    initial = agent.send("A burger")
    milkshake = agent.send("What milkshakes do you have?")["message"]
    assert "Milkshake" in milkshake
    assert "$5.50" in milkshake
    assert all(flavor in milkshake for flavor in ["vanilla", "chocolate", "strawberry", "oreo"])
    assert "required" in milkshake
    entire_menu = agent.send("Show the menu")["message"]
    assert all(name in entire_menu for name in [
        "Classic Burger", "Spicy Jalapeño Burger", "Margherita Pizza", "French Fries",
        "Onion Rings", "Soft Drink", "Milkshake",
    ])
    assert "-$2.00" in entire_menu
    assert agent.send("Show my draft") == initial


@pytest.mark.parametrize("selections,total,details", [
    ([add("margherita", options={"size": "medium"}, extras=["olives"]),
      add("fries", options={"size": "large"}, extras=["parmesan"]),
      add("soda", options={"size": "large"})], "$22.25", ["cola", "regular"]),
    ([add("margherita", quantity=2, options={"size": "small"}, extras=["olives", "olives"])],
     "$23.00", ["2 ×", "small"]),
    ([add("fries", quantity=2, options={"size": "small"}),
      add("soda", quantity=2, options={"size": "small"})], "$8.00", ["cola"]),
    ([add("spicy_burger", extras=["jalapeños"]), add("onion_rings", options={"size": "small"}),
      add("milkshake", options={"flavor": "oreo"}, extras=["whipped_cream", "cherry_on_top"])],
     "$20.00", ["medium", "oreo", "regular"]),
])
def test_menu_pricing_examples(tmp_path, selections, total, details):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter({"operations": selections}),
                           log_path=tmp_path / "turns.jsonl")
    response = agent.send("Add these fully specified selections")["message"]
    assert f"Total: {total}" in response
    assert all(detail in response for detail in details)


def test_over_limit_draft_stays_open_to_additions(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", quantity=6)]},
        {"operations": [add("soda")]},
    ), log_path=tmp_path / "turns.jsonl")
    assert "Total: $51.00" in agent.send("Six burgers")["message"]
    assert "Total: $53.00" in agent.send("Add a cola")["message"]


def test_missing_milkshake_flavor_lists_choices_and_does_not_add_other_items(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("fries"), add("milkshake")]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    response = agent.send("Fries and a milkshake")["message"]
    assert "flavor" in response and "vanilla" in response
    assert "Total: $0.00" in agent.send("Show draft")["message"]


def test_unsupported_request_does_not_apply_its_other_additions(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger")]},
        {"operations": [add("fries"), {"type": "unsupported", "reason": "not_available"}]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    initial = agent.send("A burger")
    response = agent.send("Add fries and submit")["message"]
    assert "not available" in response
    assert "unchanged" in response
    assert agent.send("Show draft") == initial


def test_typed_proposals_are_revalidated_at_the_send_boundary(tmp_path):
    from food_ordering.proposals import Add, Proposal

    malformed = Proposal.model_construct(operations=[
        Add.model_construct(type="add", item_id="fries", quantity=True, options={}, extras=[]),
    ])
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(malformed, {"operations": [{"type": "summary"}]}),
                           log_path=tmp_path / "turns.jsonl")
    assert "unchanged" in agent.send("Fries")["message"]
    assert "Total: $0.00" in agent.send("Show draft")["message"]
