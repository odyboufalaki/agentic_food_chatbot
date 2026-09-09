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


def test_remove_extra_reprices_the_draft(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", options={"size": "large"}, extras=["cheese", "bacon"])]},
        {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "remove_extras": ["bacon"]}]},
    ), log_path=tmp_path / "turns.jsonl")
    assert "Total: $13.00" in agent.send("A large burger with cheese and bacon")["message"]
    edited = agent.send("Remove the bacon from my burger")["message"]
    assert "Total: $11.50" in edited
    assert "cheese" in edited and "bacon" not in edited


def test_resize_then_remove_the_identified_line(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", options={"size": "large"}, extras=["cheese"]), add("soda")]},
        {"operations": [{"type": "edit", "target": {"item_id": "burger"}, "options": {"size": "regular"}}]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "burger"}}]},
    ), log_path=tmp_path / "turns.jsonl")
    agent.send("A large burger with cheese and a cola")
    resized = agent.send("Make the burger regular size")["message"]
    assert "size: regular" in resized and "cheese" in resized
    assert "Total: $11.50" in resized
    removed = agent.send("Remove the burger")["message"]
    assert "Classic Burger" not in removed and "Soft Drink" in removed
    assert "Total: $2.00" in removed


def test_set_increase_and_remove_units_can_bring_draft_under_limit(tmp_path):
    target = {"item_id": "burger"}
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", quantity=6)]},
        {"operations": [{"type": "set_quantity", "target": target, "quantity": 2}]},
        {"operations": [{"type": "increase_quantity", "target": target, "quantity": 3}]},
        {"operations": [{"type": "remove_units", "target": target, "quantity": 4}]},
        {"operations": [{"type": "remove_units", "target": target, "quantity": 1}]},
    ), log_path=tmp_path / "turns.jsonl")
    for message, total, quantity in [
        ("Six burgers", "$51.00", 6),
        ("Make that two burgers", "$17.00", 2),
        ("Increase the burger quantity by three", "$42.50", 5),
        ("Remove four burgers", "$8.50", 1),
        ("Remove one burger", "$0.00", 0),
    ]:
        response = agent.send(message)["message"]
        assert f"Total: {total}" in response
        if quantity:
            assert f"{quantity} × Classic Burger" in response
        else:
            assert "Classic Burger" not in response


def test_cancel_draft_clears_it_and_later_additions_start_fresh(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", quantity=6), add("soda")]},
        {"operations": [{"type": "clear_draft"}]},
        {"operations": [{"type": "clear_draft"}]},
        {"operations": [add("fries")]},
    ), log_path=tmp_path / "turns.jsonl")
    agent.send("Six burgers and a cola")
    cleared = agent.send("Cancel my entire order")
    assert "Total: $0.00" in cleared["message"]
    assert "Classic Burger" not in cleared["message"]
    assert agent.send("Clear the draft again") == cleared
    fresh = agent.send("Add fries")["message"]
    assert "Total: $3.50" in fresh and "Classic Burger" not in fresh


def test_resolved_mixed_changes_apply_together_without_duplicate_extras(tmp_path):
    target = {"item_id": "burger"}
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", extras=["cheese", "bacon"])]},
        {"operations": [
            {"type": "edit", "target": target, "remove_extras": ["bacon"], "add_extras": ["cheese", "cheese"]},
            {"type": "increase_quantity", "target": target, "quantity": 1},
            add("soda"),
        ]},
    ), log_path=tmp_path / "turns.jsonl")
    agent.send("A burger with cheese and bacon")
    response = agent.send("Remove bacon, add cheese again, make it two burgers, and add a cola")["message"]
    assert "Total: $21.00" in response
    assert "2 × Classic Burger" in response
    assert response.count("cheese") == 1 and "bacon" not in response


@pytest.mark.parametrize("invalid", [
    {"type": "edit", "target": {"item_id": "milkshake"}, "options": {"flavor": "oreo"}},
    {"type": "remove_line", "target": {"item_id": "spicy_burger"}},
    {"type": "remove_line", "target": {"line_id": "nonexistent"}},
    {"type": "remove_line", "target": {}},
    {"type": "edit", "target": {"item_id": "burger"}, "options": {"size": "giant"}},
    {"type": "edit", "target": {"item_id": "burger"}, "options": {"flavor": "oreo"}},
    {"type": "edit", "target": {"item_id": "burger"}, "add_extras": ["unlisted"]},
    {"type": "edit", "target": {"item_id": "burger"}, "remove_extras": ["bacon"]},
    {"type": "edit", "target": {"item_id": "burger"}, "add_extras": ["cheese"], "remove_extras": ["cheese"]},
    {"type": "edit", "target": {"item_id": "burger"}},
    {"type": "edit", "target": {"item_id": "burger"}, "item_id": "spicy_burger"},
    {"type": "remove_units", "target": {"item_id": "burger"}, "quantity": 3},
    {"type": "unsupported", "reason": "unclear"},
])
def test_invalid_mixed_edits_preserve_the_entire_draft(tmp_path, invalid):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", quantity=2, extras=["cheese"]), add("soda")]},
        {"operations": [
            {"type": "remove_line", "target": {"item_id": "soda"}},
            {"type": "edit", "target": {"item_id": "burger"}, "options": {"size": "large"}},
            invalid,
        ]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    original = agent.send("Two cheeseburgers and a cola")
    assert "unchanged" in agent.send("Remove the cola, resize the burgers and make another change")["message"]
    assert agent.send("Show draft") == original


@pytest.mark.parametrize("kind", ["set_quantity", "increase_quantity", "remove_units"])
@pytest.mark.parametrize("quantity", [0, -1, True, 1.5, "2", None])
def test_invalid_edit_quantities_never_remove_all_servings(tmp_path, kind, quantity):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger", quantity=2)]},
        {"operations": [{"type": kind, "target": {"item_id": "burger"}, "quantity": quantity}]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    original = agent.send("Two burgers")
    assert "unchanged" in agent.send("Change the quantity")["message"]
    assert agent.send("Show draft") == original


@pytest.mark.parametrize("second_options", [{}, {"size": "large"}])
def test_ambiguous_reference_never_selects_an_arbitrary_line(tmp_path, second_options):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger"), add("burger", options=second_options)]},
        {"operations": [{"type": "remove_units", "target": {"item_id": "burger"}, "quantity": 1}]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    original = agent.send("Add two separate burger selections")
    response = agent.send("Remove one burger")["message"]
    assert "unchanged" in response and "which" in response
    assert agent.send("Show draft") == original


def test_options_and_extras_identify_a_line_without_replacing_its_product(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger"), add("burger", options={"size": "large"}, extras=["cheese"])]},
        {"operations": [{"type": "edit", "target": {
            "item_id": "burger", "options": {"size": "large"}, "extras": ["cheese"],
        }, "options": {"patty": "veggie"}}]},
    ), log_path=tmp_path / "turns.jsonl")
    agent.send("A regular burger and a large cheeseburger")
    response = agent.send("Make the large cheeseburger veggie")["message"]
    assert "Total: $20.00" in response
    assert "patty: beef" in response and "patty: veggie" in response
    assert response.count("Classic Burger") == 2


def test_invalid_change_after_cancellation_rolls_back_the_clear(tmp_path):
    agent = FoodOrderAgent(interpreter=ScriptedInterpreter(
        {"operations": [add("burger")]},
        {"operations": [{"type": "clear_draft"}, add("milkshake")]},
        {"operations": [{"type": "summary"}]},
    ), log_path=tmp_path / "turns.jsonl")
    original = agent.send("A burger")
    assert "unchanged" in agent.send("Cancel my order and add a milkshake")["message"]
    assert agent.send("Show draft") == original


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
