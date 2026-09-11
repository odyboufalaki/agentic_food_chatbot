import json

import pytest

from test_agent import add
from test_submission import RestaurantTransport, restaurant_agent


def edit(**changes):
    return {"type": "edit", "target": {"item_id": "burger"}, **changes}


def test_customize_one_burger_then_review_and_submit_scoped_notes(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger"), add("burger")]},
        {"operations": [edit(servings=1, options={"patty": "chicken"}, instructions="no onions")]},
        {"operations": [{"type": "review"}]},
        {"operations": [{"type": "confirm"}]},
    )
    original = agent.send("Two identical burgers")["message"]
    assert "2 × Classic Burger" in original and "Total: $17.00" in original
    changed = agent.send("Make one chicken with no onions")["message"]
    assert changed.count("1 × Classic Burger") == 2
    assert "patty: beef" in changed and "patty: chicken" in changed
    assert changed.count("no onions") == 1 and "Total: $17.00" in changed
    assert "no onions" in agent.send("Review")["message"]
    assert transport.calls == []
    response = agent.send("Yes")
    assert "ORD-12345" in response["message"] and len(transport.calls) == 1
    payload = transport.calls[0]["arguments"]
    assert [item["quantity"] for item in payload["items"]] == [1, 1]
    assert payload["special_instructions"] == (
        "Item 1: 1 × Classic Burger (size: regular, patty: chicken): no onions"
    )
    assert all("line_id" not in item for item in payload["items"])
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert records[1]["response"]["message"] == changed
    assert records[-1]["tool_calls"][0]["arguments"] == payload
    assert records[-1]["totals"]["after_cents"] == 1700


@pytest.mark.parametrize("quantity, expected_quantities", [(2, [1, 1, 1]), ("all", [1, 2])])
def test_explicit_servings_span_stored_lines_in_earliest_order(tmp_path, quantity, expected_quantities):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger"), add("burger", quantity=2)]},
        {"operations": [edit(servings=quantity, instructions="cut in half")]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("One burger, then two more")
    changed = agent.send(f"Cut {quantity} burgers in half")["message"]
    assert "Total: $25.50" in changed
    agent.send("Review")
    agent.send("Yes")
    payload = transport.calls[0]["arguments"]
    assert [item["quantity"] for item in payload["items"]] == expected_quantities
    assert "Item 1: 1 ×" in payload["special_instructions"]
    if quantity == 2:
        assert "Item 3: 1 ×" in payload["special_instructions"]
        assert "Item 2:" not in payload["special_instructions"]


def test_ambiguous_milkshake_target_does_not_use_recent_flavor(tmp_path):
    transport = RestaurantTransport()
    target = {"item_id": "milkshake"}
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("milkshake", options={"flavor": "strawberry"})]},
        {"operations": [add("milkshake", options={"flavor": "chocolate"})]},
        {"operations": [{"type": "edit", "target": target, "options": {"size": "large"}}]},
        {"operations": [{"type": "edit", "target": {**target, "options": {"flavor": "strawberry"}},
                         "options": {"size": "large"}}]},
    )
    agent.send("One strawberry milkshake")
    agent.send("And a chocolate milkshake")
    question = agent.send("Make the milkshake large")["message"]
    assert "which" in question.lower() and "haven't changed" in question
    result = agent.send("The strawberry one")["message"]
    assert "size: large, flavor: strawberry" in result
    assert "size: regular, flavor: chocolate" in result
    assert "Total: $13.00" in result and transport.calls == []


def test_ambiguous_number_of_servings_requires_quantity_then_splits(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", quantity=3)]},
        {"operations": [{"type": "clarify", "reason": "quantity", "item_id": "burger"}]},
        {"operations": [edit(servings=1, instructions="no onions")]},
    )
    agent.send("Three burgers")
    question = agent.send("Make some without onions")["message"]
    assert "how many" in question.lower() and "haven't changed" in question
    result = agent.send("One")["message"]
    assert "2 × Classic Burger" in result and "1 × Classic Burger" in result
    assert result.count("no onions") == 1 and "Total: $25.50" in result


def test_general_notes_survive_item_removal_and_require_fresh_approval(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{**add("burger"), "instructions": "no onions"}, add("fries"),
                        {"type": "set_instructions", "instructions": "pack separately"}]},
        {"operations": [{"type": "review"}]},
        {"operations": [{"type": "set_instructions", "instructions": "no cutlery"},
                        {"type": "confirm"}]},
        {"operations": [{"type": "remove_line", "target": {"item_id": "burger"}}]},
        {"operations": [{"type": "confirm"}]}, {"operations": [{"type": "confirm"}]},
    )
    initial = agent.send("A burger without onions, fries, pack separately")["message"]
    assert "no onions" in initial and "pack separately" in initial
    agent.send("Review")
    revised = agent.send("Yes but no cutlery instead")["message"]
    assert "no cutlery" in revised and "pack separately" not in revised
    assert transport.calls == []
    removed = agent.send("Remove the burger")["message"]
    assert "no onions" not in removed and "no cutlery" in removed
    assert "confirm" in agent.send("Yes")["message"].lower()
    assert transport.calls == []
    response = agent.send("Yes")
    assert len(transport.calls) == 1
    assert response["tool_calls"][0]["arguments"]["special_instructions"] == "General: no cutlery"


def test_flavor_clarification_distributes_exactly_two_established_milkshakes(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{"type": "clarify", "reason": "quantity", "item_id": "milkshake"}]},
        {"operations": [add("milkshake", quantity=2)]},
        {"operations": [add("milkshake", options={"flavor": "vanilla"}),
                        add("milkshake", options={"flavor": "strawberry"})]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    assert "how many" in agent.send("I want milkshakes")["message"].lower()
    assert "flavor" in agent.send("2")["message"]
    resolved = agent.send("vanilla and strawberry")["message"]
    assert resolved.count("1 × Milkshake") == 2 and "Total: $11.00" in resolved
    agent.send("Review")
    agent.send("Yes")
    assert [item["quantity"] for item in transport.calls[0]["arguments"]["items"]] == [1, 1]


@pytest.mark.parametrize("replacement", [
    [add("milkshake", quantity=3, options={"flavor": "vanilla", "size": "large"}, extras=["cherry_on_top"])],
    [add("milkshake", quantity=2, options={"flavor": "vanilla"}, extras=["cherry_on_top"])],
    [add("milkshake", quantity=2, options={"flavor": "vanilla", "size": "large"})],
    [add("milkshake", quantity=2, options={"flavor": "vanilla", "size": "large"}, extras=["cherry_on_top"]),
     add("milkshake", options={"flavor": "strawberry", "size": "large"}, extras=["cherry_on_top"])],
])
def test_clarification_cannot_silently_change_resolved_quantity_options_or_extras(tmp_path, replacement):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("milkshake", quantity=2, options={"size": "large"}, extras=["cherry_on_top"])]},
        {"operations": replacement},
        {"operations": [add("milkshake", quantity=2, options={"size": "large", "flavor": "vanilla"},
                            extras=["cherry_on_top"])]},
    )
    agent.send("Two large milkshakes with cherries")
    response = agent.send("Vanilla")["message"]
    assert response.startswith("I couldn't apply that answer to the change we're working on.")
    assert "preserve the resolved" not in response.lower()
    assert "haven't changed" in response
    resolved = agent.send("Vanilla, keeping everything else")["message"]
    assert "2 × Milkshake" in resolved and "Total: $15.50" in resolved


@pytest.mark.parametrize("operation, expected", [
    (edit(options={"size": "large"}), [(2, "large", "no onions")]),
    (edit(servings=1, options={"patty": "chicken"}), [(1, "regular", "no onions"), (1, "regular", "no onions")]),
    ({"type": "set_quantity", "target": {"item_id": "burger"}, "quantity": 3}, [(3, "regular", "no onions")]),
])
def test_notes_follow_resizing_and_splitting_without_duplicate_extras(tmp_path, operation, expected):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{**add("burger", quantity=2, extras=["cheese"]), "instructions": "no onions"}]},
        {"operations": [operation]},
        {"operations": [edit(servings="all", add_extras=["cheese", "cheese"])]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("Two cheeseburgers without onions")
    changed = agent.send("Change the servings")["message"]
    assert "no onions" in changed
    agent.send("Add cheese to all of them again")
    agent.send("Review")
    agent.send("Yes")
    payload = transport.calls[0]["arguments"]
    for index, (item, (quantity, size, note)) in enumerate(zip(payload["items"], expected, strict=True), 1):
        assert item["quantity"] == quantity and item["options"]["size"] == size
        assert item["extras"] == ["cheese"]
        assert f"Item {index}:" in payload["special_instructions"]
        assert note in payload["special_instructions"]


@pytest.mark.parametrize("decision, has_note", [("keep", True), ("discard", False)])
def test_replacement_asks_whether_existing_notes_still_apply(tmp_path, decision, has_note):
    transport = RestaurantTransport()
    replacement = edit(replacement_item_id="fries")
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{**add("burger"), "instructions": "no onions"}]},
        {"operations": [replacement]},
        {"operations": [{**replacement, "replacement_notes": decision}]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger without onions")
    question = agent.send("Replace the burger with fries")["message"]
    assert "keep" in question.lower() and "discard" in question.lower() and "haven't changed" in question
    resolved = agent.send(decision)["message"]
    assert "French Fries" in resolved and "Classic Burger" not in resolved
    assert ("no onions" in resolved) is has_note
    agent.send("Review")
    agent.send("Yes")
    assert ("special_instructions" in transport.calls[0]["arguments"]) is has_note


def test_explicit_pending_correction_changes_only_the_named_field(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("milkshake", quantity=2, options={"size": "large"})]},
        {"operations": [add("milkshake", quantity=3, options={"flavor": "vanilla"})],
         "corrected_fields": ["0.quantity"]},
        {"operations": [add("milkshake", quantity=3, options={"flavor": "vanilla", "size": "large"})],
         "corrected_fields": ["0.quantity"]},
    )
    agent.send("Two large milkshakes")
    assert "haven't changed" in agent.send("Actually three, vanilla")["message"]
    result = agent.send("Three large vanilla milkshakes")["message"]
    assert "3 × Milkshake" in result and "Total: $22.50" in result


@pytest.mark.parametrize("operation, explanation", [
    ({"type": "unsupported", "reason": "dietary_guarantee"}, "does not verify"),
    (edit(add_extras=["lobster"]), "Unsupported extra"),
])
def test_unverified_dietary_guarantees_and_unlisted_additions_do_not_become_notes(tmp_path, operation, explanation):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger")]},
        {"operations": [operation]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("A burger")
    assert explanation in agent.send("An unavailable customization")["message"]
    agent.send("Review")
    agent.send("Yes")
    payload = transport.calls[0]["arguments"]
    assert "special_instructions" not in payload and payload["items"][0]["extras"] == []


def test_notes_distinguish_otherwise_identical_servings_and_can_be_cleared(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", quantity=2)]},
        {"operations": [edit(servings=1, instructions="no onions")]},
        {"operations": [edit(target={"item_id": "burger", "instructions": "no onions"}, instructions="")]},
        {"operations": [{"type": "review"}]}, {"operations": [{"type": "confirm"}]},
    )
    agent.send("Two burgers")
    split = agent.send("One without onions")["message"]
    assert split.count("1 × Classic Burger") == 2
    merged_display = agent.send("Clear the no onions request")["message"]
    assert "2 × Classic Burger" in merged_display and "no onions" not in merged_display
    agent.send("Review")
    agent.send("Yes")
    payload = transport.calls[0]["arguments"]
    assert len(payload["items"]) == 2 and "special_instructions" not in payload


def test_split_log_accounts_for_affected_and_unaffected_servings(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", quantity=3)]},
        {"operations": [edit(servings=1, instructions="no onions")]},
    )
    agent.send("Three burgers")
    response = agent.send("One without onions")
    record = json.loads((tmp_path / "turns.jsonl").read_text().splitlines()[-1])
    split = next(operation for operation in record["operations"] if operation["type"] == "split")
    assert [line["quantity"] for line in split["lines"]] == [2, 1]
    assert [line["instructions"] for line in split["lines"]] == ["", "no onions"]
    assert record["response"] == response and record["totals"]["after_cents"] == 2550


def test_target_clarification_can_identify_an_existing_extra(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("burger", extras=["cheese"]), add("burger", extras=["bacon"])]},
        {"operations": [edit(options={"size": "large"})]},
        {"operations": [edit(target={"item_id": "burger", "extras": ["cheese"]},
                             options={"size": "large"})]},
    )
    agent.send("One cheeseburger and one bacon burger")
    assert "which" in agent.send("Make the burger large")["message"].lower()
    result = agent.send("The one with cheese")["message"]
    assert "size: large, patty: beef; extras: cheese" in result
    assert "size: regular, patty: beef; extras: bacon" in result and "Total: $21.50" in result


def test_corrected_quantity_can_be_distributed_across_flavors(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("milkshake", quantity=2), add("fries")]},
        {"operations": [add("milkshake", quantity=2, options={"flavor": "vanilla"}),
                        add("milkshake", options={"flavor": "strawberry"}), add("fries")],
         "corrected_fields": ["0.quantity"]},
    )
    agent.send("Two milkshakes and fries")
    result = agent.send("Actually three milkshakes: two vanilla and one strawberry")["message"]
    assert "2 × Milkshake" in result and "1 × Milkshake" in result
    assert "1 × French Fries" in result and "Total: $20.00" in result


def test_flavor_answer_preserves_menu_default_size(tmp_path):
    transport = RestaurantTransport()
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [add("milkshake", quantity=2)]},
        {"operations": [add("milkshake", quantity=2, options={"flavor": "vanilla", "size": "large"})]},
        {"operations": [add("milkshake", quantity=2, options={"flavor": "vanilla"})]},
    )
    agent.send("Two milkshakes")
    assert "haven't changed" in agent.send("Vanilla")["message"]
    result = agent.send("Vanilla, keeping the default size")["message"]
    assert "size: regular" in result and "Total: $11.00" in result


def test_unclear_replacement_note_answer_repeats_the_question(tmp_path):
    transport = RestaurantTransport()
    replacement = edit(replacement_item_id="fries")
    agent = restaurant_agent(tmp_path, transport,
        {"operations": [{**add("burger"), "instructions": "no onions"}]},
        {"operations": [replacement]},
        {"operations": [{"type": "clarify", "reason": "replacement_notes", "item_id": "burger"}]},
        {"operations": [{**replacement, "replacement_notes": "discard"}]},
    )
    agent.send("A burger without onions")
    agent.send("Replace it with fries")
    question = agent.send("Maybe")["message"]
    assert "keep" in question.lower() and "discard" in question.lower() and "haven't changed" in question
    result = agent.send("Discard")["message"]
    assert "French Fries" in result and "no onions" not in result
