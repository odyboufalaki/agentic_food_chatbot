from collections.abc import Sequence
from typing import Any

import pytest

from food_ordering.menu import load_menu
from food_ordering.model_adapter import (
    AssistantMessage,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
)
from food_ordering.order import OrderLine
from food_ordering.session import Session
from food_ordering.turn_processor import TurnProcessor


class ScriptedModel:
    def __init__(self, *responses: AssistantMessage) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[ModelMessage, ...]] = []

    def complete(
        self, *, messages: Sequence[ModelMessage], tools: Sequence[ToolSpec],
    ) -> AssistantMessage:
        self.requests.append(tuple(messages))
        return next(self._responses)


def _payload(message: ModelMessage) -> Any:
    assert isinstance(message, ToolResultMessage)
    return message.payload.model_dump(mode="json", exclude_none=True)


def _burger(
    *,
    line_id: str = "L1",
    quantity: int = 1,
    extras: tuple[str, ...] = (),
    instructions: str = "",
) -> OrderLine:
    return OrderLine(
        line_id=line_id,
        item_id="classic_burger",
        name="Classic Burger",
        quantity=quantity,
        options=(("size", "regular"), ("patty", "beef")),
        extras=extras,
        instructions=instructions,
        unit_cents=850 + (100 if "cheese" in extras else 0),
    )


def test_setting_total_quantity_resizes_the_line_without_customizing_a_subset() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="set-four",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 4,
            },
        ),)),
        AssistantMessage(content="There are now four burgers."),
    )
    session = Session(
        lines=[_burger()], next_line_number=2, revision=1, reviewed_revision=1,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make the quantity four",
    )

    assert [(line.line_id, line.quantity) for line in session.lines] == [("L1", 4)]
    assert session.revision == 2
    assert session.reviewed_revision is None
    result = _payload(model.requests[1][-1])
    assert result["outcome"] == "APPLIED"
    assert result["effect"] == {
        "operation": "change_quantity",
        "affected_line_ids": ["L1"],
        "created_line_ids": [],
        "removed_line_ids": [],
    }
    assert result["draft"]["total_cents"] == 3400


def test_ambiguous_quantity_change_returns_target_facts_without_mutation() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="ambiguous-burgers",
            name="change_quantity",
            arguments={
                "target": {"type": "match", "item_id": "classic_burger"},
                "mode": "set",
                "quantity": 4,
            },
        ),)),
        AssistantMessage(content="Which burger selection do you mean?"),
    )
    original = [_burger(), _burger(line_id="L2", extras=("cheese",))]
    session = Session(lines=list(original), next_line_number=3, revision=2)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make the burgers four",
    )

    result = _payload(model.requests[1][-1])
    assert result["outcome"] == "INCOMPLETE"
    assert result["key"] == "target"
    assert [choice["value"] for choice in result["alternatives"]] == ["L1", "L2"]
    assert "cheese" in result["alternatives"][1]["label"]
    assert session.lines == original
    assert session.revision == 2


def test_removing_the_full_quantity_removes_only_the_target_line() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="remove-two",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "remove",
                "quantity": 2,
            },
        ),)),
        AssistantMessage(content="I removed both burgers."),
    )
    remaining = _burger(line_id="L2", extras=("cheese",))
    session = Session(
        lines=[_burger(quantity=2), remaining],
        next_line_number=3,
        revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Remove both plain burgers",
    )

    assert session.lines == [remaining]
    assert session.revision == 3
    result = _payload(model.requests[1][-1])
    assert result["effect"]["affected_line_ids"] == []
    assert result["effect"]["removed_line_ids"] == ["L1"]


@pytest.mark.parametrize(
    ("mode", "amount", "expected_quantity", "expected_outcome", "expected_revision"),
    [
        ("increase", 3, 5, "APPLIED", 5),
        ("remove", 1, 1, "APPLIED", 5),
        ("set", 2, 2, "ALREADY_APPLIED", 4),
    ],
)
def test_quantity_modes_apply_the_requested_total_or_report_a_no_op(
    mode: str,
    amount: int,
    expected_quantity: int,
    expected_outcome: str,
    expected_revision: int,
) -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="resize",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": mode,
                "quantity": amount,
            },
        ),)),
        AssistantMessage(content="Quantity handled."),
    )
    session = Session(
        lines=[_burger(quantity=2)],
        next_line_number=2,
        revision=4,
        reviewed_revision=4,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Change the burger quantity",
    )

    assert session.lines[0].quantity == expected_quantity
    assert session.revision == expected_revision
    assert session.reviewed_revision == (4 if expected_outcome == "ALREADY_APPLIED" else None)
    assert _payload(model.requests[1][-1])["outcome"] == expected_outcome


def test_removing_more_than_the_line_quantity_returns_positive_bounds() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="too-many",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "remove",
                "quantity": 3,
            },
        ),)),
        AssistantMessage(content="How many should I remove?"),
    )
    original = _burger(quantity=2)
    session = Session(lines=[original], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Remove three burgers",
    )

    result = _payload(model.requests[1][-1])
    assert result["outcome"] == "INCOMPLETE"
    assert result["key"] == "quantity"
    assert result["constraint"] == {"minimum": 1, "maximum": 2}
    assert session.lines == [original]
    assert session.revision == 1


def test_remove_item_removes_only_the_validated_line() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="remove-line",
            name="remove_item",
            arguments={"target": {"type": "line", "line_id": "L1"}},
        ),)),
        AssistantMessage(content="I removed the plain burgers."),
    )
    remaining = _burger(line_id="L2", extras=("cheese",), instructions="no onions")
    session = Session(
        lines=[_burger(quantity=2), remaining],
        instructions="leave at reception",
        next_line_number=3,
        revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Remove the plain burger line",
    )

    assert session.lines == [remaining]
    assert session.instructions == "leave at reception"
    assert session.revision == 3
    result = _payload(model.requests[1][-1])
    assert result["effect"]["operation"] == "remove_item"
    assert result["effect"]["removed_line_ids"] == ["L1"]


def test_ambiguous_removal_returns_matching_line_facts_without_guessing() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="ambiguous-removal",
            name="remove_item",
            arguments={"target": {"type": "match", "item_id": "classic_burger"}},
        ),)),
        AssistantMessage(content="Which burger selection should I remove?"),
    )
    original = [
        _burger(instructions="no onions"),
        _burger(line_id="L2", extras=("cheese",)),
    ]
    session = Session(lines=list(original), next_line_number=3, revision=2)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Remove the burger",
    )

    result = _payload(model.requests[1][-1])
    assert result["outcome"] == "INCOMPLETE"
    assert result["key"] == "target"
    assert [choice["value"] for choice in result["alternatives"]] == ["L1", "L2"]
    assert "no onions" in result["alternatives"][0]["label"]
    assert "cheese" in result["alternatives"][1]["label"]
    assert session.lines == original
    assert session.revision == 2


def test_clear_draft_removes_all_lines_and_general_instructions_explicitly() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="clear",
            name="clear_draft",
            arguments={},
        ),)),
        AssistantMessage(content="I cleared the draft."),
    )
    session = Session(
        lines=[_burger(), _burger(line_id="L2", instructions="no onions")],
        instructions="leave at reception",
        next_line_number=3,
        revision=2,
        reviewed_revision=2,
    )

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Clear my draft",
    )

    assert session.lines == []
    assert session.instructions == ""
    assert session.next_line_number == 3
    assert session.revision == 3
    assert session.reviewed_revision is None
    result = _payload(model.requests[1][-1])
    assert result["effect"]["operation"] == "clear_draft"
    assert result["effect"]["removed_line_ids"] == ["L1", "L2"]


def test_general_instructions_can_be_set_replaced_and_cleared_without_item_changes() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(
                call_id="set-notes",
                name="set_order_instructions",
                arguments={"instructions": "  no cutlery  "},
            ),
            ToolCall(
                call_id="replace-notes",
                name="set_order_instructions",
                arguments={"instructions": "ring the bell"},
            ),
            ToolCall(
                call_id="clear-notes",
                name="set_order_instructions",
                arguments={"instructions": ""},
            ),
        )),
        AssistantMessage(content="I cleared the general instructions."),
    )
    item_line = _burger(instructions="no onions")
    session = Session(lines=[item_line], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Set no cutlery, replace that with ring the bell, then clear the general note",
    )

    assert session.instructions == ""
    assert session.lines == [item_line]
    assert session.revision == 4
    results = [_payload(message) for message in model.requests[1][-3:]]
    assert [result["outcome"] for result in results] == ["APPLIED"] * 3
    assert [result["draft"]["general_instructions"] for result in results] == [
        "no cutlery",
        "ring the bell",
        "",
    ]


def test_clear_and_instruction_no_ops_preserve_revision_and_review_eligibility() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(
            ToolCall(call_id="clear-empty", name="clear_draft", arguments={}),
            ToolCall(
                call_id="same-notes",
                name="set_order_instructions",
                arguments={"instructions": ""},
            ),
        )),
        AssistantMessage(content="The draft is already clear."),
    )
    session = Session(revision=3, reviewed_revision=3)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Clear everything and remove any general instructions",
    )

    assert session.revision == 3
    assert session.reviewed_revision == 3
    results = [_payload(message) for message in model.requests[1][-2:]]
    assert [result["outcome"] for result in results] == ["ALREADY_APPLIED"] * 2
    assert [result["operation"] for result in results] == [
        "clear_draft",
        "set_order_instructions",
    ]


def test_zero_total_is_malformed_and_never_becomes_a_serving_subset_edit() -> None:
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(
            call_id="zero-total",
            name="change_quantity",
            arguments={
                "target": {"type": "line", "line_id": "L1"},
                "mode": "set",
                "quantity": 0,
            },
        ),)),
        AssistantMessage(content="Use remove_item to remove the line."),
    )
    original = _burger(quantity=4)
    session = Session(lines=[original], next_line_number=2, revision=1)

    TurnProcessor(model=model, menu=load_menu(), session=session).process(
        "Make the burger quantity zero",
    )

    result = _payload(model.requests[1][-1])
    assert result["outcome"] == "MALFORMED"
    assert result["tool_name"] == "change_quantity"
    assert any(issue["path"] == ["quantity"] for issue in result["issues"])
    assert session.lines == [original]
    assert session.revision == 1
