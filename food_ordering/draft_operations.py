"""Pure validators for typed Draft order operations."""

import re

from food_ordering.menu import Menu
from food_ordering.order import AddSelectionIssue, OrderLine, normalize_add_selection
from food_ordering.tool_protocol import (
    AddItem,
    Alternative,
    DraftState,
    Incomplete,
    OperationEffect,
    Unsatisfiable,
    ValidationOutcome,
    Valid,
)


def _alternatives(values: list[tuple[str, str]]) -> list[Alternative]:
    return [
        Alternative(value=value, label=label.replace("_", " "))
        for value, label in values
    ]


def validate_add_item(
    operation: AddItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate one addition against a Draft candidate without mutating Session."""

    line_id = f"L{draft.next_line_number}"
    normalized = normalize_add_selection(operation, menu, line_id=line_id)
    if isinstance(normalized, AddSelectionIssue) and normalized.kind == "item":
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="The requested item is not on the Menu.",
            resolution="Choose an item listed in the Menu.",
            subject=operation.item_id,
            key="item_id",
            alternatives=_alternatives([
                (candidate.id, candidate.name) for candidate in menu.menu
            ]),
        )
    if isinstance(normalized, AddSelectionIssue) and normalized.kind == "option_name":
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{normalized.item_name} does not support the {normalized.key} option.",
            resolution=f"Choose a supported option for {normalized.item_name}.",
            subject=normalized.item_name,
            key=normalized.key,
            alternatives=_alternatives([(name, name) for name in normalized.alternatives]),
        )
    if isinstance(normalized, AddSelectionIssue) and normalized.kind in {
        "required_option", "option_value",
    }:
        assert normalized.key is not None
        reason = (
            f"{normalized.item_name} requires a {normalized.key}."
            if normalized.kind == "required_option"
            else (
                f"{normalized.value} is not a supported {normalized.key} "
                f"for {normalized.item_name}."
            )
        )
        return Incomplete(
            outcome="INCOMPLETE",
            remedy="ask_customer",
            reason=reason,
            resolution=(
                f"Ask the customer to choose a {normalized.key}."
                if normalized.kind == "required_option"
                else f"Ask the customer to choose a supported {normalized.key}."
            ),
            subject=normalized.item_name,
            key=normalized.key,
            alternatives=_alternatives([
                (choice, choice) for choice in normalized.alternatives
            ]),
        )
    if isinstance(normalized, AddSelectionIssue):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{normalized.item_name} does not support the {normalized.value} extra.",
            resolution=f"Choose a supported extra for {normalized.item_name}.",
            subject=normalized.item_name,
            key="extras",
            alternatives=_alternatives([
                (extra, extra) for extra in normalized.alternatives
            ]),
        )

    assert isinstance(normalized, OrderLine)
    instruction_text = normalized.instructions.casefold().replace("_", " ")
    selected_extras = set(normalized.extras)
    named_unselected_extra = next(
        (
            extra.id
            for menu_item in menu.menu
            for extra in menu_item.extras.choices
            if extra.id not in selected_extras
            and re.search(
                rf"(?<!\w){re.escape(extra.id.casefold().replace('_', ' '))}(?!\w)",
                instruction_text,
            )
        ),
        None,
    )
    if named_unselected_extra is not None:
        supported_extras = next(
            item.extras.choices for item in menu.menu if item.id == normalized.item_id
        )
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="A Menu Extra cannot be added through Special instructions.",
            resolution="Select the Extra explicitly when it is supported by the item.",
            subject=normalized.name,
            key="instructions",
            alternatives=_alternatives([
                (extra.id, extra.id) for extra in supported_extras
            ]),
            note=f"Instruction named the unselected Extra {named_unselected_extra}.",
        )
    return Valid(
        candidate=DraftState(
            lines=(*draft.lines, normalized),
            general_instructions=draft.general_instructions,
            next_line_number=draft.next_line_number + 1,
        ),
        effect=OperationEffect(operation="add_item", created_line_ids=(line_id,)),
        changed=True,
    )
