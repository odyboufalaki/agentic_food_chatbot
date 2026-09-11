"""Pure validators for typed Draft order operations."""

from food_ordering.menu import Menu
from food_ordering.order import (
    MissingRequiredOption,
    OrderLine,
    UnsupportedExtra,
    UnsupportedInstructionAddition,
    UnsupportedItem,
    UnsupportedOption,
    UnsupportedOptionValue,
    normalize_add_selection,
)
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
    if isinstance(normalized, UnsupportedItem):
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
    if isinstance(normalized, UnsupportedOption):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{normalized.item_name} does not support the {normalized.key} option.",
            resolution=f"Choose a supported option for {normalized.item_name}.",
            subject=normalized.item_name,
            key=normalized.key,
            alternatives=_alternatives([(name, name) for name in normalized.alternatives]),
        )
    if isinstance(normalized, (MissingRequiredOption, UnsupportedOptionValue)):
        reason = (
            f"{normalized.item_name} requires a {normalized.key}."
            if isinstance(normalized, MissingRequiredOption)
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
                if isinstance(normalized, MissingRequiredOption)
                else f"Ask the customer to choose a supported {normalized.key}."
            ),
            subject=normalized.item_name,
            key=normalized.key,
            alternatives=_alternatives([
                (choice, choice) for choice in normalized.alternatives
            ]),
        )
    if isinstance(normalized, UnsupportedExtra):
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

    if isinstance(normalized, UnsupportedInstructionAddition):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="An addition cannot be added through Special instructions.",
            resolution="Select the Extra explicitly when it is supported by the item.",
            subject=normalized.item_name,
            key="instructions",
            alternatives=_alternatives([
                (extra, extra) for extra in normalized.alternatives
            ]),
            note=f"Instruction requested the unselected addition {normalized.value}.",
        )
    assert isinstance(normalized, OrderLine)
    return Valid(
        candidate=DraftState(
            lines=(*draft.lines, normalized),
            general_instructions=draft.general_instructions,
            next_line_number=draft.next_line_number + 1,
        ),
        effect=OperationEffect(operation="add_item", created_line_ids=(line_id,)),
        changed=True,
    )
