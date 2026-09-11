"""Pure validators for typed Draft order operations."""

from collections.abc import Callable
from dataclasses import replace
from typing import assert_never

from food_ordering.menu import ALIASES, Menu
from food_ordering.order import (
    AddSelectionIssue,
    MissingRequiredOption,
    OrderLine,
    UnsupportedExtra,
    UnsupportedInstructionAddition,
    UnsupportedItem,
    UnsupportedOption,
    UnsupportedOptionValue,
    describe_line,
    normalize_add_selection,
)
from food_ordering.tool_protocol import (
    AddItem,
    Alternative,
    ChangeQuantity,
    ClearDraft,
    CustomizeServings,
    DraftState,
    Incomplete,
    OperationEffect,
    RemoveItem,
    SetOrderInstructions,
    ReplaceServings,
    Unsatisfiable,
    UpdateItem,
    ValidationOutcome,
    Valid,
    ValueConstraint,
)


def _alternatives(values: list[tuple[str, str]]) -> list[Alternative]:
    return [
        Alternative(value=value, label=label.replace("_", " "))
        for value, label in values
    ]


def _selection_issue_outcome(
    issue: AddSelectionIssue,
    menu: Menu,
) -> Incomplete | Unsatisfiable:
    if isinstance(issue, UnsupportedItem):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="The requested item is not on the Menu.",
            resolution="Choose an item listed in the Menu.",
            subject=issue.item_id,
            key="item_id",
            alternatives=_alternatives([
                (candidate.id, candidate.name) for candidate in menu.menu
            ]),
        )
    if isinstance(issue, UnsupportedOption):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{issue.item_name} does not support the {issue.key} option.",
            resolution=f"Choose a supported option for {issue.item_name}.",
            subject=issue.item_name,
            key=issue.key,
            alternatives=_alternatives([(name, name) for name in issue.alternatives]),
        )
    if isinstance(issue, (MissingRequiredOption, UnsupportedOptionValue)):
        return Incomplete(
            outcome="INCOMPLETE",
            remedy="ask_customer",
            reason=(
                f"{issue.item_name} requires a {issue.key}."
                if isinstance(issue, MissingRequiredOption)
                else f"{issue.value} is not a supported {issue.key} for {issue.item_name}."
            ),
            resolution=(
                f"Ask the customer to choose a {issue.key}."
                if isinstance(issue, MissingRequiredOption)
                else f"Ask the customer to choose a supported {issue.key}."
            ),
            subject=issue.item_name,
            key=issue.key,
            alternatives=_alternatives([
                (choice, choice) for choice in issue.alternatives
            ]),
        )
    if isinstance(issue, UnsupportedExtra):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{issue.item_name} does not support the {issue.value} extra.",
            resolution=f"Choose a supported extra for {issue.item_name}.",
            subject=issue.item_name,
            key="extras",
            alternatives=_alternatives([(extra, extra) for extra in issue.alternatives]),
        )
    if isinstance(issue, UnsupportedInstructionAddition):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="An addition cannot be added through Special instructions.",
            resolution="Select the Extra explicitly when it is supported by the item.",
            subject=issue.item_name,
            key="instructions",
            alternatives=_alternatives([(extra, extra) for extra in issue.alternatives]),
            note=f"Instruction requested the unselected addition {issue.value}.",
        )
    assert_never(issue)


def validate_add_item(
    operation: AddItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate one addition against a Draft candidate without mutating Session."""

    line_id = f"L{draft.next_line_number}"
    normalized = normalize_add_selection(operation, menu, line_id=line_id)
    if not isinstance(normalized, OrderLine):
        return _selection_issue_outcome(normalized, menu)
    return Valid(
        candidate=DraftState(
            lines=(*draft.lines, normalized),
            general_instructions=draft.general_instructions,
            next_line_number=draft.next_line_number + 1,
        ),
        effect=OperationEffect(operation="add_item", created_line_ids=(line_id,)),
        changed=True,
    )


def validate_change_quantity(
    operation: ChangeQuantity,
    draft: DraftState,
) -> ValidationOutcome:
    """Validate a total quantity change without treating it as customization."""

    matches = _matching_lines(operation, draft)
    if not matches:
        return _missing_target()
    if len(matches) > 1:
        return _ambiguous_target(matches, action="change the quantity of")
    line = matches[0]
    candidate = list(draft.lines)
    if operation.mode == "remove" and operation.quantity > line.quantity:
        return Incomplete(
            outcome="INCOMPLETE",
            remedy="ask_customer",
            reason=f"Only {line.quantity} servings are selected on that Order line.",
            resolution=(
                f"Ask the customer to choose from 1 to {line.quantity} servings to remove."
            ),
            subject=line.name,
            key="quantity",
            constraint=ValueConstraint(minimum=1, maximum=line.quantity),
        )
    quantity = {
        "set": operation.quantity,
        "increase": line.quantity + operation.quantity,
        "remove": line.quantity - operation.quantity,
    }[operation.mode]
    if quantity == 0:
        candidate.remove(line)
        affected_line_ids: tuple[str, ...] = ()
        removed_line_ids: tuple[str, ...] = (line.line_id,)
    else:
        updated = replace(line, quantity=quantity)
        candidate[candidate.index(line)] = updated
        affected_line_ids = (line.line_id,)
        removed_line_ids = ()
    return Valid(
        candidate=replace(draft, lines=tuple(candidate)),
        effect=OperationEffect(
            operation="change_quantity",
            affected_line_ids=affected_line_ids,
            removed_line_ids=removed_line_ids,
        ),
        changed=quantity != line.quantity,
    )


def validate_remove_item(
    operation: RemoveItem,
    draft: DraftState,
) -> ValidationOutcome:
    """Validate removal of one uniquely targeted Order line."""

    matches = _matching_lines(operation, draft)
    if not matches:
        return _missing_target()
    if len(matches) > 1:
        return _ambiguous_target(matches, action="remove")
    line = matches[0]
    candidate = list(draft.lines)
    candidate.remove(line)
    return Valid(
        candidate=replace(draft, lines=tuple(candidate)),
        effect=OperationEffect(
            operation="remove_item",
            removed_line_ids=(line.line_id,),
        ),
        changed=True,
    )


def validate_clear_draft(
    operation: ClearDraft,
    draft: DraftState,
) -> ValidationOutcome:
    """Validate explicitly clearing every Draft-order field."""

    del operation
    return Valid(
        candidate=replace(draft, lines=(), general_instructions=""),
        effect=OperationEffect(
            operation="clear_draft",
            removed_line_ids=tuple(line.line_id for line in draft.lines),
        ),
        changed=bool(draft.lines or draft.general_instructions),
    )


def validate_set_order_instructions(
    operation: SetOrderInstructions,
    draft: DraftState,
) -> ValidationOutcome:
    """Validate replacement of general instructions independently of item notes."""

    instructions = operation.instructions.strip()
    return Valid(
        candidate=replace(draft, general_instructions=instructions),
        effect=OperationEffect(operation="set_order_instructions"),
        changed=instructions != draft.general_instructions,
    )


def _matching_lines(
    operation: UpdateItem | ChangeQuantity | RemoveItem,
    draft: DraftState,
) -> list[OrderLine]:
    target = operation.target
    if target.type == "line":
        return [line for line in draft.lines if line.line_id == target.line_id]
    item_id = ALIASES.get(target.item_id, target.item_id)
    return [
        line
        for line in draft.lines
        if line.item_id == item_id
        and target.options.items() <= dict(line.options).items()
        and set(target.extras) <= set(line.extras)
        and (target.instructions is None or target.instructions == line.instructions)
    ]


def _missing_target() -> Unsatisfiable:
    return Unsatisfiable(
        outcome="UNSATISFIABLE",
        remedy="change_request",
        reason="No Order line matches that target in the Draft order.",
        resolution="Choose an existing Order line from the Draft order.",
        key="target",
    )


def _ambiguous_target(matches: list[OrderLine], *, action: str) -> Incomplete:
    return Incomplete(
        outcome="INCOMPLETE",
        remedy="ask_customer",
        reason="More than one Order line matches that target.",
        resolution=f"Ask the customer to identify which matching Order line to {action}.",
        subject=matches[0].name,
        key="target",
        alternatives=[
            Alternative(
                value=line.line_id,
                label=(
                    describe_line(line)
                    + (f"; instructions: {line.instructions}" if line.instructions else "")
                ),
            )
            for line in matches
        ],
    )


def _validate_selected_servings(
    operation: UpdateItem,
    draft: DraftState,
    *,
    action: str,
) -> tuple[list[OrderLine], int] | Incomplete | Unsatisfiable:
    matches = _matching_lines(operation, draft)
    if not matches:
        return _missing_target()
    if operation.servings is None and len(matches) > 1:
        return _ambiguous_target(matches, action=action)

    available = sum(line.quantity for line in matches)
    selected = available if operation.servings == "all" else operation.servings
    if selected is None:
        selected = matches[0].quantity
    if selected > available:
        return Incomplete(
            outcome="INCOMPLETE",
            remedy="ask_customer",
            reason=f"Only {available} matching servings are selected.",
            resolution=(
                f"Ask the customer to choose from 1 to {available} matching servings."
            ),
            subject=matches[0].name,
            key="servings",
            constraint=ValueConstraint(minimum=1, maximum=available),
        )
    return matches, selected


def _apply_to_selected_servings(
    draft: DraftState,
    selection: tuple[list[OrderLine], int],
    transform: Callable[[OrderLine, int], OrderLine | Incomplete | Unsatisfiable],
) -> ValidationOutcome:
    matches, selected = selection
    remaining = selected
    candidate = list(draft.lines)
    next_line_number = draft.next_line_number
    affected_line_ids: list[str] = []
    created_line_ids: list[str] = []
    changed = False
    for line in matches:
        if remaining == 0:
            break
        quantity = min(remaining, line.quantity)
        normalized = transform(line, quantity)
        if not isinstance(normalized, OrderLine):
            return normalized
        line_changed = normalized != replace(line, quantity=quantity)
        index = candidate.index(line)
        if quantity < line.quantity and line_changed:
            new_line_id = f"L{next_line_number}"
            next_line_number += 1
            normalized = replace(normalized, line_id=new_line_id)
            candidate[index:index + 1] = [
                replace(line, quantity=line.quantity - quantity),
                normalized,
            ]
            affected_line_ids.append(line.line_id)
            created_line_ids.append(new_line_id)
        elif line_changed:
            candidate[index] = normalized
            affected_line_ids.append(line.line_id)
        changed = changed or line_changed
        remaining -= quantity

    return Valid(
        candidate=DraftState(
            lines=tuple(candidate),
            general_instructions=draft.general_instructions,
            next_line_number=next_line_number,
        ),
        effect=OperationEffect(
            operation="update_item",
            affected_line_ids=tuple(affected_line_ids),
            created_line_ids=tuple(created_line_ids),
        ),
        changed=changed,
    )


def validate_customize_item(
    operation: UpdateItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate deterministic serving customization without mutating Session."""

    assert isinstance(operation.change, CustomizeServings)
    selection = _validate_selected_servings(operation, draft, action="customize")
    if not isinstance(selection, tuple):
        return selection
    change = operation.change
    matches, _ = selection
    overlapping_extras = set(change.add_extras) & set(change.remove_extras)
    if overlapping_extras:
        extra = sorted(overlapping_extras)[0]
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"The {extra} extra cannot be added and removed together.",
            resolution="Choose whether to add or remove that extra.",
            subject=matches[0].name,
            key="extras",
        )

    def customize(
        line: OrderLine,
        quantity: int,
    ) -> OrderLine | Incomplete | Unsatisfiable:
        missing_extra = next(
            (extra for extra in change.remove_extras if extra not in line.extras),
            None,
        )
        if missing_extra is not None:
            return Unsatisfiable(
                outcome="UNSATISFIABLE",
                remedy="change_request",
                reason=f"The {missing_extra} extra is not selected on that Order line.",
                resolution="Choose an extra currently selected on the matching servings.",
                subject=line.name,
                key="extras",
                alternatives=_alternatives([(extra, extra) for extra in line.extras]),
            )
        normalized = normalize_add_selection(
            AddItem(
                item_id=line.item_id,
                quantity=quantity,
                options={**dict(line.options), **change.options},
                extras=sorted(
                    (set(line.extras) - set(change.remove_extras))
                    | set(change.add_extras)
                ),
                instructions=(
                    line.instructions
                    if change.instructions is None
                    else change.instructions
                ),
            ),
            menu,
            line_id=line.line_id,
        )
        if not isinstance(normalized, OrderLine):
            return _selection_issue_outcome(normalized, menu)
        return normalized

    return _apply_to_selected_servings(draft, selection, customize)


def _replacement_issue_outcome(
    issue: AddSelectionIssue,
    menu: Menu,
) -> Incomplete | Unsatisfiable:
    if isinstance(issue, UnsupportedOptionValue):
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=(
                f"{issue.value} is not a supported {issue.key} for {issue.item_name}."
            ),
            resolution=f"Choose a supported {issue.key} for {issue.item_name}.",
            subject=issue.item_name,
            key=issue.key,
            alternatives=_alternatives([
                (choice, choice) for choice in issue.alternatives
            ]),
        )
    return _selection_issue_outcome(issue, menu)


def validate_replace_item(
    operation: UpdateItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate deterministic product replacement without mutating Session."""

    assert isinstance(operation.change, ReplaceServings)
    selection = _validate_selected_servings(operation, draft, action="replace")
    if not isinstance(selection, tuple):
        return selection
    change = operation.change

    def replace_serving(
        line: OrderLine,
        quantity: int,
    ) -> OrderLine | Incomplete | Unsatisfiable:
        normalized = normalize_add_selection(
            AddItem(
                item_id=change.item_id,
                quantity=quantity,
                options=change.options,
                extras=change.extras,
                instructions=change.instructions,
            ),
            menu,
            line_id=line.line_id,
        )
        if not isinstance(normalized, OrderLine):
            return _replacement_issue_outcome(normalized, menu)
        return normalized

    return _apply_to_selected_servings(draft, selection, replace_serving)
