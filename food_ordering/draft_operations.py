"""Pure validators for typed Draft order operations."""

from dataclasses import replace

from food_ordering.menu import ALIASES, Menu
from food_ordering.order import (
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
    CustomizeServings,
    DraftState,
    Incomplete,
    OperationEffect,
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


def _matching_lines(operation: UpdateItem, draft: DraftState) -> list[OrderLine]:
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


def validate_customize_item(
    operation: UpdateItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate deterministic serving customization without mutating Session."""

    assert isinstance(operation.change, CustomizeServings)
    matches = _matching_lines(operation, draft)
    if not matches:
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason="No Order line matches that target in the Draft order.",
            resolution="Choose an existing Order line from the Draft order.",
            key="target",
        )
    if operation.servings is None and len(matches) > 1:
        return Incomplete(
            outcome="INCOMPLETE",
            remedy="ask_customer",
            reason="More than one Order line matches that target.",
            resolution="Ask the customer to identify which matching Order line to customize.",
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
    change = operation.change
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
        if isinstance(normalized, UnsupportedItem):
            return Unsatisfiable(
                outcome="UNSATISFIABLE",
                remedy="change_request",
                reason="The targeted item is no longer on the Menu.",
                resolution="Choose an item listed in the Menu.",
                subject=normalized.item_id,
                key="item_id",
                alternatives=_alternatives([
                    (candidate.id, candidate.name) for candidate in menu.menu
                ]),
            )
        if isinstance(normalized, UnsupportedOption):
            return Unsatisfiable(
                outcome="UNSATISFIABLE",
                remedy="change_request",
                reason=(
                    f"{normalized.item_name} does not support the {normalized.key} option."
                ),
                resolution=f"Choose a supported option for {normalized.item_name}.",
                subject=normalized.item_name,
                key=normalized.key,
                alternatives=_alternatives([
                    (name, name) for name in normalized.alternatives
                ]),
            )
        if isinstance(normalized, (MissingRequiredOption, UnsupportedOptionValue)):
            return Incomplete(
                outcome="INCOMPLETE",
                remedy="ask_customer",
                reason=(
                    f"{normalized.item_name} requires a {normalized.key}."
                    if isinstance(normalized, MissingRequiredOption)
                    else (
                        f"{normalized.value} is not a supported {normalized.key} "
                        f"for {normalized.item_name}."
                    )
                ),
                resolution=f"Ask the customer to choose a supported {normalized.key}.",
                subject=normalized.item_name,
                key=normalized.key,
                alternatives=_alternatives([
                    (value, value) for value in normalized.alternatives
                ]),
            )
        if isinstance(normalized, UnsupportedExtra):
            return Unsatisfiable(
                outcome="UNSATISFIABLE",
                remedy="change_request",
                reason=(
                    f"{normalized.item_name} does not support the {normalized.value} extra."
                ),
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
