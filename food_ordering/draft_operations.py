"""Pure validators for typed Draft order operations."""

from food_ordering.menu import ALIASES, Menu
from food_ordering.order import OrderLine
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
    return [Alternative(value=value, label=label) for value, label in values]


def validate_add_item(
    operation: AddItem,
    draft: DraftState,
    menu: Menu,
) -> ValidationOutcome:
    """Validate one addition against a Draft candidate without mutating Session."""

    item_id = ALIASES.get(operation.item_id, operation.item_id)
    item = next((candidate for candidate in menu.menu if candidate.id == item_id), None)
    if item is None:
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

    unsupported_option = next(
        (name for name in operation.options if name not in item.options),
        None,
    )
    if unsupported_option is not None:
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{item.name} does not support the {unsupported_option} option.",
            resolution=f"Choose a supported option for {item.name}.",
            subject=item.name,
            key=unsupported_option,
            alternatives=_alternatives([(name, name) for name in item.options]),
        )

    normalized_options: dict[str, str] = {}
    unit_cents = item.base_price
    for name, option in item.options.items():
        value = operation.options.get(name, option.default)
        if value is None:
            if option.required:
                return Incomplete(
                    outcome="INCOMPLETE",
                    remedy="ask_customer",
                    reason=f"{item.name} requires a {name}.",
                    resolution=f"Ask the customer to choose a {name}.",
                    subject=item.name,
                    key=name,
                    alternatives=_alternatives([(choice, choice) for choice in option.choices]),
                )
            continue
        if value not in option.choices:
            return Incomplete(
                outcome="INCOMPLETE",
                remedy="ask_customer",
                reason=f"{value} is not a supported {name} for {item.name}.",
                resolution=f"Ask the customer to choose a supported {name}.",
                subject=item.name,
                key=name,
                alternatives=_alternatives([(choice, choice) for choice in option.choices]),
            )
        normalized_options[name] = value
        unit_cents += option.price_modifier.get(value, 0)

    extra_prices = {extra.id: extra.price for extra in item.extras.choices}
    unsupported_extra = next(
        (extra for extra in operation.extras if extra not in extra_prices),
        None,
    )
    if unsupported_extra is not None:
        return Unsatisfiable(
            outcome="UNSATISFIABLE",
            remedy="change_request",
            reason=f"{item.name} does not support the {unsupported_extra} extra.",
            resolution=f"Choose a supported extra for {item.name}.",
            subject=item.name,
            key="extras",
            alternatives=_alternatives([
                (extra.id, extra.id) for extra in item.extras.choices
            ]),
        )

    extras = tuple(sorted(set(operation.extras)))
    unit_cents += sum(extra_prices[extra] for extra in extras)
    line_id = f"L{draft.next_line_number}"
    line = OrderLine(
        item_id=item.id,
        name=item.name,
        quantity=operation.quantity,
        options=tuple(normalized_options.items()),
        extras=extras,
        unit_cents=unit_cents,
        line_id=line_id,
        instructions=operation.instructions.strip(),
    )
    return Valid(
        candidate=DraftState(
            lines=(*draft.lines, line),
            general_instructions=draft.general_instructions,
            next_line_number=draft.next_line_number + 1,
        ),
        effect=OperationEffect(operation="add_item", created_line_ids=(line_id,)),
        changed=True,
    )
